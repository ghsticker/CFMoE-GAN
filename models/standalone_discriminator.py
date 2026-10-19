"""
Standalone Discriminator from TransGAN (Celeba256 version).
Based on: https://github.com/VITA-Group/TransGAN

A pure-Transformer discriminator that processes 256x256 images using
multi-scale patch embedding + window-based self-attention.

Usage:
    from transgan_discriminator import TransGANDiscriminator

    # Simple config (no DiffAugment needed)
    disc = TransGANDiscriminator(img_size=256, embed_dim=384, depth=7,
                                  num_heads=4, patch_size=4, window_size=8)

    score = disc(images)  # (B, 1) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from copy import deepcopy

# ============================================================================
# Section 1: ViT Helper (DropPath, trunc_normal_, to_2tuple)
# ============================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _ntuple(n):
    def parse(x):
        if isinstance(x, (list, tuple)):
            return x
        return tuple([x] * n)
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        import warnings
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ============================================================================
# Section 2: ADA Matrix Helpers (for DiffAugment, optional)
# ============================================================================

_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format
    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x if isinstance(x, torch.Tensor) else constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))


def translate2d(tx, ty, **kwargs):
    return matrix([1, 0, tx], [0, 1, ty], [0, 0, 1], **kwargs)


def scale2d(sx, sy, **kwargs):
    return matrix([sx, 0, 0], [0, sy, 0], [0, 0, 1], **kwargs)


def translate2d_inv(tx, ty, **kwargs):
    return translate2d(-tx, -ty, **kwargs)


def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)


# ============================================================================
# Section 3: Wavelet Filter Bank (for DiffAugment filter augmentation)
# ============================================================================

_wavelets = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1':  [0.7071067811865476, 0.7071067811865476],
    'db2':  [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'db3':  [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'db4':  [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
    'db5':  [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
    'db6':  [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
    'db7':  [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
    'db8':  [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
}


# ============================================================================
# Section 4: Differentiable Augmentation (DiffAugment, optional)
# If you don't need data augmentation, pass `diff_aug=None` and use `aug=False`
# ============================================================================

def DiffAugment(x, policy='', channels_first=True, affine=None):
    """Differentiable Augmentation from StyleGAN-ADA.
    Policy examples: 'color,translation', 'translation,filter,geo', 'None'
    """
    if not policy or policy == 'None':
        return x
    if not channels_first:
        x = x.permute(0, 3, 1, 2)
    for p in policy.split(','):
        if p in AUGMENT_FNS:
            for f in AUGMENT_FNS[p]:
                x = f(x, affine=affine)
    if not channels_first:
        x = x.permute(0, 2, 3, 1)
    return x.contiguous()


def rand_brightness(x, affine=None):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def rand_saturation(x, affine=None):
    x_mean = x.mean(dim=1, keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean


def rand_contrast(x, affine=None):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean


def rand_translation(x, ratio=0.2, affine=None):
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device), indexing='ij')
    grid_x = torch.clamp(grid_x + tx + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + ty + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5, affine=None):
    import random
    if random.random() < 0.3:
        cutout_h = int(x.size(2) * ratio + 0.5)
        cutout_w = int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_h % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_w % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_h, dtype=torch.long, device=x.device),
            torch.arange(cutout_w, dtype=torch.long, device=x.device), indexing='ij')
        grid_x = torch.clamp(grid_x + offset_x - cutout_h // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_w // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
    return x


def rand_geo(x, affine=None):
    """Geometric augmentation using affine transformations (simplified)."""
    batch_size, num_channels, height, width = x.size()
    device = x.device
    import random
    I_3 = torch.eye(3, device=device)
    G_inv = I_3

    # x-flip
    if random.random() < 0.5:
        G_inv = G_inv @ scale2d_inv(-1, 1)

    # isotropic scaling
    s = torch.exp2(torch.randn([batch_size], device=device) * 0.2)
    G_inv = G_inv @ scale2d_inv(s, s)

    # fractional translation
    t = torch.randn([batch_size, 2], device=device) * 0.125
    G_inv = G_inv @ translate2d_inv(t[:, 0] * width, t[:, 1] * height)

    if G_inv is not I_3:
        G_inv = scale2d(2 / width, 2 / height) @ G_inv @ scale2d_inv(2 / x.shape[3], 2 / x.shape[2])
        grid = F.affine_grid(G_inv[:, :2, :].unsqueeze(0).expand(batch_size, -1, -1),
                              [batch_size, num_channels, height, width],
                              align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
    return x


AUGMENT_FNS = {
    'color':       [rand_brightness, rand_saturation, rand_contrast],
    'translation':  [rand_translation],
    'cutout':      [rand_cutout],
    'geo':         [rand_geo],
}


# ============================================================================
# Section 5: Core Transformer Building Blocks
# ============================================================================

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)


class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu
        else:
            self.act_layer = gelu

    def forward(self, x):
        return self.act_layer(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., is_mask=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_mask = is_mask

    def forward(self, x):
        B, N, C = x.shape
        if self.is_mask == 1:
            H = W = int(math.sqrt(N))
            image = x.view(B, H, W, C).view(B * H, W, C)
            qkv = self.qkv(image).reshape(B * H, W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B * H, W, C).view(B, H, W, C).view(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        elif self.is_mask == 2:
            H = W = int(math.sqrt(N))
            image = x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B * W, H, C)
            qkv = self.qkv(image).reshape(B * W, H, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B * W, H, C).view(B, W, H, C).permute(0, 2, 1, 3).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)

    def forward(self, x):
        if self.norm_type in ("bn", "in"):
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x
        elif self.norm_type == "none":
            return x
        return self.norm(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DisBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=leakyrelu, norm_layer=nn.LayerNorm, separate=0):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, is_mask=separate)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gain = np.sqrt(0.5) if norm_layer == "none" else 1

    def forward(self, x):
        x = x * self.gain + self.drop_path(self.attn(self.norm1(x))) * self.gain
        x = x * self.gain + self.drop_path(self.mlp(self.norm2(x))) * self.gain
        return x


# ============================================================================
# Section 6: TransGAN Discriminator
# ============================================================================

class TransGANDiscriminator(nn.Module):
    """
    A pure-Transformer discriminator for high-resolution image classification.

    Architecture:
        - Multi-scale patch embedding (fRGB_1/2/3 at 3 different stride scales)
        - Three stages of window-based Transformer blocks (DisBlock)
        - Final cls token → linear head

    Args:
        img_size:       Input image size (default 256 for CelebA-HQ)
        in_chans:       Number of input channels (default 3)
        num_classes:    Output classes (default 1 for GAN discriminator)
        embed_dim:      Base embedding dimension (default 384)
        depth:          Number of DisBlocks per stage (default 7)
        num_heads:      Number of attention heads (default 4)
        mlp_ratio:      MLP hidden dim ratio (default 4.)
        patch_size:     Patch embedding stride (default 4)
        window_size:    Window attention window size (default 8)
        norm_layer:     Normalization type: "ln", "bn", "in", "pn", "none" (default "ln")
        act_layer:      Activation: "gelu" or "leakyrelu" (default "leakyrelu")
        diff_aug:       Differentiable augmentation policy, e.g. "color,translation,cutout"
                        or None / "None" to disable (default None)
        drop_rate:      Dropout rate (default 0.)
        attn_drop_rate: Attention dropout rate (default 0.)
    """

    def __init__(self, img_size=256, in_chans=3, num_classes=1, embed_dim=384, depth=7,
                 num_heads=4, mlp_ratio=4., patch_size=4, window_size=8,
                 norm_layer="ln", act_layer="leakyrelu", diff_aug=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 qkv_bias=False, qk_scale=None):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.diff_aug = diff_aug if diff_aug else "None"

        # ---- Multi-scale patch embedding (fRGB) ----
        # Stage 1: stride = patch_size, output dim = embed_dim // 4
        self.fRGB_1 = nn.Conv2d(in_chans, embed_dim // 4, kernel_size=patch_size * 2,
                                  stride=patch_size, padding=patch_size // 2)
        # Stage 2: stride = patch_size * 2, output dim = embed_dim // 4
        self.fRGB_2 = nn.Conv2d(in_chans, embed_dim // 4, kernel_size=patch_size * 2,
                                  stride=patch_size * 2, padding=0)
        # Stage 3: stride = patch_size * 4, output dim = embed_dim // 2
        self.fRGB_3 = nn.Conv2d(in_chans, embed_dim // 2, kernel_size=patch_size * 4,
                                  stride=patch_size * 4, padding=0)

        num_patches_1 = (img_size // patch_size) ** 2
        num_patches_2 = ((img_size // 2) // patch_size) ** 2
        num_patches_3 = ((img_size // 4) // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, num_patches_1, embed_dim // 4))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, num_patches_2, embed_dim // 2))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, num_patches_3, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # ---- Three-stage Transformer blocks ----
        self.blocks_1 = nn.ModuleList([
            DisBlock(dim=embed_dim // 4, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                     attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer,
                     norm_layer=norm_layer)
            for _ in range(depth)])

        self.blocks_2 = nn.ModuleList([
            DisBlock(dim=embed_dim // 2, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                     attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer,
                     norm_layer=norm_layer)
            for _ in range(depth - 1)])

        self.blocks_21 = nn.ModuleList([
            DisBlock(dim=embed_dim // 2, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                     attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer,
                     norm_layer=norm_layer)
            for _ in range(1)])

        self.blocks_3 = nn.ModuleList([
            DisBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                     attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer,
                     norm_layer=norm_layer)
            for _ in range(depth + 1)])

        self.last_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            act_layer=act_layer, norm_layer=norm_layer)

        self.norm = CustomNorm(norm_layer, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed_1, std=.02)
        trunc_normal_(self.pos_embed_2, std=.02)
        trunc_normal_(self.pos_embed_3, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # ---- DiffAugment filter bank setup (optional) ----
        if 'filter' in self.diff_aug:
            Hz_lo = np.asarray(_wavelets['sym2'])
            Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size))
            Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2
            Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2
            Hz_fbank = np.eye(4, 1)
            for i in range(1, Hz_fbank.shape[0]):
                Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
                Hz_fbank = np.convolve(Hz_fbank.flatten(), [Hz_lo2]).reshape(Hz_fbank.shape)
                Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2: (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
            self.register_buffer('Hz_fbank', torch.as_tensor(Hz_fbank, dtype=torch.float32))
        else:
            self.Hz_fbank = None

        if 'geo' in self.diff_aug:
            self.register_buffer('Hz_geom', torch.as_tensor(_wavelets['sym6'], dtype=torch.float32))
        else:
            self.Hz_geom = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, aug=True):
        # ---- Optional DiffAugment ----
        if "None" not in self.diff_aug and aug:
            x = DiffAugment(x, self.diff_aug, True, [self.Hz_geom, self.Hz_fbank])
        elif "None" not in self.diff_aug:
            x = DiffAugment(x, "translation", True, [self.Hz_geom, self.Hz_fbank])

        B, _, H, W = x.size()
        H = W = H // self.patch_size

        # ---- Three-scale patch embedding ----
        x_1 = self.fRGB_1(x).flatten(2).permute(0, 2, 1)
        x_2 = self.fRGB_2(x).flatten(2).permute(0, 2, 1)
        x_3 = self.fRGB_3(x).flatten(2).permute(0, 2, 1)

        # ---- Stage 1: fine-grained ----
        x = x_1 + self.pos_embed_1
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        for blk in self.blocks_1:
            x = blk(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)

        # ---- Stage 2: medium-grained ----
        _, _, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = nn.AvgPool2d(2)(x)
        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = torch.cat([x, x_2], dim=-1)
        x = x + self.pos_embed_2

        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        for blk in self.blocks_2:
            x = blk(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)
        for blk in self.blocks_21:
            x = blk(x)

        # ---- Stage 3: coarse-grained ----
        _, _, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = nn.AvgPool2d(2)(x)
        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = torch.cat([x, x_3], dim=-1)
        x = x + self.pos_embed_3

        for blk in self.blocks_3:
            x = blk(x)

        # ---- Classification head ----
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.last_block(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, aug=True):
        """
        Args:
            x:   Image tensor of shape (B, 3, img_size, img_size)
            aug: Whether to apply DiffAugment (default True)
        Returns:
            Logits of shape (B, 1) for GAN discriminator use
        """
        x = self.forward_features(x, aug=aug)
        x = self.head(x)
        return x


# ============================================================================
# Section 7: Convenience factory — build directly without args
# ============================================================================

def build_discriminator(img_size=256, embed_dim=384, depth=7, num_heads=4,
                       patch_size=4, window_size=8, num_classes=1,
                       norm_layer="ln", act_layer="leakyrelu", diff_aug=None,
                       drop_rate=0., **kwargs):
    """
    Build a TransGAN Discriminator without needing an `args` object.

    Default values match the CelebA-HQ 256x256 checkpoint from the paper.

    Example:
        disc = build_discriminator(img_size=256, embed_dim=384, depth=7)
        score = disc(images)       # GAN training
        score = disc(images, aug=False)  # no DiffAugment
    """
    return TransGANDiscriminator(
        img_size=img_size, in_chans=3, num_classes=num_classes,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        patch_size=patch_size, window_size=window_size,
        norm_layer=norm_layer, act_layer=act_layer,
        diff_aug=diff_aug, drop_rate=drop_rate, **kwargs
    )
