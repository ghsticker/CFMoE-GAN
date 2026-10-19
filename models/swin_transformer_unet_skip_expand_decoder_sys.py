import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class MoEFFNGating(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts):
        super(MoEFFNGating, self).__init__()
        self.gating_network = nn.Linear(dim, dim)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)) for _ in range(num_experts)])

    def forward(self, x):
        weights = self.gating_network(x)
        weights = torch.nn.functional.softmax(weights, dim=-1)
        outputs = [expert(x) for expert in self.experts]
        outputs = torch.stack(outputs, dim=0)
        outputs = (weights.unsqueeze(0) * outputs).sum(dim=0)
        return outputs


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _valid_group_count(channels, max_groups=8):
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class PPBFilter(nn.Module):
    def __init__(self, kernel_size=5, eps=1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = F.unfold(self.pad(x), kernel_size=self.kernel_size)
        patches = patches.view(B, C, self.kernel_size * self.kernel_size, H, W)
        center = x.unsqueeze(2)
        local_var = patches.var(dim=2, keepdim=True, unbiased=False)
        weights = torch.softmax(-((patches - center) ** 2) / (local_var + self.eps), dim=2)
        return (patches * weights).sum(dim=2)


class SoftPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, eps=1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = eps

    def forward(self, x):
        weights = torch.exp(torch.clamp(x, max=20.0))
        weighted = F.avg_pool2d(
            x * weights,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        normalizer = F.avg_pool2d(
            weights,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        return weighted / normalizer.clamp_min(self.eps)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.main = nn.Sequential(
            nn.GroupNorm(_valid_group_count(in_channels, groups), in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_valid_group_count(out_channels, groups), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class SCPEDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            SoftPool2d(kernel_size=2, stride=2),
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class SCPEPyramid(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        stem_dim = max(embed_dim // 4, 8)
        self.in_chans = in_chans
        # self.ppb = PPBFilter(kernel_size=5)
        self.stem = nn.Conv2d(3, stem_dim, kernel_size=3, padding=1)
        self.down1 = SCPEDownBlock(stem_dim, embed_dim // 2)
        self.down2 = SCPEDownBlock(embed_dim // 2, embed_dim)
        self.down3 = SCPEDownBlock(embed_dim, embed_dim * 2)
        self.down4 = SCPEDownBlock(embed_dim * 2, embed_dim * 4)

    # def _to_sar_intensity(self, x):
    #     if x.shape[1] == 1:
    #         return x
    #     return x.mean(dim=1, keepdim=True)

    def forward(self, x, return_all=False):
        # c0 = self.ppb(self._to_sar_intensity(x))
        c1 = self.stem(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        c5 = self.down4(c4)
        stage_priors = {0: c3, 1: c4, 2: c5}
        if return_all:
            return stage_priors, {"c2": c2, "c3": c3, "c4": c4, "c5": c5}
        return stage_priors


class SFTBridge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.semantic_norm = LayerNorm2d(dim)
        self.gamma = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )
        self.beta = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.gamma[-1].weight)
        nn.init.zeros_(self.gamma[-1].bias)
        nn.init.zeros_(self.beta[-1].weight)
        nn.init.zeros_(self.beta[-1].bias)

    def forward(self, semantic, prior):
        if prior.shape[-2:] != semantic.shape[-2:]:
            prior = F.interpolate(prior, size=semantic.shape[-2:], mode="bilinear", align_corners=False)
        gamma = self.gamma(prior)
        beta = self.beta(prior)
        return self.semantic_norm(semantic) * (1 + gamma) + beta


# class SharedExpert(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.pre = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
#             nn.GELU(),
#         )
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
#         self.proj = nn.Conv2d(dim, dim, kernel_size=1)
#         self.last_gates = None
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.weight_proj = nn.Conv2d(dim, dim, kernel_size=1)

#     def forward(self, x):
#         x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         feat = self.pre(x_norm)
#         q, k, v = torch.chunk(self.qkv(feat), 3, dim=1)
#         B, C, H, W = q.shape
#         q = q.view(B, C, -1)
#         k = k.view(B, C, -1)
#         v = v.view(B, C, -1)
#         scale = q.shape[-1] ** -0.5
#         attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * scale, dim=-1)
#         out = torch.matmul(attn, v).view(B, C, H, W)
#         out = self.proj(out)
#         out = out + x
#         weights = self.weight_proj(self.pool(out))
#         return weights

class SharedExpert(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SharedExpert, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



# 动态低高通分解
class dynamic_filter_channel(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter_channel, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.conv_gate = nn.Conv2d(group * kernel_size ** 2, group * kernel_size ** 2, kernel_size=1, stride=1,
                                   bias=False)
        self.act_gate = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        self.ap_1 = nn.AdaptiveAvgPool2d((1, 1))
        # self.ap_2 = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        identity_input = x
        low_filter1 = self.ap_1(x)
        # low_filter2 = self.ap_2(x)

        low_filter = self.conv(low_filter1)
        low_filter = low_filter * self.act_gate(self.conv_gate(low_filter))
        low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,
                                                                        self.kernel_size ** 2, h * w)

        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)

        low_filter = self.act(low_filter)
        # print('low_filter size',low_filter.shape)
        # print('low_filter n,c1,p,q',n,c1,p,q)

        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        return low_part, out_high

class ExpertGroup(nn.Module):
    def __init__(self, expert_factory, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([expert_factory() for _ in range(num_experts)])

    def forward(self, x, weights):
        # weights: [B, K, H, W]
        outputs = []
        for idx, expert in enumerate(self.experts):
            y = expert(x)
            outputs.append(weights[:, idx:idx + 1] * y)
        return torch.stack(outputs, dim=0).sum(dim=0)


class SpatialAwareRouter(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, num_experts, kernel_size=1)

    def forward(self, x):
        return torch.softmax(self.proj(self.dwconv(x)), dim=1)


class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = y.squeeze(-1).transpose(-1, -2)  # B, 1, C
        y = self.conv(y)
        y = self.act(y).transpose(-1, -2).unsqueeze(-1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mask = self.act(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return mask


class LowToHighExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv_d2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim)
        self.dwconv_d1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.act = nn.GELU()
        self.eca = EfficientChannelAttention(dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x = self.act(self.norm1(self.dwconv_d2(x)))
        x = self.act(self.norm2(self.dwconv_d1(x)))
        x = self.eca(x)
        x = self.pw(x)
        return x


class HighToHighExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.act = nn.GELU()
        self.spatial_attn = SpatialAttention(kernel_size=7)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        feat = self.act(self.norm(self.dwconv(x)))
        mask = self.spatial_attn(feat)
        out = mask * self.proj(x) + feat
        return out


class LowToLowExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x = self.act(self.norm(self.dwconv(x)))
        x = self.pw(x)
        return x


class HighToLowExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.act = nn.GELU()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x = self.act(self.norm(self.dwconv(x)))
        x = self.pool(x)
        x = self.pw(x)
        return x



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_cfmoe=False, route_dim=None,
                 num_experts_per_group=4, intra_topk=2, topk_renorm=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.use_cfmoe = use_cfmoe
        self.last_route_features = None
        if self.use_cfmoe:
            self.mlp = CFMoE(
                dim=dim,
                input_resolution=input_resolution,
                route_dim=route_dim,
                num_experts_per_group=num_experts_per_group,
                intra_topk=intra_topk,
                topk_renorm=topk_renorm,
            )
        else:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, route_x=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x_norm = self.norm2(x)
        if self.use_cfmoe:
            x = x + self.drop_path(self.mlp(x_norm, route_x=route_x))
            self.last_route_features = getattr(self.mlp, "last_route_features", None)
        else:
            x = x + self.drop_path(self.mlp(x_norm))
            self.last_route_features = None

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_cfmoe=False, route_dim=None, cfmoe_block_indices=None,
                 num_experts_per_group=4, intra_topk=2, topk_renorm=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_cfmoe = use_cfmoe

        if cfmoe_block_indices is not None:
            cfmoe_block_indices = set(cfmoe_block_indices)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_cfmoe=(use_cfmoe and (cfmoe_block_indices is None or i in cfmoe_block_indices)),
                                 route_dim=route_dim if (use_cfmoe and (cfmoe_block_indices is None or i in cfmoe_block_indices)) else None,
                                 num_experts_per_group=num_experts_per_group,
                                 intra_topk=intra_topk,
                                 topk_renorm=topk_renorm)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 use_cfmoe=False, route_dim=None,
                 num_experts_per_group=4, intra_topk=2, topk_renorm=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_cfmoe=use_cfmoe,
                                 route_dim=route_dim,
                                 num_experts_per_group=num_experts_per_group,
                                 intra_topk=intra_topk,
                                 topk_renorm=topk_renorm)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, route_x=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                if route_x is None:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = checkpoint.checkpoint(blk, x, route_x)
            else:
                x = blk(x, route_x=route_x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, out_chans=None,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", dino_route_dim=None,
                 num_experts_per_group=4, intra_topk=2, topk_renorm=False,
                 use_scpe_sft_skip=True, use_cfmoe=True, **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.out_chans = out_chans if out_chans is not None else num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.dino_route_dim = dino_route_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.use_scpe_sft_skip = use_scpe_sft_skip
        self.use_cfmoe = use_cfmoe
        self.last_scpe_priors = {}
        self.last_sft_skips = {}

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_scpe_sft_skip:
            self.scpe_pyramid = SCPEPyramid(in_chans=in_chans, embed_dim=embed_dim)
            self.sft_bridges = nn.ModuleList([
                SFTBridge(int(embed_dim * 2 ** stage_idx))
                for stage_idx in range(self.num_layers - 1)
            ])
        else:
            self.scpe_pyramid = None
            self.sft_bridges = nn.ModuleList()

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            is_bottleneck = i_layer == self.num_layers - 1
            # 这里是将bottleneck 增加了MoE的结构
            bottleneck_cfmoe_indices = {1} if is_bottleneck else None
            if is_bottleneck and depths[i_layer] <= 1:
                bottleneck_cfmoe_indices = {depths[i_layer] - 1}

            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               use_cfmoe=False,
                               route_dim=self.dino_route_dim if is_bottleneck else None,
                               cfmoe_block_indices=bottleneck_cfmoe_indices,
                               num_experts_per_group=num_experts_per_group,
                               intra_topk=intra_topk,
                               topk_renorm=topk_renorm)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.skip_fusion = nn.ModuleList()
        for i_layer in range(self.num_layers):
            stage_idx = self.num_layers - 1 - i_layer
            input_resolution = (
                patches_resolution[0] // (2 ** stage_idx),
                patches_resolution[1] // (2 ** stage_idx),
            )
            stage_dim = int(embed_dim * 2 ** stage_idx)
            concat_linear = nn.Linear(2 * stage_dim, stage_dim) if i_layer > 0 else nn.Identity()
            skip_fusion = nn.Conv2d(2 * stage_dim, stage_dim, kernel_size=1) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=input_resolution,
                    dim=stage_dim, dim_scale=2, norm_layer=norm_layer)
            else:
                # Enable CF-MoE only at decoder 1/16, 1/8, and 1/4 resolution stages.
                use_cfmoe = self.use_cfmoe and (stage_idx in [2, 1, 0])
                route_dim = self.dino_route_dim if use_cfmoe else None
                layer_up = BasicLayer_up(dim=stage_dim,
                                         input_resolution=input_resolution,
                                         depth=depths[stage_idx],
                                         num_heads=num_heads[stage_idx],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:stage_idx]):sum(depths[:stage_idx + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint,
                                         use_cfmoe=use_cfmoe,
                                         route_dim=route_dim,
                                         num_experts_per_group=num_experts_per_group,
                                         intra_topk=intra_topk,
                                         topk_renorm=topk_renorm)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
            self.skip_fusion.append(skip_fusion)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=embed_dim, out_channels=self.out_chans, kernel_size=1, bias=False),
                nn.Tanh(),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def _collect_layer_cfmoe_stats(self, layer_module):
        if not hasattr(layer_module, "blocks"):
            return

        for blk in layer_module.blocks:
            if hasattr(blk, "use_cfmoe") and blk.use_cfmoe:
                routing_maps = getattr(blk.mlp, "last_routing_maps", None)
                if routing_maps is not None:
                    self.last_routing_maps.append(routing_maps)
                route_feat = getattr(blk, "last_route_features", None)
                if route_feat is not None:
                    self.last_route_features.append(route_feat)

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x) # Patch Partition + Linear Embedding
        if self.ape:
            x = x + self.absolute_pos_embed #
        x = self.pos_drop(x) # Dropout + Position Embedding
        x_downsample = []

        for layer in self.layers: # 在layers里面就包含了Swin Transfomer Block + Patch Merging
            x_downsample.append(x)
            x = layer(x)
            self._collect_layer_cfmoe_stats(layer)

        x = self.norm(x)  # B L C

        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample, scpe_priors=None, return_features=False):
        decoder_feats = {}
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                stage_idx = self.num_layers - 1 - inx
                skip = x_downsample[stage_idx]
                B, L, C = x.shape
                H, W = layer_up.input_resolution
                if L != H * W:
                    raise ValueError(f"Decoder token length {L} mismatches resolution {(H, W)}.")
                if skip.shape[1] != L or skip.shape[2] != C:
                    raise ValueError("Encoder skip feature shape mismatches decoder feature shape.")

                if self.use_scpe_sft_skip:
                    if scpe_priors is None or stage_idx not in scpe_priors:
                        raise ValueError(f"SCPE prior for stage {stage_idx} is missing.")
                    skip_2d = skip.transpose(1, 2).contiguous().view(B, C, H, W)
                    prior = scpe_priors[stage_idx]
                    v_modulated = self.sft_bridges[stage_idx](skip_2d, prior)
                    x_2d = x.transpose(1, 2).contiguous().view(B, C, H, W)
                    x = self.skip_fusion[inx](torch.cat([x_2d, v_modulated], dim=1))
                    x = x.flatten(2).transpose(1, 2).contiguous()
                    route_x = v_modulated.flatten(2).transpose(1, 2).contiguous()
                    self.last_sft_skips[f"stage{stage_idx}"] = v_modulated
                    x = layer_up(x, route_x=route_x)
                else:
                    skip = skip.transpose(1, 2).reshape(B, C, H, W)
                    skip = skip.flatten(2).transpose(1, 2)
                    x = torch.cat([x, skip], -1)
                    x = self.concat_back_dim[inx](x)
                    x = layer_up(x)

            if return_features and inx in {0, 1, 2}:
                B, L, C = x.shape
                H, W = layer_up.input_resolution
                H, W = H * 2, W * 2
                if L != H * W:
                    raise ValueError(f"Decoder token length {L} mismatches resolution {(H, W)}.")
                decoder_feats[f"up{inx}"] = x.transpose(1, 2).reshape(B, C, H, W)

            self._collect_layer_cfmoe_stats(layer_up)

        x = self.norm_up(x)  # B L C # 归一化

        return x, decoder_feats

    def forward(self, x, return_features=False):
        self.last_routing_maps = []
        self.last_route_features = []
        self.last_sft_skips = {}
        if self.use_scpe_sft_skip:
            scpe_priors, scpe_all_priors = self.scpe_pyramid(x, return_all=True)
        else:
            scpe_priors, scpe_all_priors = None, {}
        self.last_scpe_priors = scpe_priors or {}
        self.last_scpe_all_priors = scpe_all_priors
        x, x_downsample = self.forward_features(x)
        x, decoder_feats = self.forward_up_features(
            x,
            x_downsample,
            scpe_priors=scpe_priors,
            return_features=return_features,
        )
        x = self.up_x4(x)

        if return_features:
            return x, decoder_feats
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution 
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W 上面三步就是做了Patch Expanding
            x = self.output(x) # Linear Projection

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
