"""Projected GAN discriminator components.

This module follows the discriminator design from autonomousvision/projected-gan:
a frozen EfficientNet-Lite feature projector with random cross-channel/cross-scale
mixing, followed by small multi-scale discriminator heads.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def norm_layer(channels):
    return nn.BatchNorm2d(channels)


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        if separable:
            self.main = nn.Sequential(
                conv2d(in_planes, in_planes, kernel_size=3, groups=in_planes, bias=False, padding=1),
                conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                norm_layer(out_planes),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2, 2),
            )
        else:
            self.main = nn.Sequential(
                conv2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
                norm_layer(out_planes),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockPatch(nn.Module):
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        self.main = nn.Sequential(
            DownBlock(in_planes, out_planes, separable=separable),
            conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, expand=False, align_corners=True):
        super().__init__()
        self.align_corners = align_corners
        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output = output + xs[1]
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(output)


def _meshgrid(*args):
    try:
        return torch.meshgrid(*args, indexing="ij")
    except TypeError:
        return torch.meshgrid(*args)


def diff_augment(x, policy=""):
    if not policy:
        return x
    for policy_name in policy.split(","):
        if policy_name == "color":
            x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
            x_mean = x.mean(dim=1, keepdim=True)
            x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
            x_mean = x.mean(dim=(1, 2, 3), keepdim=True)
            x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
        elif policy_name == "translation":
            shift_x = int(x.size(2) * 0.125 + 0.5)
            shift_y = int(x.size(3) * 0.125 + 0.5)
            translation_x = torch.randint(-shift_x, shift_x + 1, size=(x.size(0), 1, 1), device=x.device)
            translation_y = torch.randint(-shift_y, shift_y + 1, size=(x.size(0), 1, 1), device=x.device)
            grid_batch, grid_x, grid_y = _meshgrid(
                torch.arange(x.size(0), dtype=torch.long, device=x.device),
                torch.arange(x.size(2), dtype=torch.long, device=x.device),
                torch.arange(x.size(3), dtype=torch.long, device=x.device),
            )
            grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
            grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
            x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
            x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        elif policy_name == "cutout":
            cutout_size = (int(x.size(2) * 0.2 + 0.5), int(x.size(3) * 0.2 + 0.5))
            offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=(x.size(0), 1, 1), device=x.device)
            offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=(x.size(0), 1, 1), device=x.device)
            grid_batch, grid_x, grid_y = _meshgrid(
                torch.arange(x.size(0), dtype=torch.long, device=x.device),
                torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
                torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
            )
            grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
            grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
            mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
            mask[grid_batch, grid_x, grid_y] = 0
            x = x * mask.unsqueeze(1)
        else:
            raise ValueError(f"Unknown DiffAugment policy: {policy_name}")
    return x.contiguous()


def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    out_channels = [cout, cout * 2, cout * 4, cout * 8] if expand else [cout] * 4
    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.CHANNELS = out_channels
    return scratch


def _make_scratch_csm(scratch, in_channels, cout, expand=False):
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], expand=expand)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0], expand=False)
    scratch.CHANNELS = [cout, cout, cout * 2, cout * 4] if expand else [cout] * 4
    return scratch


class SequentialFeatureBackbone(nn.Module):
    def __init__(self, layer0, layer1, layer2, layer3):
        super().__init__()
        self.layer0 = layer0
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    def forward(self, x):
        out0 = self.layer0(x)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return {"0": out0, "1": out1, "2": out2, "3": out3}


class TimmFeatureBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model(x)
        if isinstance(features, dict):
            features = list(features.values())
        if len(features) < 4:
            raise RuntimeError(f"Projected discriminator needs at least 4 feature maps, got {len(features)}.")
        features = features[-4:]
        return {str(i): feat for i, feat in enumerate(features)}


def _make_efficientnet(model):
    return SequentialFeatureBackbone(
        nn.Sequential(model.conv_stem, model.bn1, model.act1, *model.blocks[0:2]),
        nn.Sequential(*model.blocks[2:3]),
        nn.Sequential(*model.blocks[3:5]),
        nn.Sequential(*model.blocks[5:9]),
    )


def _calc_channels(pretrained, inp_res=224):
    was_training = pretrained.training
    pretrained.eval()
    tmp = torch.zeros(1, 3, inp_res, inp_res)
    with torch.no_grad():
        features = pretrained(tmp)
        channels = [features[str(i)].shape[1] for i in range(4)]
    pretrained.train(was_training)
    return channels


def _make_projector(im_res, cout, proj_type, expand, backbone, pretrained):
    if proj_type not in {0, 1, 2}:
        raise ValueError("projected_proj_type must be 0, 1, or 2.")

    try:
        import timm
    except ImportError as exc:
        raise ImportError("Projected discriminator requires timm. Install it with `pip install timm`.") from exc

    model = timm.create_model(backbone, pretrained=pretrained)
    if all(hasattr(model, attr) for attr in ("conv_stem", "bn1", "act1", "blocks")):
        feature_net = _make_efficientnet(model)
    else:
        feature_model = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        feature_net = TimmFeatureBackbone(feature_model)
    im_res = 256
    feature_net.RESOLUTIONS = [im_res // 4, im_res // 8, im_res // 16, im_res // 32]
    feature_net.CHANNELS = _calc_channels(feature_net)

    if proj_type == 0:
        return feature_net, None

    scratch = nn.Module()
    scratch = _make_scratch_ccm(scratch, in_channels=feature_net.CHANNELS, cout=cout, expand=expand)
    feature_net.CHANNELS = scratch.CHANNELS

    if proj_type == 1:
        return feature_net, scratch

    scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS, cout=cout, expand=expand)
    feature_net.RESOLUTIONS = [res * 2 for res in feature_net.RESOLUTIONS]
    feature_net.CHANNELS = scratch.CHANNELS
    return feature_net, scratch


class FRandomProj(nn.Module):
    def __init__(
        self,
        im_res=256,
        cout=64,
        expand=True,
        proj_type=2,
        backbone="tf_efficientnet_lite0",
        pretrained=True,
    ):
        super().__init__()
        self.proj_type = proj_type
        self.pretrained, self.scratch = _make_projector(
            im_res=im_res,
            cout=cout,
            proj_type=proj_type,
            expand=expand,
            backbone=backbone,
            pretrained=pretrained,
        )
        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS

    def forward(self, x):
        out = self.pretrained(x)
        if self.proj_type == 0:
            return out

        out0 = self.scratch.layer0_ccm(out["0"])
        out1 = self.scratch.layer1_ccm(out["1"])
        out2 = self.scratch.layer2_ccm(out["2"])
        out3 = self.scratch.layer3_ccm(out["3"])
        out = {"0": out0, "1": out1, "2": out2, "3": out3}
        if self.proj_type == 1:
            return out

        out3 = self.scratch.layer3_csm(out3)
        out2 = self.scratch.layer2_csm(out3, out2)
        out1 = self.scratch.layer1_csm(out2, out1)
        out0 = self.scratch.layer0_csm(out1, out0)
        return {"0": out0, "1": out1, "2": out2, "3": out3}


class SingleDisc(nn.Module):
    def __init__(self, nc, start_sz=256, end_sz=8, ndf=None, separable=False, patch=False):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64, 256: 32, 512: 16, 1024: 8}
        if start_sz not in channel_dict:
            start_sz = min(channel_dict.keys(), key=lambda size: abs(size - start_sz))

        nfc = {k: ndf for k in channel_dict} if ndf is not None else dict(channel_dict)
        nfc[start_sz] = nc

        layers = []
        down_block = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(down_block(nfc[start_sz], nfc[start_sz // 2]))
            start_sz = start_sz // 2
        layers.append(conv2d(nfc[end_sz], 1, kernel_size=4, stride=1, padding=0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MultiScaleD(nn.Module):
    def __init__(self, channels, resolutions, num_discs=4, ndf=None, separable=False, patch=False):
        super().__init__()
        if num_discs < 1 or num_discs > 4:
            raise ValueError("projected_num_discs must be between 1 and 4.")

        mini_discs = []
        for i, (cin, res) in enumerate(zip(channels[:num_discs], resolutions[:num_discs])):
            start_sz = 16 if patch else res
            mini_discs.append((str(i), SingleDisc(nc=cin, start_sz=start_sz, ndf=ndf, separable=separable, patch=patch)))
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features):
        logits = []
        for key, disc in self.mini_discs.items():
            logits.append(disc(features[key]).view(features[key].size(0), -1))
        return torch.cat(logits, dim=1)


class RepeatToRGB(nn.Module):
    def forward(self, x):
        return x.repeat(1, 3, 1, 1)


class ProjectedDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc=3,
        im_res=256,
        cout=64,
        expand=True,
        proj_type=2,
        num_discs=4,
        backbone="tf_efficientnet_lite0",
        pretrained=True,
        diffaug=True,
        diffaug_policy="color,translation,cutout",
        interp224=True,
        ndf=None,
        separable=False,
        patch=False,
    ):
        super().__init__()
        self.diffaug = diffaug
        self.diffaug_policy = diffaug_policy
        self.interp224 = interp224

        if input_nc == 3:
            self.input_adapter = nn.Identity()
        elif input_nc == 1:
            self.input_adapter = RepeatToRGB()
        else:
            self.input_adapter = nn.Conv2d(input_nc, 3, kernel_size=1, stride=1, padding=0)

        self.feature_network = FRandomProj(
            im_res=im_res,
            cout=cout,
            expand=expand,
            proj_type=proj_type,
            backbone=backbone,
            pretrained=pretrained,
        )
        for param in self.feature_network.parameters():
            param.requires_grad = False
        self.feature_network.eval()

        self.discriminator = MultiScaleD(
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,
            num_discs=num_discs,
            ndf=ndf,
            separable=separable,
            patch=patch,
        )

    def apply(self, fn):
        self.input_adapter.apply(fn)
        self.discriminator.apply(fn)
        return self

    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def set_discriminator_requires_grad(self, requires_grad):
        for param in self.input_adapter.parameters():
            param.requires_grad = requires_grad
        for param in self.discriminator.parameters():
            param.requires_grad = requires_grad
        for param in self.feature_network.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.input_adapter.train(mode)
        self.feature_network.train(False)
        self.discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x):
        x = self.input_adapter(x)
        if self.diffaug:
            x = diff_augment(x, policy=self.diffaug_policy)
        if self.interp224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        features = self.feature_network(x)
        return self.discriminator(features)
