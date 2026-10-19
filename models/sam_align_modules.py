import torch
import torch.nn as nn
import torch.nn.functional as F


class SOABlock(nn.Module):
    """Single SOA unit with runtime channel inference."""

    def __init__(self, out_channels, target_size=None, downsample=False, num_groups=8):
        super().__init__()
        self.target_size = target_size
        self.downsample = downsample
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.built = False

    def _build(self, in_channels, device):
        groups = min(self.num_groups, in_channels)
        while groups > 1 and in_channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, in_channels).to(device)
        self.conv3x3 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2 if self.downsample else 1,
            padding=1,
            bias=False,
        ).to(device)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False).to(device)
        self.built = True

    def forward(self, x):
        if not self.built:
            self._build(x.shape[1], x.device)
        x = self.norm(x)
        x = self.conv3x3(x)
        x = self.act(x)
        x = self.proj(x)
        if self.target_size is not None and x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        return x


class SOAStack(nn.Module):
    """Stack of SOA blocks for progressive scale adaptation."""

    def __init__(self, out_channels, num_blocks, target_size, downsample_schedule=None, num_groups=8):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if downsample_schedule is None:
            downsample_schedule = [False] * num_blocks
        if len(downsample_schedule) != num_blocks:
            raise ValueError("downsample_schedule length must equal num_blocks")

        self.blocks = nn.ModuleList([
            SOABlock(
                out_channels=out_channels,
                target_size=target_size if i == num_blocks - 1 else None,
                downsample=downsample_schedule[i],
                num_groups=num_groups,
            )
            for i in range(num_blocks)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
