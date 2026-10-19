import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Haar Wavelet Filters
# ---------------------------------------------------------------------------
# 2D Haar analysis filters (orthonormal, norm=1):
#   low  = [1, 1] / sqrt(2)   → average (approximate identity)
#   high = [1, -1] / sqrt(2)  → difference (edge detector)
#
# For a 2-level decomposition A→(LL,LH,HL,HH) where:
#   LL = avg(avg(row), col)    (low-pass in both dims)
#   LH = avg(diff(row), col)   (high-pass rows, low-pass cols → horizontal edges)
#   HL = diff(avg(row), col)   (low-pass rows, high-pass cols → vertical edges)
#   HH = diff(diff(row), col)  (high-pass in both dims → diagonal edges)

def _haar_row_transform(x):
    """Row-wise Haar transform with stride-2 subsampling.

    Args:
        x: (B, C, H, W)

    Returns:
        low:  (B, C, H, W//2)  — row averages
        high: (B, C, H, W//2)  — row differences
    """
    C = x.shape[1]
    low_w = torch.tensor([[1.0, 1.0]], dtype=x.dtype, device=x.device) / (2 ** 0.5)
    high_w = torch.tensor([[1.0, -1.0]], dtype=x.dtype, device=x.device) / (2 ** 0.5)

    # Grouped conv: each channel gets its own 1x2 Haar filter
    low = F.conv2d(x, low_w.view(1, 1, 1, 2).repeat(C, 1, 1, 1),
                   stride=(1, 2), padding=0, groups=C)
    high = F.conv2d(x, high_w.view(1, 1, 1, 2).repeat(C, 1, 1, 1),
                    stride=(1, 2), padding=0, groups=C)
    return low, high


def _haar_col_transform(x):
    """Column-wise Haar transform with stride-2 subsampling.

    Args:
        x: (B, C, H, W)

    Returns:
        low:  (B, C, H//2, W)  — col averages
        high: (B, C, H//2, W)  — col differences
    """
    C = x.shape[1]
    low_w = torch.tensor([[1.0], [1.0]], dtype=x.dtype, device=x.device) / (2 ** 0.5)
    high_w = torch.tensor([[1.0], [-1.0]], dtype=x.dtype, device=x.device) / (2 ** 0.5)

    # Grouped conv: each channel gets its own 2x1 Haar filter
    low = F.conv2d(x, low_w.view(1, 1, 2, 1).repeat(C, 1, 1, 1),
                   stride=(2, 1), padding=0, groups=C)
    high = F.conv2d(x, high_w.view(1, 1, 2, 1).repeat(C, 1, 1, 1),
                    stride=(2, 1), padding=0, groups=C)
    return low, high


def dwt_haar2d(x):
    """One-level 2D Haar DWT on tensor x.

    Args:
        x: (B, C, H, W) — H and W must be even

    Returns:
        (LL, LH, HL, HH):
            LL: (B, C, H//2, W//2) — low freq (both dims)
            LH: (B, C, H//2, W//2) — low-row, high-col  (horizontal edges)
            HL: (B, C, H//2, W//2) — high-row, low-col  (vertical edges)
            HH: (B, C, H//2, W//2) — high-row, high-col (diagonal edges)
    """
    row_low, row_high = _haar_row_transform(x)       # (B,C,H,W//2)
    LL, LH = _haar_col_transform(row_low)            # (B,C,H//2,W//2)
    HL, HH = _haar_col_transform(row_high)           # (B,C,H//2,W//2)
    return LL, LH, HL, HH


# ---------------------------------------------------------------------------
# Local Wavelet Energy Style Loss
# ---------------------------------------------------------------------------

class LocalizedWaveletEnergyLoss(nn.Module):
    """Localized Wavelet Energy Style Loss.

    Aligns local high-frequency texture statistics between generated and
    ground-truth images using multi-level Haar DWT and half-overlapping
    sliding windows.

    Formula:
        L_wavelet = Σ_{k=1}^{K} Σ_{s∈{LH,HL,HH}} [
            ||M_local(s_pred) - M_local(s_GT)||_1
            + λ_var * ||V_local(s_pred) - V_local(s_GT)||_1
        ]

    where:
        E(s)       = |s|                          (absolute energy)
        M_local    = AvgPool_{P×P, S}(E)          (local mean energy)
        V_local    = AvgPool_{P×P, S}(E²) - M²   (local variance)

    Window/stride per level (half-overlapping, as specified):
        Level 1 (finest):  P=16, S=8
        Level 2:           P=8,  S=4
        Level 3:           P=4,  S=2
    """

    def __init__(self, levels=3, lambda_var=1.0):
        """
        Args:
            levels:    Number of DWT decomposition levels. Default 3.
            lambda_var: Weight for the variance term. Default 1.0.
        """
        super().__init__()
        self.levels = levels
        self.lambda_var = lambda_var

        self.windows = [16, 8, 4]
        self.strides = [8, 4, 2]

    def _local_stats(self, energy, window_size, stride):
        """Compute local mean and variance of an energy map.

        Args:
            energy:      (B, C, H, W) absolute energy tensor
            window_size: P for AvgPool
            stride:      S for sliding window

        Returns:
            M_local: (B, C, H', W') local mean energy
            V_local: (B, C, H', W') local variance
            (or (None, None) if energy is too small)
        """
        if energy.shape[-1] < window_size or energy.shape[-2] < window_size:
            return None, None

        M = F.avg_pool2d(energy, kernel_size=window_size, stride=stride, padding=0)
        E2 = energy ** 2
        M2 = F.avg_pool2d(E2, kernel_size=window_size, stride=stride, padding=0)
        V = (M2 - M ** 2).clamp(min=0.0)

        return M, V

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, C, H, W) generated image
            target: (B, C, H, W) ground-truth image

        Returns:
            Scalar loss
        """
        total = 0.0

        cur_pred = pred
        cur_target = target

        for level in range(self.levels):
            LL_p, LH_p, HL_p, HH_p = dwt_haar2d(cur_pred)
            LL_t, LH_t, HL_t, HH_t = dwt_haar2d(cur_target)

            subbands_pred = [LH_p, HL_p, HH_p]
            subbands_target = [LH_t, HL_t, HH_t]

            win = self.windows[level]
            stride = self.strides[level]

            for sp, st in zip(subbands_pred, subbands_target):
                Ep = sp.abs()
                Et = st.abs()

                Mp, Vp = self._local_stats(Ep, win, stride)
                Mt, Vt = self._local_stats(Et, win, stride)

                if Mp is None:
                    continue

                total = total + F.l1_loss(Mp, Mt.detach())
                total = total + self.lambda_var * F.l1_loss(Vp, Vt.detach())

            # Standard wavelet pyramid: use LL at this level as input for next
            # level. This naturally captures multi-scale frequency content.
            cur_pred = LL_p
            cur_target = LL_t

        return total


def localized_wavelet_energy_loss(pred, target, levels=3, lambda_var=1.0):
    """Functional version."""
    loss_fn = LocalizedWaveletEnergyLoss(levels=levels, lambda_var=lambda_var)
    return loss_fn(pred, target)
