# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Time       ：2026/4/23 23:31
# Author     ：Gao He
# Description：
"""
import torch
import torch.nn as nn
import torch.fft

class FocalFrequencyLoss(nn.Module):
    """
    标准的 Focal Frequency Loss (ICCV 2021).
    """
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, pred, target):
        eps = 1e-8
        fft_pred = torch.fft.fft2(pred, norm='ortho')
        fft_target = torch.fft.fft2(target, norm='ortho')
        amp_pred = torch.abs(fft_pred) + eps
        amp_target = torch.abs(fft_target) + eps
        diff = amp_pred - amp_target
        focal_weight = (torch.abs(diff) ** self.alpha).detach()
        loss = focal_weight * torch.abs(diff)
        return self.loss_weight * torch.mean(loss)


class StrictFocalFreqLoss(nn.Module):
    """
    Strict Focal Frequency Loss — 复数域严格频域对齐损失。

    强制对齐预测图与真实图的 FFT 实部 + 虚部，
    并通过焦点权重动态放大难以生成的高频误差。

    Formula:
        E_real = |R_pred - R_GT|
        E_imag = |I_pred - I_GT|
        W_focal = (E_real + E_imag) ** alpha          # detached from gradient
        L = mean( W_focal * (E_real**2 + E_imag**2) )
    """

    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        fft_pred = torch.fft.fft2(pred, norm='ortho')
        fft_target = torch.fft.fft2(target, norm='ortho')

        R_pred = fft_pred.real
        R_gt = fft_target.real
        I_pred = fft_pred.imag
        I_gt = fft_target.imag

        E_real = (R_pred - R_gt).abs()
        E_imag = (I_pred - I_gt).abs()

        W_focal = (E_real + E_imag).pow(self.alpha)
        W_focal = W_focal / W_focal.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        W_focal = W_focal.detach()

        loss = W_focal * (E_real.pow(2) + E_imag.pow(2))
        return loss.mean()
