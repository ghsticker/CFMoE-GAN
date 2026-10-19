from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SamPrior:
    masks: List[np.ndarray]
    boundaries: List[np.ndarray]
    areas: List[int]
    scores: List[float]
    image_size: Tuple[int, int]


class SamPriorCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.memory_cache: Dict[str, SamPrior] = {}

    @staticmethod
    def _stem_for_path(image_path: str) -> str:
        return Path(image_path).stem

    def cache_path(self, image_path: str) -> Path:
        stem = self._stem_for_path(image_path)
        for suffix in (".npz", ".npy", ".json"):
            p = self.cache_dir / f"{stem}_sam{suffix}"
            if p.exists():
                return p
        return self.cache_dir / f"{stem}_sam.npz"

    def _load_from_path(self, path: Path) -> Optional[SamPrior]:
        if not path.exists():
            return None
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            masks = [np.asarray(m, dtype=np.float32) for m in data.get("masks", [])]
            boundaries = [np.asarray(b, dtype=np.float32) for b in data.get("boundaries", [])]
            areas = [int(a) for a in data.get("areas", [])]
            scores = [float(s) for s in data.get("scores", [])]
            image_size = tuple(data.get("image_size", masks[0].shape if masks else (0, 0)))
            return SamPrior(masks=masks, boundaries=boundaries, areas=areas, scores=scores, image_size=image_size)

        data = np.load(path, allow_pickle=True)
        masks = [np.asarray(m, dtype=np.float32) for m in data["masks"]]
        boundaries = [np.asarray(b, dtype=np.float32) for b in data["boundaries"]]
        areas = [int(a) for a in data["areas"]] if "areas" in data else [int(m.sum()) for m in masks]
        scores = [float(s) for s in data["scores"]] if "scores" in data else [1.0] * len(masks)
        image_size = tuple(data["image_size"].tolist()) if "image_size" in data else masks[0].shape
        return SamPrior(masks=masks, boundaries=boundaries, areas=areas, scores=scores, image_size=image_size)

    def load(self, image_path: str) -> Optional[SamPrior]:
        if image_path in self.memory_cache:
            return self.memory_cache[image_path]
        prior = self._load_from_path(self.cache_path(image_path))
        if prior is not None:
            self.memory_cache[image_path] = prior
        return prior

    def preload(self, image_paths: Sequence[str]) -> int:
        loaded = 0
        for image_path in image_paths:
            if image_path in self.memory_cache:
                loaded += 1
                continue
            prior = self._load_from_path(self.cache_path(image_path))
            if prior is not None:
                self.memory_cache[image_path] = prior
                loaded += 1
        return loaded

    def save(self, image_path: str, prior: SamPrior) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.cache_path(image_path)
        np.savez_compressed(
            out_path,
            masks=np.asarray(prior.masks, dtype=object),
            boundaries=np.asarray(prior.boundaries, dtype=object),
            areas=np.asarray(prior.areas, dtype=np.int64),
            scores=np.asarray(prior.scores, dtype=np.float32),
            image_size=np.asarray(prior.image_size, dtype=np.int64),
        )
        return out_path


class SamRFLosses:
    @staticmethod
    def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 3 else x

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        mask = mask.float().unsqueeze(0).unsqueeze(0)
        return F.interpolate(mask, size=size, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

    @staticmethod
    def _low_high_decompose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        return low, high

    @classmethod
    def sam_recon_loss(cls, fake_b: torch.Tensor, real_b: torch.Tensor, masks: Sequence[np.ndarray], eps: float = 1e-8) -> torch.Tensor:
        fake_b = cls._ensure_4d(fake_b)
        real_b = cls._ensure_4d(real_b)
        if not masks:
            return torch.zeros((), device=fake_b.device, dtype=fake_b.dtype)
        total = torch.zeros((), device=fake_b.device, dtype=fake_b.dtype)
        for mask_np in masks:
            mask = torch.from_numpy(mask_np).to(fake_b.device, dtype=fake_b.dtype)
            mask = cls._resize_mask(mask, fake_b.shape[-2:])
            weight = mask / (mask.sum() + eps)
            total = total + ((fake_b - real_b).abs() * weight.unsqueeze(0)).sum(dim=(1, 2, 3)).mean()
        return total / max(len(masks), 1)

    @classmethod
    def sam_rfc_loss(cls, fake_b: torch.Tensor, real_b: torch.Tensor, masks: Sequence[np.ndarray]) -> torch.Tensor:
        fake_b = cls._ensure_4d(fake_b)
        real_b = cls._ensure_4d(real_b)
        if not masks:
            return torch.zeros((), device=fake_b.device, dtype=fake_b.dtype)
        total = torch.zeros((), device=fake_b.device, dtype=fake_b.dtype)
        for mask_np in masks:
            mask = torch.from_numpy(mask_np).to(fake_b.device, dtype=fake_b.dtype)
            mask = cls._resize_mask(mask, fake_b.shape[-2:])
            fake_region = fake_b * mask.unsqueeze(0)
            real_region = real_b * mask.unsqueeze(0)
            fake_low, fake_high = cls._low_high_decompose(fake_region)
            real_low, real_high = cls._low_high_decompose(real_region)
            total = total + F.l1_loss(fake_low, real_low) + F.l1_loss(fake_high, real_high)
        return total / max(len(masks), 1)

    @classmethod
    def sam_boundary_loss(cls, fake_b: torch.Tensor, real_b: torch.Tensor, boundaries: Sequence[np.ndarray], eps: float = 1e-8) -> torch.Tensor:
        fake_b = cls._ensure_4d(fake_b)
        real_b = cls._ensure_4d(real_b)
        if not boundaries:
            return torch.zeros((), device=fake_b.device, dtype=fake_b.dtype)
        total = torch.zeros((), device=fake_b.device, dtype=fake_b.dtype)
        for boundary_np in boundaries:
            boundary = torch.from_numpy(boundary_np).to(fake_b.device, dtype=fake_b.dtype)
            boundary = cls._resize_mask(boundary, fake_b.shape[-2:])
            weight = boundary / (boundary.sum() + eps)
            total = total + ((fake_b - real_b).abs() * weight.unsqueeze(0)).sum(dim=(1, 2, 3)).mean()
        return total / max(len(boundaries), 1)


def build_boundary_from_mask(mask: np.ndarray, width: int = 3) -> np.ndarray:
    mask_u8 = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((width, width), dtype=np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return np.logical_and(dilated > 0, eroded == 0).astype(np.float32)


def sanitize_masks(masks: Sequence[np.ndarray], min_area: int, max_area: int) -> List[np.ndarray]:
    clean = []
    for mask in masks:
        mask = (np.asarray(mask) > 0.5).astype(np.float32)
        area = int(mask.sum())
        if min_area <= area <= max_area:
            clean.append(mask)
    return clean
