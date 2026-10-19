from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from sam_rfl import SamPrior, SamPriorCache, build_boundary_from_mask, sanitize_masks


def collect_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            images.append(p)
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(description="Offline SAM cache builder")
    parser.add_argument("--image_root", type=str, required=True, help="root directory containing images")
    parser.add_argument("--cache_dir", type=str, required=True, help="output cache directory")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="path to SAM checkpoint")
    parser.add_argument("--sam_model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type")
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--min_mask_region_area", type=int, default=64)
    parser.add_argument("--max_mask_region_area", type=int, default=10**9)
    parser.add_argument("--boundary_width", type=int, default=3)
    parser.add_argument("--max_masks", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    image_root = Path(args.image_root)
    cache_dir = Path(args.cache_dir)
    cache = SamPriorCache(str(cache_dir))

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(args.device)
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )

    images = collect_images(image_root)
    for img_path in tqdm(images, desc="Building SAM cache"):
        image = np.array(Image.open(img_path).convert("RGB"))
        masks_info = mask_generator.generate(image)
        masks = []
        scores = []
        areas = []
        for item in sorted(masks_info, key=lambda x: x.get("area", 0), reverse=True)[: args.max_masks]:
            mask = item["segmentation"].astype(np.float32)
            masks.append(mask)
            scores.append(float(item.get("predicted_iou", 1.0)))
            areas.append(int(item.get("area", mask.sum())))

        masks = sanitize_masks(masks, args.min_mask_region_area, args.max_mask_region_area)
        boundaries = [build_boundary_from_mask(mask, width=args.boundary_width) for mask in masks]
        areas = [int(mask.sum()) for mask in masks]
        prior = SamPrior(
            masks=masks,
            boundaries=boundaries,
            areas=areas,
            scores=scores[: len(masks)],
            image_size=(image.shape[0], image.shape[1]),
        )
        cache.save(str(img_path), prior)


if __name__ == "__main__":
    main()
