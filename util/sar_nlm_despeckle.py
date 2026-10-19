#!/usr/bin/env python3
"""Channel-wise SAR non-local means despeckling for PNG/JPG images.

This script is intended for SAR images that have already been exported as
ordinary image files.  It applies non-local means in a normalized log-intensity
domain, which is a practical approximation for multiplicative speckle noise.
For RGB PNGs, each color channel is filtered independently.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    from skimage.restoration import denoise_nl_means
except ImportError as exc:  # pragma: no cover - runtime dependency message
    raise SystemExit(
        "Missing dependency: scikit-image. Install it with:\n"
        "  pip install scikit-image pillow numpy"
    ) from exc


DEFAULT_INPUT = Path(
    "/home/gh/data/sar_to_optical/test_subsets/QXSLAB_SAROPT_subset_seed42/testA"
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_exts(value: str) -> set[str]:
    exts = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        exts.add(item if item.startswith(".") else f".{item}")
    return exts


def iter_images(input_dir: Path, exts: set[str], recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def image_to_float(array: np.ndarray) -> tuple[np.ndarray, np.dtype]:
    dtype = array.dtype
    if dtype == np.uint8:
        return array.astype(np.float32) / 255.0, dtype
    if dtype == np.uint16:
        return array.astype(np.float32) / 65535.0, dtype

    arr = array.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32), dtype

    max_value = float(np.nanmax(arr[finite]))
    if max_value > 1.0:
        arr = arr / max_value
    return np.clip(arr, 0.0, 1.0), dtype


def float_to_image(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    array = np.clip(array, 0.0, 1.0)
    if dtype == np.uint16:
        return np.round(array * 65535.0).astype(np.uint16)
    return np.round(array * 255.0).astype(np.uint8)


def despeckle_channel(
    channel: np.ndarray,
    *,
    patch_size: int,
    patch_distance: int,
    h: float,
    eps: float,
    low_percentile: float,
    high_percentile: float,
    fast_mode: bool,
) -> np.ndarray:
    """Filter one image channel in normalized log-intensity space."""
    x = np.clip(channel.astype(np.float32), 0.0, 1.0)

    log_x = np.log(np.maximum(x, eps))
    lo, hi = np.percentile(log_x, [low_percentile, high_percentile])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return x

    log_norm = np.clip((log_x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    filtered_norm = denoise_nl_means(
        log_norm,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        fast_mode=fast_mode,
        preserve_range=True,
        channel_axis=None,
    )
    filtered_log = filtered_norm * (hi - lo) + lo
    return np.clip(np.exp(filtered_log), 0.0, 1.0).astype(np.float32)


def despeckle_array(
    array: np.ndarray,
    *,
    patch_size: int,
    patch_distance: int,
    h: float,
    eps: float,
    low_percentile: float,
    high_percentile: float,
    fast_mode: bool,
) -> np.ndarray:
    common_kwargs = dict(
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        eps=eps,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        fast_mode=fast_mode,
    )

    if array.ndim == 2:
        return despeckle_channel(array, **common_kwargs)

    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    rgb = array[..., :3]
    alpha = array[..., 3:] if array.shape[2] == 4 else None

    # Many grayscale SAR PNGs are stored as RGB with identical channels.
    if rgb.shape[2] == 3 and np.max(np.abs(rgb[..., 0] - rgb[..., 1])) < 1e-7:
        if np.max(np.abs(rgb[..., 0] - rgb[..., 2])) < 1e-7:
            y = despeckle_channel(rgb[..., 0], **common_kwargs)
            out_rgb = np.stack([y, y, y], axis=-1)
        else:
            out_rgb = np.stack(
                [despeckle_channel(rgb[..., c], **common_kwargs) for c in range(3)],
                axis=-1,
            )
    else:
        out_rgb = np.stack(
            [despeckle_channel(rgb[..., c], **common_kwargs) for c in range(rgb.shape[2])],
            axis=-1,
        )

    if alpha is not None:
        return np.concatenate([out_rgb, alpha], axis=-1)
    return out_rgb


def process_one(src: Path, dst: Path, args: argparse.Namespace) -> None:
    with Image.open(src) as img:
        # Convert uncommon modes to RGB while preserving grayscale and RGBA paths.
        if img.mode in {"1", "P", "CMYK", "YCbCr", "LAB", "HSV"}:
            img = img.convert("RGB")
        elif img.mode == "LA":
            img = img.convert("RGBA")

        arr = np.asarray(img)
        arr_float, dtype = image_to_float(arr)
        out_float = despeckle_array(
            arr_float,
            patch_size=args.patch_size,
            patch_distance=args.patch_distance,
            h=args.h,
            eps=args.eps,
            low_percentile=args.low_percentile,
            high_percentile=args.high_percentile,
            fast_mode=not args.slow_mode,
        )
        out = float_to_image(out_float, dtype)

        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(dst)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply channel-wise log-domain SAR non-local means despeckling to "
            "PNG/JPG images."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: sibling folder named <input-dir>_sarnlm",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--exts", default="png,jpg,jpeg,bmp,tif,tiff")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)

    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--patch-distance", type=int, default=6)
    parser.add_argument(
        "--h",
        type=float,
        default=0.08,
        help="NLM filtering strength in normalized log-intensity space. "
        "Try 0.06-0.12; larger means smoother.",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.5)
    parser.add_argument(
        "--slow-mode",
        action="store_true",
        help="Use exact NLM mode. Slower than the default fast approximation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_dir = args.input_dir.expanduser()
    if args.output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_sarnlm"
    else:
        output_dir = args.output_dir.expanduser()

    if args.patch_size < 1 or args.patch_size % 2 != 1:
        raise SystemExit("--patch-size must be a positive odd integer")
    if args.patch_distance < 1:
        raise SystemExit("--patch-distance must be positive")
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    exts = parse_exts(args.exts)
    images = list(iter_images(input_dir, exts, args.recursive))
    if args.max_images is not None:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(images)}")
    print(
        "Params: "
        f"patch_size={args.patch_size}, "
        f"patch_distance={args.patch_distance}, "
        f"h={args.h}"
    )

    done = 0
    skipped = 0
    for src in images:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        process_one(src, dst, args)
        done += 1
        print(f"[{done + skipped:>5}/{len(images)}] {rel}")

    print(f"Done. processed={done}, skipped={skipped}, output_dir={output_dir}")


if __name__ == "__main__":
    main()
