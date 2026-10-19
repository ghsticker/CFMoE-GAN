"""Plot training loss curves from loss_log.txt into PNG subplots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

'''
../checkpoints/sar2opt_swin_unet_moe_dinov2_swin_mspatch_deep/loss_log.txt
--output
../checkpoints/sar2opt_swin_unet_moe_dinov2_swin_mspatch_deep/loss_plot.png
'''
def parse_loss_log(path: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    step_pattern = re.compile(r"iters:\s*(\d+)")
    kv_pattern = re.compile(r"([A-Za-z0-9_]+):\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)")

    steps: List[int] = []
    values: Dict[str, List[float]] = {}

    last_iter = None
    offset = 0
    exclude_keys = {"epoch", "iters", "time", "data"}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "iters:" not in line:
                continue
            step_match = step_pattern.search(line)
            if not step_match:
                continue
            step = int(step_match.group(1))

            if last_iter is not None and step < last_iter:
                offset += last_iter
            last_iter = step
            global_step = offset + step

            kvs = kv_pattern.findall(line)
            if not kvs:
                continue

            steps.append(global_step)
            for key, val in kvs:
                if key in exclude_keys:
                    continue
                values.setdefault(key, []).append(float(val))

    return steps, values


def align_series(steps: List[int], values: Dict[str, List[float]]) -> Tuple[List[int], Dict[str, np.ndarray]]:
    max_len = len(steps)
    aligned: Dict[str, np.ndarray] = {}
    for key, series in values.items():
        if len(series) < max_len:
            padded = series + [np.nan] * (max_len - len(series))
            aligned[key] = np.array(padded, dtype=float)
        else:
            aligned[key] = np.array(series[:max_len], dtype=float)
    return steps, aligned


def smooth_series(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return data
    if np.isnan(data).all():
        return data
    kernel = np.ones(window, dtype=float) / float(window)
    valid = np.where(np.isnan(data), 0.0, data)
    counts = np.where(np.isnan(data), 0.0, 1.0)
    smoothed = np.convolve(valid, kernel, mode="same")
    norm = np.convolve(counts, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = smoothed / norm
    smoothed[norm == 0] = np.nan
    return smoothed


def plot_losses(
    steps: List[int],
    values: Dict[str, np.ndarray],
    output_path: Path,
    smooth_window: int,
    use_log_y: bool,
) -> None:
    keys = sorted(values.keys())
    if not keys:
        raise ValueError("No loss items found in the log file.")

    steps_arr = np.array(steps, dtype=int)

    n = len(keys)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.5 * rows), squeeze=False)

    for idx, key in enumerate(keys):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        series = values[key]
        if smooth_window > 1:
            series = smooth_series(series, smooth_window)
        ax.plot(steps_arr, series, linewidth=1.2)
        ax.set_title(key)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        if use_log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curves from loss_log.txt to PNG.")
    parser.add_argument("log_path", type=Path, help="Path to loss_log.txt")
    parser.add_argument("--output", type=Path, default=Path("loss_plot.png"), help="Output PNG path")
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window (>=1)")
    parser.add_argument("--logy", action="store_true", help="Use log scale for y-axis")
    args = parser.parse_args()

    steps, values = parse_loss_log(args.log_path)
    steps, aligned = align_series(steps, values)
    plot_losses(steps, aligned, args.output, args.smooth, args.logy)


if __name__ == "__main__":
    main()
