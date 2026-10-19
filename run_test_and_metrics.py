#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
'''
Usage examples:
  # Range: test every epoch from 100 to 500
  python run_test_and_metrics.py --start_epoch 100 --end_epoch 500 \
      --train_name sar2opt_swin_unet_newMoE_baseloss_skipSSDF \
      --test_script scripts/test_pix2pix.sh

  # Range with step: test every 50 epochs
  python run_test_and_metrics.py --start_epoch 100 --end_epoch 500 --step 50 \
      --train_name ... --test_script ...

  # Resume (skip already-tested epochs in CSV):
  python run_test_and_metrics.py --start_epoch 100 --end_epoch 500 --step 10 --resume \
      --train_name ... --test_script ...

  # Explicit list: test only these epochs
  python run_test_and_metrics.py --epochs 350,360,400,430,500,520,550,580 \
      --train_name sar2opt_swin_unet_newMoE_baseloss_skipSSDF \
      --test_script scripts/test_pix2pix.sh
'''

METRIC_PATTERNS = {
    "PSNR": re.compile(r"PSNR:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "SSIM": re.compile(r"SSIM:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "RMSE": re.compile(r"RMSE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "LPIPS": re.compile(r"LPIPS:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "DISTS": re.compile(r"DISTS:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "SAM": re.compile(r"SAM\(deg\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "FID": re.compile(r"FID:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "NIQE": re.compile(r"NIQE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "VIF": re.compile(r"VIF:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
}

CSV_FIELDS = [
    "timestamp",
    "train_name",
    "epoch",
    "phase",
    "result_dir",
    "PSNR",
    "SSIM",
    "RMSE",
    "LPIPS",
    "DISTS",
    "SAM",
    "FID",
    "NIQE",
    "VIF",
    "status",
    "message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pix2pix test by epoch(s), compute metrics, and append CSV records."
    )
    epoch_group = parser.add_mutually_exclusive_group(required=True)
    epoch_group.add_argument(
        "--epochs",
        help="Comma-separated epochs, e.g. 50 or 50,100,150 (mutually exclusive with --start/--end)",
    )
    epoch_group.add_argument(
        "--start_epoch", type=int,
        help="First epoch to test (inclusive). Use with --end_epoch",
    )
    parser.add_argument(
        "--end_epoch", type=int,
        help="Last epoch to test (inclusive). Use with --start_epoch",
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Step size when using --start/--end (default: 1, i.e. every epoch)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip epochs already present in the CSV (by train_name+epoch key)",
    )
    parser.add_argument(
        "--train_name",
        default="sar2opt_swin_unet_newMoE_dinov1_skipSGDA",
        help="Fixed train/test --name used by pix2pix test",
    )
    parser.add_argument(
        "--phase",
        default="test",
        help="Phase name in results dir (default: test)",
    )
    parser.add_argument(
        "--results_root",
        default="/home/gh/pycharmProject/CS-MoE/results",
        help="Root directory where pix2pix results are saved",
    )
    parser.add_argument(
        "--csv_path",
        default="metrics_results.csv",
        help="CSV file path for appending metrics",
    )
    parser.add_argument(
        "--test_script",
        default="scripts/test_SEN12.sh",
        help="Test script path",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="If set, copy each epoch result to archive dir",
    )
    parser.add_argument(
        "--archive_dir",
        default="results_archive",
        help="Archive root directory (used with --archive)",
    )
    args = parser.parse_args()

    if args.start_epoch is not None and args.end_epoch is None:
        parser.error("--end_epoch is required when --start_epoch is set")
    if args.end_epoch is not None and args.start_epoch is None:
        parser.error("--start_epoch is required when --end_epoch is set")
    if args.start_epoch is not None and args.end_epoch is not None:
        if args.start_epoch > args.end_epoch:
            parser.error(f"--start_epoch ({args.start_epoch}) > --end_epoch ({args.end_epoch})")

    return args


def parse_epochs(raw: str) -> List[str]:
    """Parse a comma-separated string into a list of epoch strings."""
    epochs = [item.strip() for item in raw.split(",") if item.strip()]
    if not epochs:
        raise ValueError("--epochs is empty")
    return epochs


def resolve_epoch_list(args: argparse.Namespace) -> List[str]:
    """Resolve the full epoch list from CLI args (range or explicit)."""
    if args.epochs is not None:
        return parse_epochs(args.epochs)
    # Range mode: build list from start to end with step
    return [str(e) for e in range(args.start_epoch, args.end_epoch + 1, args.step)]


def load_tested_epochs(csv_path: Path, train_name: str) -> set:
    """Load (train_name, epoch) keys already in the CSV."""
    if not csv_path.exists():
        return set()
    tested: set = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tested.add((row.get("train_name", "").strip(), row.get("epoch", "").strip()))
    return tested


def parse_metrics(output: str) -> Dict[str, Optional[float]]:
    values: Dict[str, Optional[float]] = {k: None for k in METRIC_PATTERNS.keys()}
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(output)
        if match:
            values[key] = float(match.group(1))
    return values


def ensure_csv_header(csv_path: Path) -> None:
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_record(csv_path: Path, record: Dict[str, object]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(record)


def run_epoch(test_script: Path, epoch: str) -> subprocess.CompletedProcess:
    cmd = ["bash", str(test_script), epoch]
    return subprocess.run(cmd, text=True, capture_output=True)


def build_result_dir(results_root: Path, train_name: str, phase: str, epoch: str) -> Path:
    return results_root / train_name / f"{phase}_{epoch}"


def archive_result(result_dir: Path, archive_root: Path, train_name: str, epoch: str) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    dst = archive_root / f"{train_name}_epoch{epoch}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(result_dir, dst)
    return dst


def main() -> int:
    args = parse_args()
    full_epoch_list = resolve_epoch_list(args)

    project_root = Path(__file__).resolve().parent
    test_script = (project_root / args.test_script).resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    csv_path = (project_root / args.csv_path).resolve()
    archive_root = (project_root / args.archive_dir).resolve()

    if not test_script.exists():
        raise FileNotFoundError(f"Test script not found: {test_script}")

    ensure_csv_header(csv_path)

    # Resume: load already-tested (train_name, epoch) pairs from CSV
    if args.resume:
        tested = load_tested_epochs(csv_path, args.train_name)
    else:
        tested = set()

    epochs_to_run = [e for e in full_epoch_list if (args.train_name, e) not in tested]
    if not epochs_to_run:
        print("[INFO] All requested epochs already have results. Nothing to do.")
        return 0

    print(f"[INFO] Epochs to test: {epochs_to_run}")

    total = len(epochs_to_run)
    for idx, epoch in enumerate(epochs_to_run, 1):
        now = dt.datetime.now().isoformat(timespec="seconds")
        print(f"\n{'='*50}  Epoch {epoch}  ({idx}/{total})  {'='*50}")

        proc = run_epoch(test_script=test_script, epoch=epoch)
        full_output = (proc.stdout or "") + "\n" + (proc.stderr or "")

        status = "success" if proc.returncode == 0 else "failed"
        message = "ok" if proc.returncode == 0 else f"test script exit code {proc.returncode}"

        metrics = parse_metrics(full_output)
        result_dir = build_result_dir(
            results_root=results_root,
            train_name=args.train_name,
            phase=args.phase,
            epoch=epoch,
        )

        if args.archive and result_dir.exists():
            archived = archive_result(
                result_dir=result_dir,
                archive_root=archive_root,
                train_name=args.train_name,
                epoch=epoch,
            )
            print(f"[INFO] Archived result to: {archived}")
        elif args.archive:
            status = "failed"
            message = f"result dir not found for archive: {result_dir}"

        record = {
            "timestamp": now,
            "train_name": args.train_name,
            "epoch": epoch,
            "phase": args.phase,
            "result_dir": str(result_dir),
            "PSNR": metrics["PSNR"],
            "SSIM": metrics["SSIM"],
            "RMSE": metrics["RMSE"],
            "LPIPS": metrics["LPIPS"],
            "DISTS": metrics["DISTS"],
            "SAM": metrics["SAM"],
            "FID": metrics["FID"],
            "NIQE": metrics["NIQE"],
            "VIF": metrics["VIF"],
            "status": status,
            "message": message,
        }

        append_record(csv_path, record)
        print(f"[INFO] CSV updated: {csv_path}")
        print(
            "[INFO] Metrics: "
            + ", ".join(
                f"{k}={record[k]}" for k in ["PSNR", "SSIM", "RMSE", "LPIPS", "DISTS", "SAM", "FID", "NIQE", "VIF"]
            )
        )

        if proc.returncode != 0:
            print("[ERROR] Test script output (tail):")
            tail = full_output[-2000:] if len(full_output) > 2000 else full_output
            print(tail)

    print(f"\n[INFO] Batch test complete. CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
