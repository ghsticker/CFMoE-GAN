import csv
from pathlib import Path
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter


class GateLogger:
    def __init__(self, save_dir: Path, csv_name: str = "gate_weights.csv", tb_subdir: str = "tb_gates"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.save_dir / csv_name
        self.tb_dir = self.save_dir / tb_subdir
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self._csv_file = None
        self._csv_writer = None
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        is_new = not self.csv_path.exists()
        self._csv_file = self.csv_path.open("a", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "step",
                "g_low_0",
                "g_low_1",
                "g_high_0",
                "g_high_1",
                "w_l2l",
                "w_l2h",
                "w_h2l",
                "w_h2h",
            ],
            extrasaction="ignore",
        )
        if is_new:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def log(self, step: int, gates: Dict[str, float]) -> None:
        if not gates:
            return

        keep_keys = ["g_low_0", "g_low_1", "g_high_0", "g_high_1", "w_l2l", "w_l2h", "w_h2l", "w_h2h"]
        normalized = {k: gates[k] for k in keep_keys if k in gates}

        if not normalized:
            return

        row = {"step": step, **normalized}
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        for key, value in normalized.items():
            self.writer.add_scalar(f"gates/{key}", value, step)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        if self.writer is not None:
            self.writer.close()


def is_main_process() -> bool:
    return not (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)
