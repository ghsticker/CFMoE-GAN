# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Time       ：2026/4/13 19:45
# Author     ：Gao He
# Description：
"""
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =========================
# 配置区：直接改这里
# =========================
DATASETS = {
    # "sen1-2": "/home/gh/data/sar_to_optical/STCC_datasets/sen1-2",
    # "QXSLAB_SAROPT": "/home/gh/data/sar_to_optical/quick_subsets_paired_real/QXSLAB_SAROPT_subset_seed42",
    "UNICORN_cyclegan": "/home/gh/data/sar_to_optical/UNICORN_cyclegan",
}

OUTPUT_ROOT = "/home/gh/data/sar_to_optical/quick_subsets_paired_real/UNICORN_subset_seed42"

# 二选一：
# 1) 按比例抽样
TRAIN_RATIO = 0.03
TEST_RATIO = 0.03

# 2) 按固定数量抽样
TRAIN_NUM = None   # 例如 2000
TEST_NUM = None    # 例如 500

SEED = 42

# True: 建软链接，快且省空间
# False: 真复制文件
USE_SYMLINK = False

# True: testA/testB 保留全量
KEEP_FULL_TEST = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在: {folder}")
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def build_name_map(files: List[Path]) -> Dict[str, Path]:
    name_map = {}
    for p in files:
        if p.name in name_map:
            raise ValueError(f"检测到重复文件名: {p.name}")
        name_map[p.name] = p
    return name_map


def get_paired_files(dir_a: Path, dir_b: Path) -> List[Tuple[Path, Path]]:
    files_a = list_images(dir_a)
    files_b = list_images(dir_b)

    map_a = build_name_map(files_a)
    map_b = build_name_map(files_b)

    common_names = sorted(set(map_a.keys()) & set(map_b.keys()))
    only_a = sorted(set(map_a.keys()) - set(map_b.keys()))
    only_b = sorted(set(map_b.keys()) - set(map_a.keys()))

    if len(common_names) == 0:
        raise ValueError(
            f"在 {dir_a} 和 {dir_b} 中没有找到同名配对文件。"
            f"请确认 paired 数据是否文件名一致。"
        )

    if only_a:
        print(f"[Warning] {dir_a} 中有 {len(only_a)} 个文件在 {dir_b} 中找不到对应文件。")
        print(f"          示例: {only_a[:5]}")
    if only_b:
        print(f"[Warning] {dir_b} 中有 {len(only_b)} 个文件在 {dir_a} 中找不到对应文件。")
        print(f"          示例: {only_b[:5]}")

    pairs = [(map_a[name], map_b[name]) for name in common_names]
    return pairs


def sample_pairs(
    pairs: List[Tuple[Path, Path]],
    ratio: Optional[float],
    num: Optional[int],
    rng: random.Random,
    keep_full: bool = False,
) -> List[Tuple[Path, Path]]:
    n = len(pairs)
    if n == 0:
        return []

    if keep_full:
        return pairs

    if num is not None:
        k = min(num, n)
    elif ratio is not None:
        if not (0 < ratio <= 1):
            raise ValueError(f"ratio 必须在 (0, 1] 之间，当前为 {ratio}")
        k = max(1, int(n * ratio))
        k = min(k, n)
    else:
        raise ValueError("ratio 和 num 不能同时为 None")

    selected = rng.sample(pairs, k)
    selected.sort(key=lambda x: x[0].name)
    return selected


def link_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if use_symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def save_selected_pairs(
    selected_pairs: List[Tuple[Path, Path]],
    out_a: Path,
    out_b: Path,
    use_symlink: bool,
) -> None:
    ensure_dir(out_a)
    ensure_dir(out_b)

    for src_a, src_b in selected_pairs:
        dst_a = out_a / src_a.name
        dst_b = out_b / src_b.name
        link_or_copy(src_a, dst_a, use_symlink)
        link_or_copy(src_b, dst_b, use_symlink)


def write_log(
    log_path: Path,
    dataset_name: str,
    dataset_root: Path,
    train_pairs_total: int,
    train_pairs_selected: List[Tuple[Path, Path]],
    test_pairs_total: int,
    test_pairs_selected: List[Tuple[Path, Path]],
) -> None:
    lines = [
        f"Dataset: {dataset_name}",
        f"Source: {dataset_root}",
        f"Seed: {SEED}",
        f"Use symlink: {USE_SYMLINK}",
        f"Train ratio: {TRAIN_RATIO}",
        f"Test ratio: {TEST_RATIO}",
        f"Train num: {TRAIN_NUM}",
        f"Test num: {TEST_NUM}",
        f"Keep full test: {KEEP_FULL_TEST}",
        "",
        f"Train total pairs: {train_pairs_total}",
        f"Train selected pairs: {len(train_pairs_selected)}",
        f"Test total pairs: {test_pairs_total}",
        f"Test selected pairs: {len(test_pairs_selected)}",
        "",
        "[Selected Train Pairs]",
    ]

    for a, b in train_pairs_selected:
        lines.append(f"{a.name}    <->    {b.name}")

    lines.append("")
    lines.append("[Selected Test Pairs]")

    for a, b in test_pairs_selected:
        lines.append(f"{a.name}    <->    {b.name}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def process_dataset(dataset_name: str, dataset_root: str) -> None:
    rng = random.Random(SEED)
    dataset_root = Path(dataset_root)

    trainA = dataset_root / "trainA"
    trainB = dataset_root / "trainB"
    testA = dataset_root / "testA"
    testB = dataset_root / "testB"

    for d in [trainA, trainB, testA, testB]:
        if not d.exists():
            raise FileNotFoundError(f"{dataset_name} 缺少目录: {d}")

    out_root = Path(OUTPUT_ROOT) / f"{dataset_name}_subset_seed{SEED}"
    out_trainA = out_root / "trainA"
    out_trainB = out_root / "trainB"
    out_testA = out_root / "testA"
    out_testB = out_root / "testB"
    ensure_dir(out_root)

    # 训练集配对
    train_pairs = get_paired_files(trainA, trainB)
    selected_train_pairs = sample_pairs(
        train_pairs,
        ratio=TRAIN_RATIO,
        num=TRAIN_NUM,
        rng=rng,
        keep_full=False,
    )
    save_selected_pairs(selected_train_pairs, out_trainA, out_trainB, USE_SYMLINK)

    # 测试集配对
    test_pairs = get_paired_files(testA, testB)
    selected_test_pairs = sample_pairs(
        test_pairs,
        ratio=TEST_RATIO,
        num=TEST_NUM,
        rng=rng,
        keep_full=KEEP_FULL_TEST,
    )
    save_selected_pairs(selected_test_pairs, out_testA, out_testB, USE_SYMLINK)

    write_log(
        out_root / "subset_log.txt",
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        train_pairs_total=len(train_pairs),
        train_pairs_selected=selected_train_pairs,
        test_pairs_total=len(test_pairs),
        test_pairs_selected=selected_test_pairs,
    )

    print(f"[Done] {dataset_name}")
    print(f"       train pairs: {len(selected_train_pairs)}/{len(train_pairs)}")
    print(f"       test  pairs: {len(selected_test_pairs)}/{len(test_pairs)}")
    print(f"       output -> {out_root}")


def main() -> None:
    for dataset_name, dataset_root in DATASETS.items():
        process_dataset(dataset_name, dataset_root)
    print("\n全部完成。")


if __name__ == "__main__":
    main()