# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Time       ：2026/5/2 00:29
# Author     ：Gao He
# Description：
"""
from pathlib import Path

trainA_dir = Path("/home/gh/data/sar_to_optical/STCC_datasets/sen1-2/trainA")
trainB_dir = Path("/home/gh/data/sar_to_optical/STCC_datasets/sen1-2/trainB")

# 第一次建议保持 True，只检查不真正改名
# 确认无误后改成 False
DRY_RUN = False

assert trainA_dir.exists(), f"trainA folder not found: {trainA_dir}"
assert trainB_dir.exists(), f"trainB folder not found: {trainB_dir}"

trainA_names = {p.name for p in trainA_dir.glob("*.png")}

rename_pairs = []
missing_in_trainA = []
conflicts = []
unchanged = []

for src_path in sorted(trainB_dir.glob("*.png")):
    old_name = src_path.name

    if "_s2_" not in old_name:
        unchanged.append(old_name)
        continue

    new_name = old_name.replace("_s2_", "_s1_", 1)
    dst_path = trainB_dir / new_name

    # 检查改名后的文件名是否能在 trainA 中找到
    if new_name not in trainA_names:
        missing_in_trainA.append((old_name, new_name))

    # 检查 trainB 中是否已经存在目标文件，避免覆盖
    if dst_path.exists() and dst_path != src_path:
        conflicts.append((old_name, new_name))
        continue

    rename_pairs.append((src_path, dst_path))


print("=" * 80)
print(f"trainA png count: {len(trainA_names)}")
print(f"trainB png count: {len(list(trainB_dir.glob('*.png')))}")
print(f"Need rename count: {len(rename_pairs)}")
print(f"Files without _s2_: {len(unchanged)}")
print(f"Missing after rename in trainA: {len(missing_in_trainA)}")
print(f"Name conflicts in trainB: {len(conflicts)}")
print("=" * 80)

if missing_in_trainA:
    print("\n[Warning] These renamed trainB files do not have matching names in trainA:")
    for old_name, new_name in missing_in_trainA[:20]:
        print(f"  {old_name}  ->  {new_name}")
    if len(missing_in_trainA) > 20:
        print(f"  ... and {len(missing_in_trainA) - 20} more")

if conflicts:
    print("\n[Error] These target filenames already exist in trainB, rename may overwrite files:")
    for old_name, new_name in conflicts[:20]:
        print(f"  {old_name}  ->  {new_name}")
    raise RuntimeError("Found filename conflicts. Please resolve them before renaming.")

print("\nPreview rename examples:")
for src_path, dst_path in rename_pairs[:20]:
    print(f"  {src_path.name}  ->  {dst_path.name}")

if DRY_RUN:
    print("\nDRY_RUN=True, no files were renamed.")
    print("After checking the preview, set DRY_RUN=False and run again.")
else:
    for src_path, dst_path in rename_pairs:
        src_path.rename(dst_path)

    print("\nRename finished.")

    # 最终验证 trainA 和 trainB 文件名是否一致
    new_trainB_names = {p.name for p in trainB_dir.glob("*.png")}

    only_in_A = trainA_names - new_trainB_names
    only_in_B = new_trainB_names - trainA_names

    print("=" * 80)
    print(f"Final trainA count: {len(trainA_names)}")
    print(f"Final trainB count: {len(new_trainB_names)}")
    print(f"Only in trainA: {len(only_in_A)}")
    print(f"Only in trainB: {len(only_in_B)}")

    if len(only_in_A) == 0 and len(only_in_B) == 0:
        print("Success: trainA and trainB filenames are exactly matched.")
    else:
        print("Warning: trainA and trainB filenames are still not fully matched.")

        if only_in_A:
            print("\nExamples only in trainA:")
            for name in sorted(list(only_in_A))[:20]:
                print(" ", name)

        if only_in_B:
            print("\nExamples only in trainB:")
            for name in sorted(list(only_in_B))[:20]:
                print(" ", name)