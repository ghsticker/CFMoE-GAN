# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Time       ：2026/4/13 17:08
# Author     ：Gao He
# Description：
"""
import os
import shutil
from tqdm import tqdm

# ====== 原始路径 ======
root_dir = "/home/gh/data/sar_to_optical/UNICORN"

sar_train = os.path.join(root_dir, "SAR", "train")
sar_val   = os.path.join(root_dir, "SAR", "validation")

eo_train  = os.path.join(root_dir, "EO", "train")
eo_val    = os.path.join(root_dir, "EO", "validation")

# ====== 目标路径 ======
target_root = "/home/gh/data/sar_to_optical/UNICORN_cyclegan"

trainA = os.path.join(target_root, "trainA")
trainB = os.path.join(target_root, "trainB")
testA  = os.path.join(target_root, "testA")
testB  = os.path.join(target_root, "testB")

# ====== 创建目录 ======
for d in [trainA, trainB, testA, testB]:
    os.makedirs(d, exist_ok=True)

# ====== 复制函数 ======
def copy_images(src_dir, dst_dir):
    files = os.listdir(src_dir)
    for f in tqdm(files, desc=f"Copying {src_dir}"):
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

# ====== 执行 ======
copy_images(sar_train, trainA)  # SAR → trainA
copy_images(eo_train,  trainB)  # EO  → trainB
copy_images(sar_val,   testA)   # SAR → testA
copy_images(eo_val,    testB)   # EO  → testB

print("✅ 数据整理完成！")