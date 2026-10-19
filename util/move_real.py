# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Time       ：2026/1/21 16:32
# Author     ：Gao He
# Description：pix2pix的结果图像fake
"""

import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Move pix2pix real_B images")
    parser.add_argument("--src_dir", required=True, help="Source images directory")
    parser.add_argument("--dst_dir", required=True, help="Destination directory for real_B")
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir

    # 如果目标目录不存在，则创建
    os.makedirs(dst_dir, exist_ok=True)

    split = "_real_B.png"

    # 遍历源目录
    for filename in os.listdir(src_dir):
        if filename.endswith(split):
            src_path = os.path.join(src_dir, filename)

            # 去掉文件名中的 _fake_B
            new_filename = filename.replace(split.replace(".png", ""), "")
            dst_path = os.path.join(dst_dir, new_filename)

            shutil.copy2(src_path, dst_path)

    print("✅ 所有 {} 图像已复制完成".format(split))


if __name__ == "__main__":
    main()
