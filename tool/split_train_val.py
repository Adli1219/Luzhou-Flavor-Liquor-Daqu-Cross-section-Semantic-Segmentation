# -*- coding: utf-8 -*-
"""
@File    : split_train_val.py
@Desc    : 将 imgs / masks 划分为 train / val，防止后续离线增强数据泄露
"""

import os
import shutil
import random
from pathlib import Path

# ================== 参数配置 ==================
SEED = 42
VAL_RATIO = 0.2   # 验证集比例，例如 0.2 表示 20%

# ================== 路径配置 ==================
# 建议基于脚本位置自动定位项目目录
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

# 原始数据目录
image_dir = DATA_DIR / "imgs"
mask_dir = DATA_DIR / "masks"

# 输出目录
train_image_dir = DATA_DIR / "train_images"
train_mask_dir = DATA_DIR / "train_masks"
val_image_dir = DATA_DIR / "val_images"
val_mask_dir = DATA_DIR / "val_masks"

# 是否覆盖已存在文件
OVERWRITE = False


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, overwrite=False):
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def list_image_files(img_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main():
    random.seed(SEED)

    print("========== 开始划分 train / val ==========")
    print(f"原始图像目录: {image_dir}")
    print(f"原始标签目录: {mask_dir}")
    print(f"验证集比例: {VAL_RATIO}")
    print(f"随机种子: {SEED}")

    if not image_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {mask_dir}")

    ensure_dir(train_image_dir)
    ensure_dir(train_mask_dir)
    ensure_dir(val_image_dir)
    ensure_dir(val_mask_dir)

    image_files = list_image_files(image_dir)
    if len(image_files) == 0:
        raise ValueError(f"图像目录中没有找到图片文件: {image_dir}")

    # 只保留有对应 mask 的图像
    valid_pairs = []
    skipped = []

    for img_path in image_files:
        mask_name = img_path.stem + ".png"   # 默认 mask 与图像同名，仅后缀为 .png
        mask_path = mask_dir / mask_name

        if mask_path.exists():
            valid_pairs.append((img_path, mask_path))
        else:
            skipped.append(img_path.name)

    print(f"找到图像总数: {len(image_files)}")
    print(f"有效图像-标签对数量: {len(valid_pairs)}")
    print(f"缺失标签数量: {len(skipped)}")

    if len(valid_pairs) == 0:
        raise ValueError("没有找到有效的图像-标签配对，请检查命名规则。")

    if skipped:
        print("以下图像因缺少同名 mask 被跳过：")
        for name in skipped[:20]:
            print("  -", name)
        if len(skipped) > 20:
            print(f"  ... 共 {len(skipped)} 个")

    # 打乱并划分
    random.shuffle(valid_pairs)

    n_total = len(valid_pairs)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val

    train_pairs = valid_pairs[:n_train]
    val_pairs = valid_pairs[n_train:]

    print(f"训练集数量: {len(train_pairs)}")
    print(f"验证集数量: {len(val_pairs)}")

    # 复制训练集
    for img_path, mask_path in train_pairs:
        copy_file(img_path, train_image_dir / img_path.name, overwrite=OVERWRITE)
        copy_file(mask_path, train_mask_dir / mask_path.name, overwrite=OVERWRITE)

    # 复制验证集
    for img_path, mask_path in val_pairs:
        copy_file(img_path, val_image_dir / img_path.name, overwrite=OVERWRITE)
        copy_file(mask_path, val_mask_dir / mask_path.name, overwrite=OVERWRITE)

    print("✅ 划分完成！")
    print(f"训练图像目录: {train_image_dir}")
    print(f"训练标签目录: {train_mask_dir}")
    print(f"验证图像目录: {val_image_dir}")
    print(f"验证标签目录: {val_mask_dir}")
    print("后续离线增强时，只对 train_images / train_masks 做增强。")


if __name__ == "__main__":
    main()