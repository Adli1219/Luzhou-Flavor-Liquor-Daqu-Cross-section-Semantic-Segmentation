# -*- coding: utf-8 -*-
"""
@File    : augment_small_objects_mosaic_safe.py
@Desc    : 防数据泄露版：只对训练集做小目标跨图增强 + Mosaic + Albumentations
"""

import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
import albumentations as A

# ================== 随机种子 ==================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ================== 路径配置 ==================
# 注意：这里必须是“已经划分好的训练集”
train_image_dir = "../data/train_images"
train_mask_dir = "../data/train_masks"

# 输出增强后的训练集
output_image_dir = "../data/train_aug/images"
output_mask_dir = "../data/train_aug/masks"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# 是否把原始训练图也复制到输出目录
# True：最终 output 里包含 原图 + 增强图，可直接训练
# False：output 里只有增强图
copy_original_to_output = True

# ================== 参数配置 ==================
num_aug = 8                  # 每张训练图做多少次单图增强
num_mosaic = 1               # 每张训练图做多少次 Mosaic
mosaic_size = 1024           # Mosaic 输出尺寸
small_classes = [2, 3, 4]    # 需要重点增强的小目标类别
small_scale_range = (1.2, 2.0)   # 小目标粘贴放大倍数，建议别太激进
max_small_copies = 3         # 每张增强图最多粘贴多少个小目标
min_small_area = 5           # 小目标最小面积过滤

# ================== Albumentations增强管道 ==================
# 注意：几何变换会同步作用于 image/mask
single_aug_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),

    A.Affine(
        scale=(0.95, 1.05),
        translate_percent=(0, 0.05),
        rotate=(-10, 10),
        shear=(-5, 5),
        p=0.3
    ),

    A.MotionBlur(blur_limit=(3, 3), p=0.08),
    A.MedianBlur(blur_limit=3, p=0.05),
    A.GaussNoise(var_limit=(10, 25), p=0.10),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.15
    ),

    # 如果你后面发现增强太猛，可先把下面这些关掉
    A.OpticalDistortion(distort_limit=0.03, p=0.05),
    A.GridDistortion(num_steps=5, distort_limit=0.02, p=0.03),
    A.ElasticTransform(alpha=20, sigma=40, alpha_affine=0, p=0.08),

    A.HueSaturationValue(
        hue_shift_limit=8,
        sat_shift_limit=10,
        val_shift_limit=8,
        p=0.15
    ),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.10),

    A.CoarseDropout(
        max_holes=6,
        max_height=24,
        max_width=24,
        p=0.15
    ),
], additional_targets={'mask': 'mask'})


# ================== 辅助函数 ==================
def load_image_mask(img_path, mask_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    if mask is None:
        raise ValueError(f"无法读取标签: {mask_path}")

    return img, mask


def list_image_files(image_dir):
    return sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])


def get_mask_name(img_name):
    return os.path.splitext(img_name)[0] + ".png"


def has_small_object(mask, target_classes):
    return np.any(np.isin(mask, target_classes))


def extract_small_objects_by_class(img, mask, target_classes, min_area=5):
    """
    按类别分别提取小目标，避免多类别直接做连通域导致污染
    返回: [(obj_img, obj_mask), ...]
    其中 obj_mask 只保留对应类别，其余为0
    """
    small_objs = []

    for cls_id in target_classes:
        binary = (mask == cls_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]

            if area < min_area:
                continue

            obj_img = img[y:y + bh, x:x + bw].copy()
            obj_mask = np.zeros((bh, bw), dtype=mask.dtype)

            local_region = (labels[y:y + bh, x:x + bw] == i)
            obj_mask[local_region] = cls_id

            small_objs.append((obj_img, obj_mask))

    return small_objs


def paste_small_objects_cross(
    img,
    mask,
    small_objs_pool,
    max_copies=3,
    scale_range=(1.2, 2.0)
):
    """
    从训练集小目标池中随机取目标，粘贴到当前训练图上
    仅在训练集内部操作，因此不会泄露到 val/test
    """
    h, w = img.shape[:2]

    if len(small_objs_pool) == 0:
        return img, mask

    n = random.randint(1, min(len(small_objs_pool), max_copies))

    out_img = img.copy()
    out_mask = mask.copy()

    for _ in range(n):
        obj_img, obj_mask = random.choice(small_objs_pool)

        oh, ow = obj_mask.shape[:2]
        if oh < 1 or ow < 1:
            continue

        scale = random.uniform(*scale_range)
        new_w = max(1, int(ow * scale))
        new_h = max(1, int(oh * scale))

        if new_w >= w or new_h >= h:
            continue

        obj_img_resized = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        obj_mask_resized = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        x = random.randint(0, w - new_w)
        y = random.randint(0, h - new_h)

        paste_region = (obj_mask_resized > 0)

        # 将目标区域贴到图像
        for c in range(3):
            patch = out_img[y:y + new_h, x:x + new_w, c]
            patch[paste_region] = obj_img_resized[:, :, c][paste_region]
            out_img[y:y + new_h, x:x + new_w, c] = patch

        # 将目标区域贴到mask
        mask_patch = out_mask[y:y + new_h, x:x + new_w]
        mask_patch[paste_region] = obj_mask_resized[paste_region]
        out_mask[y:y + new_h, x:x + new_w] = mask_patch

    return out_img, out_mask


def mosaic_augment_clear(images, masks, out_size=1024):
    """
    清晰 Mosaic 拼接，保持比例，不拉伸
    仅从训练集内部抽样，因此不会泄露到 val/test
    """
    assert len(images) == 4 and len(masks) == 4, "Mosaic需要4张图"

    half_h = out_size // 2
    half_w = out_size // 2

    mosaic_img = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    mosaic_mask = np.zeros((out_size, out_size), dtype=np.uint8)

    positions = [
        (0, 0),
        (0, half_w),
        (half_h, 0),
        (half_h, half_w)
    ]

    for i in range(4):
        img, mask = images[i], masks[i]
        ih, iw = img.shape[:2]

        scale = min(half_h / ih, half_w / iw)
        nh, nw = max(1, int(ih * scale)), max(1, int(iw * scale))

        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        y0, x0 = positions[i]
        y_offset = y0 + (half_h - nh) // 2
        x_offset = x0 + (half_w - nw) // 2

        mosaic_img[y_offset:y_offset + nh, x_offset:x_offset + nw] = img_resized
        mosaic_mask[y_offset:y_offset + nh, x_offset:x_offset + nw] = mask_resized

    return mosaic_img, mosaic_mask


def safe_copy(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)


# ================== 主流程 ==================
def main():
    print("========== 防数据泄露离线增强开始 ==========")
    print(f"训练图目录: {train_image_dir}")
    print(f"训练标签目录: {train_mask_dir}")
    print(f"输出图目录: {output_image_dir}")
    print(f"输出标签目录: {output_mask_dir}")
    print("注意：本脚本只应对训练集运行，验证集/测试集不要参与增强。")

    image_files = list_image_files(train_image_dir)
    print(f"发现训练图像数量: {len(image_files)}")

    # 过滤出有对应 mask 的样本
    valid_image_files = []
    for img_name in image_files:
        mask_name = get_mask_name(img_name)
        mask_path = os.path.join(train_mask_dir, mask_name)
        if os.path.exists(mask_path):
            valid_image_files.append(img_name)
        else:
            print(f"⚠️ 跳过，无对应mask: {img_name}")

    image_files = valid_image_files
    print(f"有效训练样本数量: {len(image_files)}")

    # 1) 可选：先复制原始训练数据到输出目录
    if copy_original_to_output:
        print("📦 正在复制原始训练数据到输出目录...")
        for img_name in tqdm(image_files, desc="复制原始训练数据"):
            src_img = os.path.join(train_image_dir, img_name)
            src_mask = os.path.join(train_mask_dir, get_mask_name(img_name))

            dst_img = os.path.join(output_image_dir, img_name)
            dst_mask = os.path.join(output_mask_dir, get_mask_name(img_name))

            safe_copy(src_img, dst_img)
            safe_copy(src_mask, dst_mask)

    # 2) 构建“小目标池” —— 只从训练集构建，防泄露
    small_objs_pool = []
    for img_name in tqdm(image_files, desc="构建训练集小目标池"):
        img_path = os.path.join(train_image_dir, img_name)
        mask_path = os.path.join(train_mask_dir, get_mask_name(img_name))

        try:
            img, mask = load_image_mask(img_path, mask_path)
            objs = extract_small_objects_by_class(
                img, mask,
                target_classes=small_classes,
                min_area=min_small_area
            )
            small_objs_pool.extend(objs)
        except Exception as e:
            print(f"⚠️ 构建小目标池失败，跳过 {img_name}: {e}")

    print(f"✅ 小目标池构建完成，共 {len(small_objs_pool)} 个小目标实例")

    # 3) 找出训练集中含小目标的图，用于优先 Mosaic
    small_target_images = []
    for img_name in image_files:
        mask_path = os.path.join(train_mask_dir, get_mask_name(img_name))
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and has_small_object(mask, small_classes):
                small_target_images.append(img_name)
        except Exception as e:
            print(f"⚠️ 读取mask失败，跳过 {img_name}: {e}")

    print(f"✅ 含小目标的训练图数量: {len(small_target_images)}")

    # 4) 开始增强
    for img_name in tqdm(image_files, desc="训练集增强中"):
        img_path = os.path.join(train_image_dir, img_name)
        mask_path = os.path.join(train_mask_dir, get_mask_name(img_name))

        try:
            image, mask = load_image_mask(img_path, mask_path)
        except Exception as e:
            print(f"⚠️ 读取失败，跳过 {img_name}: {e}")
            continue

        base = os.path.splitext(img_name)[0]

        # ---- 单图增强 + 小目标跨图增强（池仅来自 train）----
        for i in range(num_aug):
            try:
                aug = single_aug_transform(image=image, mask=mask)
                aug_img, aug_mask = aug['image'], aug['mask']

                if len(small_objs_pool) > 0:
                    aug_img, aug_mask = paste_small_objects_cross(
                        aug_img,
                        aug_mask,
                        small_objs_pool=small_objs_pool,
                        max_copies=max_small_copies,
                        scale_range=small_scale_range
                    )

                out_img_name = f"{base}_aug{i + 1}.jpg"
                out_mask_name = f"{base}_aug{i + 1}.png"

                cv2.imwrite(os.path.join(output_image_dir, out_img_name), aug_img)
                cv2.imwrite(os.path.join(output_mask_dir, out_mask_name), aug_mask)
            except Exception as e:
                print(f"⚠️ 单图增强失败 {img_name} aug{i + 1}: {e}")

        # ---- Mosaic增强（抽样仅来自 train）----
        for j in range(num_mosaic):
            try:
                candidate_pool = small_target_images if len(small_target_images) >= 4 else image_files
                if len(candidate_pool) < 4:
                    print(f"⚠️ 训练图不足4张，无法做Mosaic，跳过 {img_name}")
                    break

                selected = random.sample(candidate_pool, 4)

                imgs, msks = [], []
                for sel in selected:
                    sel_img_path = os.path.join(train_image_dir, sel)
                    sel_mask_path = os.path.join(train_mask_dir, get_mask_name(sel))
                    im, mk = load_image_mask(sel_img_path, sel_mask_path)
                    imgs.append(im)
                    msks.append(mk)

                mos_img, mos_mask = mosaic_augment_clear(
                    imgs, msks, out_size=mosaic_size
                )

                out_img_name = f"{base}_mosaic{j + 1}.jpg"
                out_mask_name = f"{base}_mosaic{j + 1}.png"

                cv2.imwrite(os.path.join(output_image_dir, out_img_name), mos_img)
                cv2.imwrite(os.path.join(output_mask_dir, out_mask_name), mos_mask)
            except Exception as e:
                print(f"⚠️ Mosaic增强失败 {img_name} mosaic{j + 1}: {e}")

    print("✅ 数据增强完成（防数据泄露版）！")
    print("输出内容仅来源于训练集，可安全用于训练。")
    print("验证集/测试集请保持原样，不要做离线增强。")


if __name__ == "__main__":
    main()