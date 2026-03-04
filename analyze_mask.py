# -*- coding: utf-8 -*-
"""
@Desc: 大曲断面图像后处理 - 计算面积占比、Fire cycle 厚度（皮张厚度）、裂缝长度 + 可视化
"""
import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from tqdm import tqdm

# ------------------ 配置路径 ------------------
mask_folder = "./data/output/masks"
image_folder = "./data/val"
save_dir = "./data/output/vis"
os.makedirs(save_dir, exist_ok=True)

# ------------------ 类别定义 ------------------
CLASS_NAMES = ["background", "Daqu", "Fire cycle", "Fissure", "plaque"]
COLOR_MAP = {
    0: (0, 0, 0),         # background - 黑色
    1: (255, 0, 0),       # Daqu - 蓝色
    2: (255, 0, 255),     # Fire cycle - 紫色
    3: (0, 255, 0),       # Fissure - 绿色
    4: (0, 0, 255),       # plaque - 红色
}

# ------------------ 可视化 ------------------
def visualize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in COLOR_MAP.items():
        color_mask[mask == cls_id] = color
    return color_mask

def calculate_plaque_ratio(mask, daqu_id=1, plaque_id=4):
    """
    计算 杂菌斑(plaque) 在 大曲(Daqu) 内部的面积占比
    返回百分比 (%)
    """
    daqu_pixels = np.sum(mask == daqu_id)
    plaque_pixels = np.sum(mask == plaque_id)

    if daqu_pixels == 0:
        return 0.0

    ratio = plaque_pixels / daqu_pixels * 100
    return ratio
# ------------------ 各类面积占比 ------------------
def calculate_area_ratio(mask):
    h, w = mask.shape
    total = h * w
    ratios = {}
    for i, name in enumerate(CLASS_NAMES):
        area = np.sum(mask == i)
        ratios[name] = area / total * 100
    return ratios

# ------------------ 裂缝长度计算 ------------------
def calculate_fissure_length(mask, fissure_id=3, skeleton_thickness=5):
    fissure_mask = (mask == fissure_id).astype(np.uint8)

    # 1 像素宽骨架
    skeleton = skeletonize(fissure_mask).astype(np.uint8)

    # --- 加粗骨架线 ---
    if skeleton_thickness > 1:
        k = skeleton_thickness
        kernel = np.ones((k, k), np.uint8)
        thick_skeleton = cv2.dilate(skeleton, kernel)
    else:
        thick_skeleton = skeleton

    fissure_length = np.sum(skeleton)  # 长度依旧用原骨架

    return fissure_length, thick_skeleton

# ------------------ 厚度计算（Fire cycle 到 Daqu 边缘距离） ------------------
def calculate_skin_thickness(mask, fire_cycle_id=2, daqu_id=1, sample_step=10):
    fire_mask = (mask == fire_cycle_id).astype(np.uint8)
    daqu_mask = (mask == daqu_id).astype(np.uint8)

    # 提取边缘
    contours_fire, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_daqu, _ = cv2.findContours(daqu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours_fire) == 0 or len(contours_daqu) == 0:
        return 0, []

    fire_points = np.vstack(contours_fire).squeeze(1)
    daqu_points = np.vstack(contours_daqu).squeeze(1)

    lines = []
    distances = []
    for i, (x, y) in enumerate(fire_points[::sample_step]):
        dists = np.sqrt((daqu_points[:, 0] - x) ** 2 + (daqu_points[:, 1] - y) ** 2)
        min_idx = np.argmin(dists)
        nearest = tuple(daqu_points[min_idx])
        dist = dists[min_idx]
        lines.append(((x, y), nearest))
        distances.append(dist)

    avg_thickness = np.mean(distances) if distances else 0
    return avg_thickness, lines

# ------------------ 主分析函数 ------------------
def analyze_features(mask_path, image_path, save_dir, sample_step=10):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"未找到 mask 文件: {mask_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"未找到原图文件: {image_path}")

    # 彩色 mask + 半透明叠加
    color_mask = visualize_mask(mask)
    overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

    # 面积占比
    area_ratios = calculate_area_ratio(mask)
    plaque_ratio = calculate_plaque_ratio(mask)

    # 裂缝骨架与长度
    fissure_length, fissure_skel = calculate_fissure_length(mask)
    overlay[fissure_skel > 0] = (255, 255, 255)  # 白色骨架

    # 厚度计算与连线
    avg_thickness, lines = calculate_skin_thickness(mask, sample_step=sample_step)
    for (p1, p2) in lines:
        cv2.line(overlay, p1, p2, (0, 255, 255), 1)

    # 显示平均厚度
    cv2.putText(overlay, f"Avg thickness: {avg_thickness:.2f}px",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 保存结果
    save_name = os.path.basename(mask_path).replace("_mask.png", ".jpg")
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, overlay)

    # 输出信息
    print(f"\n--- {save_name} ---")
    print(f"平均皮张厚度: {avg_thickness:.2f} px")
    print(f"裂缝长度: {fissure_length} px")
    for k, v in area_ratios.items():
        print(f"{k:<12}: {v:.2f}%")
    print(f"杂菌斑占比面积：{plaque_ratio:.2f}%")

# ------------------ 批量处理 ------------------
if __name__ == "__main__":
    for mask_name in tqdm(os.listdir(mask_folder), desc="Processing masks"):
        if mask_name.endswith(".png"):
            mask_path = os.path.join(mask_folder, mask_name)
            # 匹配原图 (假设 val/ 中是 .jpg 格式)
            base_name = mask_name.replace("_mask.png", ".jpg")
            image_path = os.path.join(image_folder, base_name)
            analyze_features(mask_path, image_path, save_dir, sample_step=15)
