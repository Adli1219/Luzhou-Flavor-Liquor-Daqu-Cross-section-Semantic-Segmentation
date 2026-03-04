# -*- coding: utf-8 -*-
import argparse
import logging
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import cv2

from unet import NestedUNet, UNet, U2NET
from utils.dataset import BasicDataset
from config import UNetConfig

cfg = UNetConfig()
'''
# 类别和固定颜色映射
CLASS_NAMES = ["Background", "Daqu", "Fire Cycle", "Fissure", "Plaque"]
COLOR_MAP = {
    0: (0, 0, 0),         # background - 黑色
    1: (0, 0, 255),       # Daqu - 蓝色
    2: (255, 0, 255),     # Fire cycle - 紫色
    3: (0, 255, 0),       # Fissure - 绿色
    4: (255, 0, 0),       # plaque - 红色
}
'''

# 类别和固定颜色映射
CLASS_NAMES = ["Background","Black","White","Yellow"]
COLOR_MAP = {
    0: (255, 255, 255),         # background - 黑色
    1: (0, 0, 255),       # Black
    2: (255, 0, 255),     # White
    3: (0, 255, 0)       # Yellow
}



def inference_one(net, image, device):
    """对单张图像进行推理"""
    net.eval()
    # BasicDataset.preprocess 返回 numpy 数组 (C, H, W)
    img_tensor = BasicDataset.preprocess(image, is_mask=False, fixed_size=cfg.fixed_size)

    # 转为 torch.Tensor 并添加 batch 维度
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img_tensor)

        # U2NET 会返回 (d0, d1, d2, d3, d4, d5, d6)
        if isinstance(output, (list, tuple)):
            output = output[0]

        # 如果用的是 deep supervision，取最后一层
        if cfg.deepsupervision and isinstance(output, (list, tuple)):
            output = output[-1]

        if cfg.n_classes > 1:
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
        else:
            probs = torch.sigmoid(output).squeeze().cpu().numpy()
            pred = (probs > cfg.out_threshold).astype(np.uint8)

        pred_pil = Image.fromarray(pred.astype(np.uint8))
        pred_resized = pred_pil.resize(image.size, resample=Image.NEAREST)
        return np.array(pred_resized)


def inference_folder(model_path, input_dir, overlay_dir, mask_dir, color_mask_dir):
    """批量推理文件夹"""
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(color_mask_dir, exist_ok=True)

    # 选择模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.model.lower() == "nestedunet":
        net = NestedUNet(
            num_classes=cfg.n_classes,
            input_channels=cfg.n_channels,
            deep_supervision=cfg.deepsupervision,
            attention_type=cfg.attention_type
        ).to(device)
    elif cfg.model.lower() == "unet":
        net = UNet(cfg).to(device)
    elif cfg.model.lower() == "u2net":
        net = U2NET(cfg).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model}")

    # 加载权重
    net.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f'Model loaded from {model_path}')

    # 遍历输入文件夹
    input_imgs = [f for f in os.listdir(input_dir) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    for img_name in tqdm(input_imgs, desc="Predicting"):
        img_path = osp.join(input_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        mask = inference_one(net, img, device)
        # --- 后处理 ---
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # 保存灰度 mask
        mask_save_path = osp.join(mask_dir, osp.splitext(img_name)[0] + '_mask.png')
        cv2.imwrite(mask_save_path, mask.astype(np.uint8))

        # 生成彩色 mask
        w, h = img.size
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in COLOR_MAP.items():
            color_mask[mask == cls_id] = color

        # 保存彩色 mask
        color_mask_save_path = osp.join(color_mask_dir, osp.splitext(img_name)[0] + '_color.png')
        cv2.imwrite(color_mask_save_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

        # 保存叠加结果
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        blended = cv2.addWeighted(img_bgr, 0.7, mask_bgr, 0.3, 0)

        overlay_save_path = osp.join(overlay_dir, img_name)
        cv2.imwrite(overlay_save_path, blended)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Run inference with UNet / UNet++ / U²Net")
    parser.add_argument(
        "-m", "--model", type=str, default="./data/checkpoints/U2Net_None/best_model_U2Net_None.pth",
        help="Path to model weights (.pth)"
    )
    parser.add_argument(
        "-i", "--input", type=str, default="./data/val",
        help="Input image directory"
    )
    parser.add_argument(
        "-o", "--overlay", type=str, default="./data/output/overlay",
        help="Output overlay directory"
    )
    parser.add_argument(
        "-l", "--mask", type=str, default="./data/output/masks",
        help="Output raw label mask directory"
    )
    parser.add_argument(
        "-c", "--colormask", type=str, default="./data/output/color_masks",
        help="Output color mask directory"
    )
    args = parser.parse_args()

    inference_folder(args.model, args.input, args.overlay, args.mask, args.colormask)
