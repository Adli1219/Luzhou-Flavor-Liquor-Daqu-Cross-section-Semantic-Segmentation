# -*- coding: utf-8 -*-
import re
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse
import os
import glob
import numpy as np

# ======================================================
# 字体加载（支持中文）
# ======================================================
FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]

font_path = next((p for p in FONT_PATHS if os.path.exists(p)), None)
if font_path:
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams['font.family'] = font_name
else:
    matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False


# ======================================================
# ⭐ 模型名称修复：解决 CSV 中不能写 “²”
# ======================================================
MODEL_NAME_FIX = {
    "u2net": "U²-Net",
    "u2netp": "U²-NetP",
    "u2-net": "U²-Net",
    "u2-netp": "U²-NetP",
    "unet": "U-Net",
    "unet++": "U-Net++",
}


# ======================================================
# 类别名称
# ======================================================
CLASS_NAMES = ["Background", "Daqu", "Fire Cycle", "Fissure", "Plaque"]


# ======================================================
# ⭐ 优化后的颜色体系（浅色 train / 深色 val）
# ======================================================
MODEL_COLOR_SETS = [
    ("#A7F3D0", "#059669"),   # 绿色系（浅绿 train / 深绿 val）
    ("#BFDBFE", "#1D4ED8"),   # 蓝色系（浅蓝 train / 深蓝 val）
    ("#FECACA", "#B91C1C"),   # 红色系（浅红 train / 深红 val）
    ("#DDD6FE", "#5B21B6"),   # 紫色系（浅紫 train / 深紫 val）
    ("#FED7AA", "#C2410C"),   # 橙色系（浅橙 train / 深橙 val）
]


# ======================================================
# 工具函数
# ======================================================
def parse_metadata(csv_path):
    """解析首行 Model=xxx 的元信息"""
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    metadata = {}
    if not first_line:
        return metadata

    norm = first_line.replace(",", " ").replace(";", " ").replace("\t", " ")
    pairs = re.findall(r'([A-Za-z0-9_\-]+)\s*=\s*([^\s]+)', norm)

    for k, v in pairs:
        metadata[k] = v.strip(' "\'(),;[]{}')

    return metadata


def read_csv(csv_path):
    """读取 CSV，防止逗号导致列错位"""
    with open(csv_path, "r", encoding="utf-8") as f:
        sep = "\t" if "\t" in f.readline() else ","

    df = pd.read_csv(csv_path, sep=sep, skiprows=[0], engine='python')
    df = df.dropna(axis=1, how='all')

    if "epoch" not in df.columns:
        df.rename(columns={df.columns[0]: "epoch"}, inplace=True)

    df = df.groupby("epoch", as_index=False).last().sort_values("epoch")
    return df


def smooth_curve(values, weight=0.6):
    """指数平滑曲线"""
    if len(values) == 0:
        return values
    smoothed = []
    last = values[0]
    for v in values:
        v = last if np.isnan(v) or np.isinf(v) else v
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed


# ======================================================
# 训练 / 验证 Loss 曲线
# ======================================================
def plot_train_val_loss(csv_paths, out_path):
    plt.figure(figsize=(10, 6))

    for idx, path in enumerate(csv_paths):
        df = read_csv(path)
        meta = parse_metadata(path)

        raw_name = meta.get("Model", os.path.basename(path).replace(".csv", ""))
        model_name = MODEL_NAME_FIX.get(raw_name.lower(), raw_name)

        train_color, val_color = MODEL_COLOR_SETS[idx % len(MODEL_COLOR_SETS)]

        if "train_loss" in df.columns:
            plt.plot(df["epoch"], smooth_curve(df["train_loss"].astype(float)),
                     label=f"{model_name} - Train",
                     color=train_color, linestyle="-", linewidth=2)

        if "val_loss" in df.columns:
            plt.plot(df["epoch"], smooth_curve(df["val_loss"].astype(float)),
                     label=f"{model_name} - Val",
                     color=val_color, linestyle="-", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title("Training & Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Loss 图已保存: {out_path}")


# ======================================================
# 多模型指标对比（miou、dice）
# ======================================================
def plot_metric_multi_model(csv_paths, metric, ylabel=None, out_path="metric.png"):
    plt.figure(figsize=(10, 6))

    for idx, path in enumerate(csv_paths):
        df = read_csv(path)
        meta = parse_metadata(path)

        raw_name = meta.get("Model", os.path.basename(path).replace(".csv", ""))
        model_name = MODEL_NAME_FIX.get(raw_name.lower(), raw_name)

        if metric not in df.columns:
            continue

        _, val_color = MODEL_COLOR_SETS[idx % len(MODEL_COLOR_SETS)]

        plt.plot(df["epoch"], smooth_curve(df[metric].astype(float)),
                 label=model_name, color=val_color, linestyle="-", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel if ylabel else metric, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title(metric, fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {metric} 已保存到: {out_path}")


# ======================================================
# 每类 IoU（不同模型深色区分）
# ======================================================
def plot_per_class_iou(csv_paths, out_dir):
    for cls_idx, cls_name in enumerate(CLASS_NAMES):

        plt.figure(figsize=(10, 6))
        found = False

        for model_idx, path in enumerate(csv_paths):
            df = read_csv(path)
            meta = parse_metadata(path)

            raw_name = meta.get("Model", os.path.basename(path).replace(".csv", ""))
            model_name = MODEL_NAME_FIX.get(raw_name.lower(), raw_name)

            col = f"val_iou_class{cls_idx}"
            if col not in df.columns:
                continue

            found = True

            _, val_color = MODEL_COLOR_SETS[model_idx % len(MODEL_COLOR_SETS)]

            plt.plot(df["epoch"],
                     smooth_curve(df[col].astype(float)),
                     label=model_name, color=val_color, linestyle="-", linewidth=2)

        if not found:
            continue

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("IoU", fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.title(f"Validation IoU - {cls_name}", fontsize=14, fontweight="bold")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

        out_path = os.path.join(out_dir, f"val_iou_{cls_name}.png")
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ {out_path} 已保存")


# ======================================================
# 主程序入口
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logs", type=str, default="logs")
    parser.add_argument("-o", "--out", type=str, default="./training_plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_paths = sorted(glob.glob(os.path.join(args.logs, "*.csv")))

    if not csv_paths:
        raise FileNotFoundError(f"在 {args.logs} 中未找到 CSV 文件")

    plot_train_val_loss(csv_paths, os.path.join(args.out, "loss_train_val_multi.png"))

    for metric in ["val_miou", "val_dice", "val_pa"]:
        plot_metric_multi_model(
            csv_paths, metric, ylabel=metric,
            out_path=os.path.join(args.out, f"{metric}_multi.png")
        )

    plot_per_class_iou(csv_paths, args.out)
