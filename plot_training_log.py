# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse
import os
import glob
import numpy as np

# ======================================================
# 字体加载
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
    matplotlib.rcParams['axes.unicode_minus'] = False
else:
    matplotlib.rcParams['font.family'] = 'Times New Roman'

# 类别名称
CLASS_NAMES = ["Background", "Daqu", "Fire Cycle", "Fissure", "Plaque"]

# 20 种颜色（模型之间完全区分）
MODEL_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))

# 线型
LINESTYLES = ["-", "--", "-.", ":"]

# ======================================================
# 工具函数
# ======================================================
def parse_metadata(csv_path):
    """读取第一行元信息，格式为 'Model=XXX lr=0.001' """
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    metadata = {}
    if first_line:
        for item in first_line.split():
            if "=" in item:
                k, v = item.split("=", 1)
                metadata[k] = v
    return metadata


def read_csv(csv_path):
    """读取 CSV，跳过首行元信息"""
    with open(csv_path, "r", encoding="utf-8") as f:
        sep = "\t" if "\t" in f.readline() else ","

    df = pd.read_csv(csv_path, sep=sep, skiprows=[0], engine='python')
    df = df.dropna(axis=1, how='all')

    if "epoch" not in df.columns:
        df.rename(columns={df.columns[0]: "epoch"}, inplace=True)

    df = df.groupby("epoch", as_index=False).last()
    df = df.sort_values("epoch")
    return df


def smooth_curve(values, weight=0.6):
    """指数平滑"""
    if len(values) == 0:
        return values

    smoothed = []
    last = values[0]
    for v in values:
        if np.isnan(v) or np.isinf(v):
            v = last
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed

def plot_train_val_loss(csv_paths, out_path):
    plt.figure(figsize=(10,6))

    for idx, path in enumerate(csv_paths):
        df = read_csv(path)
        meta = parse_metadata(path)
        model_name = meta.get("Model", os.path.basename(path).replace(".csv",""))

        # -------- train_loss --------
        if "train_loss" in df.columns:
            y_train = smooth_curve(df["train_loss"].astype(float).values)
            plt.plot(
                df["epoch"], y_train,
                label=f"{model_name} - Train Loss",
                color=MODEL_COLORS[idx % 20],
                linestyle="-",
                linewidth=2,
                marker=None
            )

        # -------- val_loss --------
        if "val_loss" in df.columns:
            y_val = smooth_curve(df["val_loss"].astype(float).values)
            plt.plot(
                df["epoch"], y_val,
                label=f"{model_name} - Val Loss",
                color=MODEL_COLORS[idx % 20],
                linestyle="--",
                linewidth=2,
                marker=None
            )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title("Train vs Validation Loss Comparison", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Train + Val Loss 图已保存: {out_path}")

# ======================================================
# 多模型对比（loss、miou 等）
# ======================================================
def plot_metric_multi_model(csv_paths, metric, ylabel=None, out_path="metric.png"):
    plt.figure(figsize=(10, 6))
    all_y = []

    for idx, path in enumerate(csv_paths):
        df = read_csv(path)
        meta = parse_metadata(path)
        model_name = meta.get("Model", os.path.basename(path).replace(".csv", ""))

        if metric not in df.columns:
            continue

        y = smooth_curve(df[metric].astype(float).values)
        all_y.extend(y)

        plt.plot(
            df["epoch"], y,
            label=model_name,
            color=MODEL_COLORS[idx % 20],      # ⭐模型颜色完全区分
            linestyle=LINESTYLES[idx % len(LINESTYLES)],
            linewidth=2,
            marker=None                        # ⭐去掉 marker（解决线条太粗）
        )

    if not all_y:
        return

    ymin, ymax = min(all_y), max(all_y)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel if ylabel else metric, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(ymin * 0.95, ymax * 1.05)
    plt.title(f"{metric} (Validation)", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ {out_path} 已保存")


# ======================================================
# 每类 IoU（每个模型：颜色不同）
# ======================================================
def plot_per_class_iou(csv_paths, out_dir):
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        plt.figure(figsize=(10, 6))
        found = False

        for model_idx, path in enumerate(csv_paths):
            df = read_csv(path)
            meta = parse_metadata(path)
            model_name = meta.get("Model", os.path.basename(path).replace(".csv", ""))

            col = f"val_iou_class{cls_idx}"
            if col not in df.columns:
                continue

            found = True
            y = smooth_curve(df[col].astype(float).values)

            plt.plot(
                df["epoch"], y,
                label=model_name,
                color=MODEL_COLORS[model_idx % 20],   # ⭐不同模型不同颜色
                linestyle=LINESTYLES[model_idx % len(LINESTYLES)],
                linewidth=2,
                marker="o",
                markersize=5,
                markevery=max(1, len(y)//10)         # ⭐稀疏 marker
            )

        if not found:
            continue

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("IoU", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.ylim(0, 1.0)
        plt.title(f"Validation IoU - {cls_name}", fontsize=14, fontweight="bold")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

        out_path = os.path.join(out_dir, f"val_iou_{cls_name}.png")
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ {out_path} 已保存")


# ======================================================
# 主程序
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

    metrics = ["train_loss", "val_loss", "val_miou", "val_pa"]
    plot_train_val_loss(csv_paths, os.path.join(args.out, "loss_train_val_multi.png"))
    for m in metrics:
        plot_metric_multi_model(csv_paths, m, ylabel=m,
                                out_path=os.path.join(args.out, f"{m}_multi.png"))

    plot_per_class_iou(csv_paths, args.out)
