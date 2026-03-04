import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ======== Auto-detect Chinese fonts (kept as-is) ========
FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

font_path = next((p for p in FONT_PATHS if os.path.exists(p)), None)

if font_path:
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    matplotlib.rcParams['font.family'] = font_name
    matplotlib.rcParams['font.sans-serif'] = [font_name]
    matplotlib.rcParams['axes.unicode_minus'] = False
    print(f"Successfully loaded font: {font_name}")
else:
    print("Warning: No Chinese font found. Falling back to default font.")
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['axes.unicode_minus'] = False

# ======== Class names (English) ========
CLASS_NAMES = ["Background", "Daqu", "Fire Cycle", "Fissure", "Contamination"]

def parse_metadata(file_path):
    """Read first line of CSV and extract model metadata"""
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    meta = {}
    for item in first_line.replace(",", " ").split():
        if "=" in item:
            k, v = item.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta


def clean_label(meta):
    """Generate legend label"""
    model = meta.get("Model", "Unknown Model")
    return f"{model}"


def plot_metrics_from_dir(input_dir="logs", output_dir="plots"):
    """Traverse log folder and plot training metric curves"""
    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {input_dir}")
        return

    all_dfs = []
    for file in csv_files:
        meta = parse_metadata(file)
        label = clean_label(meta)
        df = pd.read_csv(file, skiprows=1)
        df["label"] = label
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    metrics = [col for col in combined.columns if col not in ["epoch", "label"]]

    # IoU metrics
    train_iou_cols = [m for m in metrics if m.startswith("train_iou_class")]
    val_iou_cols = [m for m in metrics if m.startswith("val_iou_class")]
    base_metrics = [m for m in metrics if m not in train_iou_cols + val_iou_cols]

    # ======== Plot basic metric curves (loss, acc, mIoU...) ========
    for metric in base_metrics:
        plt.figure(figsize=(8, 6))
        for label, df in combined.groupby("label"):
            plt.plot(df["epoch"], df[metric], label=label, linewidth=1.8)

        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} Curve Comparison")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    # ======== Plot class-wise IoU ========
    def plot_class_subplots(iou_cols, tag):
        if not iou_cols:
            return
        n_classes = len(iou_cols)
        fig, axes = plt.subplots(n_classes, 1, figsize=(10, 3 * n_classes), sharex=True)

        if n_classes == 1:
            axes = [axes]

        for ax, col in zip(axes, iou_cols):
            try:
                class_id = int(col.split("class")[-1])
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
            except:
                class_name = col

            for label, df in combined.groupby("label"):
                ax.plot(df["epoch"], df[col], label=label, linewidth=1.5)

            ax.set_ylabel(f"{class_name} IoU")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()

        axes[-1].set_xlabel("Epoch")
        plt.suptitle(f"{'Training' if tag == 'Train' else 'Validation'} Set Class-wise IoU Curves")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = os.path.join(output_dir, f"{tag.lower()}_class_iou.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    plot_class_subplots(train_iou_cols, "Train")
    plot_class_subplots(val_iou_cols, "Val")


if __name__ == "__main__":
    plot_metrics_from_dir(input_dir="logs", output_dir="plots")
