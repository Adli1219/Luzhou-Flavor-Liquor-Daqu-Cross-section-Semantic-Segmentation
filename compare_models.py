import torch
from thop import profile
import pandas as pd

# ===== 导入模型 =====
from unet.model import UNet, NestedUNet, U2NET
from config import UNetConfig


def get_model_stats(model, input_size=(3, 256, 256), device="cuda"):
    """统计参数量和FLOPs"""
    model = model.to(device)
    model.eval()
    input_tensor = torch.randn(1, *input_size).to(device)

    macs, params = profile(model, inputs=(input_tensor,), verbose=False)

    return {
        "Params (M)": f"{params / 1e6:.2f}",
        "FLOPs (G)": f"{macs / 1e9:.2f}"
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = (3, 320, 320)
    num_classes = 5  # ⚠️ 分类数
    cfg = UNetConfig()

    models = {
        "UNet": UNet(cfg),
        "UNet++ (No Attention)": NestedUNet(cfg),
        "U2NET (No Attention)": U2NET(cfg)
    }

    results = {}
    for name, model in models.items():
        stats = get_model_stats(model, input_size=input_size, device=device)
        results[name] = stats
        print(f"{name:<25} | Params: {stats['Params (M)']} M | FLOPs: {stats['FLOPs (G)']} G")

    # 保存到 CSV 文件
    df = pd.DataFrame(results).T
    df.to_csv("model_stats.csv", index=True)
    print("\n统计结果已保存到 model_stats.csv")
