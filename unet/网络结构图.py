# -*- coding: utf-8 -*-
import torch
from types import SimpleNamespace
from graphviz import Digraph

from unet.model import UNet, NestedUNet, U2NET

def make_cfg(n_channels=3, n_classes=5, bilinear=True, base_c=64, dropout=0.0):
    return SimpleNamespace(n_channels=n_channels, n_classes=n_classes,
                           bilinear=bilinear, base_c=base_c, dropout=dropout)

def visualize_unet_skip_connections(model, model_name):
    """
    自动生成模块级图，并显示跳跃连接（skip connection）
    """
    dot = Digraph(comment=f'{model_name} Module Graph', format='png')
    dot.attr(rankdir='LR')

    # 输入输出节点
    dot.node("Input", "Input\n3x256x256")
    dot.node("Output", "Output\n5x256x256")

    # 收集下采样和上采样模块
    down_modules = []
    up_modules = []

    for name, module in model.named_children():
        label = f"{name}\n({module.__class__.__name__})"
        dot.node(name, label)
        if "down" in name.lower() or "encoder" in name.lower():
            down_modules.append(name)
        elif "up" in name.lower() or "decoder" in name.lower():
            up_modules.append(name)
        elif "out" in name.lower():
            output_module = name

    # 简单连接下采样模块
    prev = "Input"
    for m in down_modules:
        dot.edge(prev, m)
        prev = m

    # 简单连接上采样模块
    prev = down_modules[-1] if down_modules else "Input"
    for m in up_modules:
        dot.edge(prev, m)
        prev = m

    # 上采样模块到输出
    dot.edge(prev, "Output")

    # 添加跳跃连接虚线：下采样到对应上采样
    # 简单假设数量相等，前后对应
    for i in range(min(len(down_modules), len(up_modules))):
        dot.edge(down_modules[i], up_modules[-(i+1)], style="dashed", color="blue", label="skip")

    dot.render(model_name)
    print(f"✅ {model_name} 模块级网络结构图（含跳跃连接）已保存为 {model_name}.png")


if __name__ == "__main__":
    cfg = make_cfg()

    # UNet
    unet = UNet(cfg)
    visualize_unet_skip_connections(unet, "UNet")

    # UNet++
    nested_unet = NestedUNet(cfg)
    visualize_unet_skip_connections(nested_unet, "UNet++")

    # U2NET
    u2net = U2NET(cfg)
    visualize_unet_skip_connections(u2net, "U2NET")
