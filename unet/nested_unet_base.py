# nested_unet_base.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== Attention Modules ==========
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(channels // reduction, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 1), channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # conv1d: in_channels=1, out_channels=1, kernel=k_size
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


# ========== Basic VGG Block with optional Attention & Dropout ==========
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, attention_type=None, dropout_rate=0.2):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # 添加Dropout
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # 注意力机制
        if attention_type == "se":
            self.attention = SEBlock(out_channels)
        elif attention_type == "cbam":
            self.attention = CBAMBlock(out_channels)
        elif attention_type == "eca":
            self.attention = ECABlock(out_channels)
        else:
            self.attention = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)  # 在第一个卷积后添加dropout

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        if self.attention is not None:
            out = self.attention(out)

        return out

