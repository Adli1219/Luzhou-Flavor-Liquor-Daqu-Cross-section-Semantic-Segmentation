# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
#                Attention Mechanisms
# ==========================================================

# ---------- SE Attention ----------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ---------- CBAM Attention ----------
class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        ca = self.sigmoid_channel(avg_out + max_out)
        x = ca * x
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid_spatial(torch.cat([avg_out, max_out], dim=1))
        x = sa * x
        return x


# ---------- ECA Attention ----------
class ECABlock(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


# ==========================================================
#                    REBNCONV (Basic Unit)
# ==========================================================
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, attention_type=None):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate, bias=False)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

        # Attention choice
        if attention_type == 'se':
            self.attention = SEBlock(out_ch)
        elif attention_type == 'cbam':
            self.attention = CBAMBlock(out_ch)
        elif attention_type == 'eca':
            self.attention = ECABlock(out_ch)
        else:
            self.attention = None

    def forward(self, x):
        x = self.conv_s1(x)
        x = self.bn_s1(x)
        x = self.relu_s1(x)
        if self.attention is not None:
            x = self.attention(x)
        return x


# ==========================================================
#                  Upsample Function
# ==========================================================
def _upsample_like(src, tar):
    # 保证上采样尺寸和目标完全一致（双线性插值）
    return F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=False)


# ==========================================================
#                     RSU Blocks
# ==========================================================

# 辅助：对齐 hx1d 到 hx_in 大小（在返回之前）
def _align_and_add(hx1d, hx_in):
    """确保 hx1d 与 hx_in 空间尺寸一致后再相加"""
    if hx1d.shape[2:] != hx_in.shape[2:]:
        hx1d = F.interpolate(hx1d, size=hx_in.shape[2:], mode='bilinear', align_corners=False)
    return hx1d + hx_in


# ---------- RSU7 ----------
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, attention_type=None):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1, attention_type=attention_type)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1, attention_type=attention_type)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1, attention_type=attention_type)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1, attention_type=attention_type)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1, attention_type=attention_type)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1, attention_type=attention_type)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1, attention_type=attention_type)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2, attention_type=attention_type)
        self.dropout = nn.Dropout(0.2)

        # decoder
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1, attention_type=attention_type)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1, attention_type=attention_type)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1, attention_type=attention_type)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1, attention_type=attention_type)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1, attention_type=attention_type)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1, attention_type=attention_type)

    def forward(self, x):
        hx_in = self.rebnconvin(x)
        hx1 = self.rebnconv1(hx_in)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)
        hx7 = self.dropout(self.rebnconv7(hx6))

        hx6d = self.rebnconv6d(torch.cat((hx7, _upsample_like(hx6, hx7)), 1))
        hx5d = self.rebnconv5d(torch.cat((hx6d, _upsample_like(hx5, hx6d)), 1))
        hx4d = self.rebnconv4d(torch.cat((hx5d, _upsample_like(hx4, hx5d)), 1))
        hx3d = self.rebnconv3d(torch.cat((hx4d, _upsample_like(hx3, hx4d)), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, _upsample_like(hx2, hx3d)), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, _upsample_like(hx1, hx2d)), 1))

        return _align_and_add(hx1d, hx_in)


# ---------- RSU6 ----------
class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, attention_type=None):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1, attention_type=attention_type)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1, attention_type=attention_type)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1, attention_type=attention_type)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1, attention_type=attention_type)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1, attention_type=attention_type)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2, attention_type=attention_type)
        self.dropout = nn.Dropout(0.2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, attention_type=attention_type)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, attention_type=attention_type)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, attention_type=attention_type)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, attention_type=attention_type)

    def forward(self, x):
        hx_in = self.rebnconvin(x)
        hx1 = self.rebnconv1(hx_in)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.dropout(self.rebnconv5(hx))

        hx4d = self.rebnconv4d(torch.cat((hx5, _upsample_like(hx4, hx5)), 1))
        hx3d = self.rebnconv3d(torch.cat((hx4d, _upsample_like(hx3, hx4d)), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, _upsample_like(hx2, hx3d)), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, _upsample_like(hx1, hx2d)), 1))

        return _align_and_add(hx1d, hx_in)


# ---------- RSU5 ----------
class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, attention_type=None):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, attention_type=attention_type)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, attention_type=attention_type)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, attention_type=attention_type)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, attention_type=attention_type)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2, attention_type=attention_type)
        self.dropout = nn.Dropout(0.2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, attention_type=attention_type)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, attention_type=attention_type)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, attention_type=attention_type)

    def forward(self, x):
        hx_in = self.rebnconvin(x)
        hx1 = self.rebnconv1(hx_in)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.dropout(self.rebnconv4(hx))
        hx3d = self.rebnconv3d(torch.cat((hx4, _upsample_like(hx3, hx4)), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, _upsample_like(hx2, hx3d)), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, _upsample_like(hx1, hx2d)), 1))
        return _align_and_add(hx1d, hx_in)


# ---------- RSU4 ----------
class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, attention_type=None):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, attention_type=attention_type)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, attention_type=attention_type)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, attention_type=attention_type)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=2, attention_type=attention_type)
        self.dropout = nn.Dropout(0.2)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, attention_type=attention_type)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, attention_type=attention_type)

    def forward(self, x):
        hx_in = self.rebnconvin(x)
        hx1 = self.rebnconv1(hx_in)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.dropout(self.rebnconv3(hx))
        hx2d = self.rebnconv2d(torch.cat((hx3, _upsample_like(hx2, hx3)), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, _upsample_like(hx1, hx2d)), 1))
        return _align_and_add(hx1d, hx_in)


# ---------- RSU4F ----------
class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, attention_type=None):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, attention_type=attention_type)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1, attention_type=attention_type)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2, attention_type=attention_type)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4, attention_type=attention_type)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8, attention_type=attention_type)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4, attention_type=attention_type)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2, attention_type=attention_type)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1, attention_type=attention_type)

    def forward(self, x):
        hx_in = self.rebnconvin(x)
        hx1 = self.rebnconv1(hx_in)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return _align_and_add(hx1d, hx_in)
