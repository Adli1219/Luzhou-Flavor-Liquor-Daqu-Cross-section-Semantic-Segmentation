#model.py
import torch.nn as nn

from .unet_base import *
from .nested_unet_base import *
from .u2net_base import RSU7, RSU6, RSU5, RSU4, _upsample_like

class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.n_channels = cfg.n_channels
        self.n_classes = cfg.n_classes
        self.bilinear = cfg.bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, self.bilinear)
        self.up2 = Up(512, 128, self.bilinear)
        self.up3 = Up(256, 64, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ---------------- U2NET ----------------
class U2NET(nn.Module):
    def __init__(self, cfg):
        super(U2NET, self).__init__()

        # ✅ 从配置中安全提取参数
        self.in_ch = getattr(cfg, "in_channels", 3)
        self.out_ch = getattr(cfg, "n_classes", 5)
        self.attention_type = getattr(cfg, "attention_type", None)

        in_ch = self.in_ch
        out_ch = self.out_ch
        attention_type = self.attention_type

        # ---------------- Encoder ----------------
        self.stage1 = RSU7(in_ch, 32, 64, attention_type=attention_type)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128, attention_type=attention_type)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256, attention_type=attention_type)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512, attention_type=attention_type)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4(512, 256, 512, attention_type=attention_type)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4(512, 256, 512, attention_type=attention_type)

        # ---------------- Decoder ----------------
        self.stage5d = RSU4(1024, 256, 512, attention_type=attention_type)
        self.stage4d = RSU4(1024, 128, 256, attention_type=attention_type)
        self.stage3d = RSU5(512, 64, 128, attention_type=attention_type)
        self.stage2d = RSU6(256, 32, 64, attention_type=attention_type)
        self.stage1d = RSU7(128, 16, 64, attention_type=attention_type)

        # ---------------- Side Outputs ----------------
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        # ---------------- Encoder ----------------
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)

        # ---------------- Decoder ----------------
        hx5d = self.stage5d(torch.cat((_upsample_like(hx6, hx5), hx5), dim=1))
        hx4d = self.stage4d(torch.cat((_upsample_like(hx5d, hx4), hx4), dim=1))
        hx3d = self.stage3d(torch.cat((_upsample_like(hx4d, hx3), hx3), dim=1))
        hx2d = self.stage2d(torch.cat((_upsample_like(hx3d, hx2), hx2), dim=1))
        hx1d = self.stage1d(torch.cat((_upsample_like(hx2d, hx1), hx1), dim=1))

        # ---------------- Side Outputs ----------------
        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), d1)
        d3 = _upsample_like(self.side3(hx3d), d1)
        d4 = _upsample_like(self.side4(hx4d), d1)
        d5 = _upsample_like(self.side5(hx5d), d1)
        d6 = _upsample_like(self.side6(hx6), d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))

        return d0, d1, d2, d3, d4, d5, d6


class NestedUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_channels = cfg.n_channels
        self.n_classes = cfg.n_classes
        self.deepsupervision = cfg.deepsupervision
        self.bilinear = cfg.bilinear

        nb_filter = [16, 32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output