# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LovaszSoftmax.pytorch import lovasz_losses as L

EPS = 1e-6  # 防止除0误差


class AutomaticClassWeights:
    """自动计算类别权重"""

    @staticmethod
    def compute_from_dataloader(dataloader, n_classes=5, max_batches=100):
        """
        从DataLoader中计算类别权重
        """
        class_counts = np.zeros(n_classes)

        for batch_idx, batch in enumerate(dataloader):
            # 你的dataloader返回的是字典，包含'image'和'mask'键
            masks = batch['mask'].numpy()
            for c in range(n_classes):
                class_counts[c] += np.sum(masks == c)

            if batch_idx >= max_batches:
                break

        # 计算中位数频率权重
        total_pixels = np.sum(class_counts)
        class_freq = class_counts / total_pixels

        # 避免除零
        class_freq_safe = class_freq + 1e-10
        median_freq = np.median(class_freq_safe)
        class_weights = median_freq / class_freq_safe
        class_weights = class_weights / np.sum(class_weights) * n_classes

        print(f"📊 类别统计:")
        print(f"  类别计数: {class_counts}")
        print(f"  类别频率: {class_freq}")
        print(f"  中位数频率: {median_freq}")
        print(f"  计算权重: {class_weights}")

        return torch.tensor(class_weights, dtype=torch.float32)


class WeightedCrossEntropyLoss(nn.Module):
    """
    支持类别权重的 CrossEntropyLoss
    在背景类远多于前景类时，通过 class_weights 传入一个 Tensor[n_classes]
    """

    def __init__(self, class_weights=None, ignore_index=255):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.loss_fn(inputs, targets)


class LovaszLossSoftmax(nn.Module):
    """Lovasz-Softmax loss for multi-class segmentation"""

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        assert inputs.dim() == 4, f"Expected 4D input, got {inputs.dim()}D"
        assert targets.dim() == 3, f"Expected 3D target, got {targets.dim()}D"
        probs = F.softmax(inputs, dim=1)
        return L.lovasz_softmax(probs, targets, ignore=self.ignore_index)


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss
    对不同类别大小更加鲁棒
    """

    def __init__(self, smooth=1e-5, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        # One-hot编码
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # 计算每个类别的权重（反向频率）
        class_counts = torch.sum(targets_one_hot, dim=(2, 3))
        class_weights = 1.0 / (class_counts ** 2 + self.smooth)

        # 计算Generalized Dice
        numerator = torch.sum(class_weights * torch.sum(probs * targets_one_hot, dim=(2, 3)), dim=1)
        denominator = torch.sum(class_weights * torch.sum(probs + targets_one_hot, dim=(2, 3)), dim=1)

        dice = 2. * numerator / (denominator + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            weight=alpha,
            ignore_index=ignore_index,
            reduction='none'
        )

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Lovasz + CrossEntropy 组合损失函数
    """

    def __init__(self, class_weights=None, lovasz_weight=0.7, ce_weight=0.3, ignore_index=255):
        super().__init__()
        self.lovasz = LovaszLossSoftmax(ignore_index=ignore_index)
        self.ce = WeightedCrossEntropyLoss(class_weights, ignore_index=ignore_index)
        self.w1 = lovasz_weight
        self.w2 = ce_weight

    def forward(self, inputs, targets):
        return self.w1 * self.lovasz(inputs, targets) + self.w2 * self.ce(inputs, targets)


class EnhancedCombinedLoss(nn.Module):
    """
    增强版组合损失函数（推荐使用）
    包含Lovasz + Dice + CE
    """

    def __init__(self, class_weights=None,
                 loss_weights=None,
                 ignore_index=255,
                 n_classes=5):
        super().__init__()
        self.n_classes = n_classes

        # 初始化损失函数
        self.lovasz = LovaszLossSoftmax(ignore_index=ignore_index)
        self.dice = GeneralizedDiceLoss(ignore_index=ignore_index)
        self.ce = WeightedCrossEntropyLoss(class_weights, ignore_index=ignore_index)

        # 设置损失权重
        if loss_weights is None:
            self.weights = {
                'lovasz': 0.4,
                'dice': 0.3,
                'ce': 0.3
            }
        else:
            self.weights = loss_weights

        # 记录损失分量
        self.last_losses = {}

    def forward(self, inputs, targets):
        # 计算各个损失
        losses = {}
        losses['lovasz'] = self.lovasz(inputs, targets)
        losses['dice'] = self.dice(inputs, targets)
        losses['ce'] = self.ce(inputs, targets)

        # 加权求和
        total_loss = 0
        for key, weight in self.weights.items():
            if weight > 0:
                total_loss += weight * losses[key]

        # 记录损失分量
        self.last_losses = {k: v.item() for k, v in losses.items()}

        return total_loss


# 保持原有的评估函数不变
def calculate_iou_per_class(pred, target, n_classes):
    """
    Compute per-class IoU
    pred, target: [H, W]
    """
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []

    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).float().sum()
        union = pred_inds.float().sum() + target_inds.float().sum() - intersection
        iou = (intersection + EPS) / (union + EPS)
        ious.append(iou.item())

    return ious


def multiclass_dice_coeff(pred, target, n_classes):
    """
    Compute mean Dice coefficient for multi-class segmentation
    pred, target: [H, W]
    """
    pred = pred.view(-1)
    target = target.view(-1)
    dice_scores = []

    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).float().sum()
        denominator = pred_inds.float().sum() + target_inds.float().sum()

        dice = (2. * intersection + EPS) / (denominator + EPS)
        dice_scores.append(dice.item())

    return np.nanmean(dice_scores)


def per_class_dice(pred, target, n_classes):
    """
    Return Dice for each class individually
    """
    pred = pred.view(-1)
    target = target.view(-1)
    dice_scores = []

    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).float().sum()
        denominator = pred_inds.float().sum() + target_inds.float().sum()

        if denominator == 0:
            dice = float('nan')
        else:
            dice = (2. * intersection + EPS) / (denominator + EPS)

        dice_scores.append(dice.item())

    return dice_scores


# 简单的损失函数选择器
def get_loss_function(loss_type='combined', class_weights=None, n_classes=5):
    """
    获取损失函数
    Args:
        loss_type: 'combined' (Lovasz+CE) 或 'enhanced' (Lovasz+Dice+CE)
        class_weights: 类别权重张量
        n_classes: 类别数
    """
    if loss_type == 'enhanced':
        return EnhancedCombinedLoss(
            class_weights=class_weights,
            loss_weights={'lovasz': 0.4, 'dice': 0.3, 'ce': 0.3},
            n_classes=n_classes
        )
    else:  # 默认使用combined
        return CombinedLoss(
            class_weights=class_weights,
            lovasz_weight=0.7,
            ce_weight=0.3
        )