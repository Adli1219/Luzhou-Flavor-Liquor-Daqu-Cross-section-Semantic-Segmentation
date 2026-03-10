# train.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import os.path as osp
import logging
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from utils.dataset import BasicDataset
from config import UNetConfig
from losses import (
    CombinedLoss,
    EnhancedCombinedLoss,
    AutomaticClassWeights,
    calculate_iou_per_class,
    multiclass_dice_coeff
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_loss_with_ds(outputs, targets, criterion):
    """兼容各类网络（UNet/UNet++/U2Net/U2NetP）的深监督输出"""
    if isinstance(outputs, (tuple, list)):
        if isinstance(criterion, EnhancedCombinedLoss):
            total_loss = 0
            all_loss_components = {}
            for o in outputs:
                loss = criterion(o, targets)
                total_loss += loss
                for k, v in criterion.last_losses.items():
                    all_loss_components[k] = all_loss_components.get(k, 0) + v

            loss = total_loss / len(outputs)

            for k in all_loss_components:
                all_loss_components[k] /= len(outputs)
            criterion.last_losses = all_loss_components
        else:
            loss = sum([criterion(o, targets) for o in outputs]) / len(outputs)

        main_out = outputs[-1]
        return loss, main_out

    return criterion(outputs, targets), outputs




def build_dataloaders(cfg):
    fixed_size = getattr(cfg, 'fixed_size', (512, 512))
    use_augment = getattr(cfg, 'use_augment', True)
    seed = getattr(cfg, 'seed', 42)

    train_set = BasicDataset(
        imgs_dir=cfg.train_images_dir,
        masks_dir=cfg.train_masks_dir,
        scale=cfg.scale,
        fixed_size=fixed_size,
        augment=use_augment,
        img_names=None
    )

    val_set = BasicDataset(
        imgs_dir=cfg.val_images_dir,
        masks_dir=cfg.val_masks_dir,
        scale=cfg.scale,
        fixed_size=fixed_size,
        augment=False,
        img_names=None
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    return train_set, val_set, train_loader, val_loader


def train_net(net, cfg, device):
    train_set, val_set, train_loader, val_loader = build_dataloaders(cfg)
    n_train = len(train_set)
    n_val = len(val_set)

    print(f"📦 训练集数量: {n_train}")
    print(f"📦 验证集数量: {n_val}")

    # 自动计算类别权重
    auto_class_weights = True
    if auto_class_weights:
        print("🔍 正在自动计算类别权重...")
        class_weights = AutomaticClassWeights.compute_from_dataloader(
            train_loader,
            n_classes=cfg.n_classes
        )
        print(f"📊 类别权重: {class_weights.tolist()}")
    else:
        class_weights = torch.tensor(
            [0.2] + [1.0] * (cfg.n_classes - 1),
            dtype=torch.float32
        )

    class_weights = class_weights.to(device)

    # 损失函数
    loss_type = 'enhanced'
    if loss_type == 'enhanced':
        criterion = EnhancedCombinedLoss(
            class_weights=class_weights,
            loss_weights={'lovasz': 0.4, 'dice': 0.3, 'ce': 0.3},
            n_classes=cfg.n_classes
        )
        print("📈 使用增强损失函数 (Lovasz + Dice + CE)")
    else:
        criterion = CombinedLoss(
            class_weights=class_weights,
            lovasz_weight=0.7,
            ce_weight=0.3
        )
        print("📈 使用组合损失函数 (Lovasz + CE)")

    if cfg.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
            nesterov=True
        )
    else:
        raise ValueError(f"不支持的优化器: {cfg.optimizer}")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.lr_decay_milestones,
        gamma=cfg.lr_decay_gamma
    )

    run_tag = f"{cfg.model}_{cfg.attention_type or 'noatt'}_{loss_type}"
    writer = SummaryWriter(
        comment=f'{run_tag}_LR{cfg.lr}_BS{cfg.batch_size}_SCALE{cfg.scale}'
    )

    # CSV 日志
    log_path = f"training_log_{cfg.model}_{loss_type}.csv"
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow([
                f"Model={cfg.model}",
                f"Attention={cfg.attention_type}",
                f"Classes={cfg.n_classes}",
                f"Loss={loss_type}"
            ])

            header = [
                "epoch", "train_loss", "val_loss",
                "train_miou", "val_miou",
                "train_dice", "train_pa", "val_pa"
            ]

            if loss_type == 'enhanced':
                header += [
                    "train_lovasz", "train_dice_loss", "train_ce",
                    "val_lovasz", "val_dice_loss", "val_ce"
                ]

            header += [f"train_iou_class{i}" for i in range(cfg.n_classes)]
            header += [f"val_iou_class{i}" for i in range(cfg.n_classes)]
            w.writerow(header)

    scaler = GradScaler(enabled=True)
    best_val_iou = 0.0

    for epoch in range(cfg.epochs):
        net.train()

        epoch_loss = 0.0
        iou_sums = np.zeros(cfg.n_classes, dtype=np.float64)
        iou_counts = np.zeros(cfg.n_classes, dtype=np.float64)
        dice_total = 0.0
        dice_count = 0
        pa_correct = 0
        pa_total = 0

        train_loss_components_sum = {}

        pbar = tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{cfg.epochs}', unit='img')

        for batch in train_loader:
            images = batch['image'].to(device=device, dtype=torch.float32)
            masks = batch['mask'].to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                outputs = net(images)
                loss, main_out = compute_loss_with_ds(outputs, masks, criterion)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            preds = torch.argmax(main_out, dim=1)

            # 训练损失分量统计
            if loss_type == 'enhanced' and hasattr(criterion, 'last_losses'):
                for k, v in criterion.last_losses.items():
                    train_loss_components_sum[k] = train_loss_components_sum.get(k, 0.0) + float(v)

            # 统计指标
            for pred, true in zip(preds, masks):
                ious = calculate_iou_per_class(pred.cpu(), true.cpu(), cfg.n_classes)
                for i, iou in enumerate(ious):
                    if not np.isnan(iou):
                        iou_sums[i] += iou
                        iou_counts[i] += 1

                dice_total += multiclass_dice_coeff(pred, true, cfg.n_classes)
                dice_count += 1

            pa_correct += (preds == masks).sum().item()
            pa_total += masks.numel()

            pbar.set_postfix(loss=float(loss.item()))
            pbar.update(images.shape[0])

        pbar.close()

        train_loss = epoch_loss / max(1, len(train_loader))
        train_class_iou = iou_sums / np.maximum(iou_counts, 1)
        train_miou = np.nanmean(train_class_iou)
        train_dice = dice_total / max(1, dice_count)
        train_pa = pa_correct / pa_total if pa_total > 0 else 0.0

        # 平均训练损失分量
        if loss_type == 'enhanced' and len(train_loss_components_sum) > 0:
            train_loss_components_avg = {
                k: v / max(1, len(train_loader))
                for k, v in train_loss_components_sum.items()
            }
        else:
            train_loss_components_avg = {}

        # 验证
        val_loss, val_miou, val_pa, val_class_iou, val_loss_components = eval_net(
            net, val_loader, device, n_val, cfg, criterion, loss_type
        )

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/train', train_miou, epoch)
        writer.add_scalar('IoU/val', val_miou, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('PixelAcc/train', train_pa, epoch)
        writer.add_scalar('PixelAcc/val', val_pa, epoch)

        if loss_type == 'enhanced':
            for key, value in train_loss_components_avg.items():
                writer.add_scalar(f'LossComponents/train_{key}', value, epoch)
            for key, value in val_loss_components.items():
                writer.add_scalar(f'LossComponents/val_{key}', value, epoch)

        # CSV
        csv_row = [
            epoch + 1,
            train_loss, val_loss,
            train_miou, val_miou,
            train_dice, train_pa, val_pa
        ]

        if loss_type == 'enhanced':
            csv_row.extend([
                train_loss_components_avg.get('lovasz', 0.0),
                train_loss_components_avg.get('dice', 0.0),
                train_loss_components_avg.get('ce', 0.0),
                val_loss_components.get('lovasz', 0.0),
                val_loss_components.get('dice', 0.0),
                val_loss_components.get('ce', 0.0),
            ])

        csv_row.extend(train_class_iou.tolist())
        csv_row.extend(val_class_iou.tolist())

        with open(log_path, "a", newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(csv_row)

        logging.info(
            f"Epoch {epoch + 1}: TrainLoss={train_loss:.4f}, "
            f"Train_mIoU={train_miou:.4f}, Val_mIoU={val_miou:.4f}"
        )

        scheduler.step()

        print(f"\nEpoch [{epoch + 1}/{cfg.epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train mIoU: {train_miou:.4f} | Val mIoU: {val_miou:.4f}")
        print(f"  Train Dice: {train_dice:.4f} | Train PA: {train_pa:.4f} | Val PA: {val_pa:.4f}")

        if loss_type == 'enhanced':
            print(f"  Train Loss Components: {train_loss_components_avg}")
            print(f"  Val   Loss Components: {val_loss_components}")

        print(f"  Train IoU per class: {[round(i, 4) for i in train_class_iou]}")
        print(f"  Val   IoU per class: {[round(i, 4) for i in val_class_iou]}")
        print("-" * 70)

        if epoch % 10 == 0:
            os.makedirs(cfg.checkpoints_dir, exist_ok=True)
            save_path = osp.join(cfg.checkpoints_dir, f"model_epoch_{epoch}.pth")

            if isinstance(net, torch.nn.DataParallel):
                torch.save(net.module.state_dict(), save_path)
            else:
                torch.save(net.state_dict(), save_path)

            print(f'✅ 模型已保存到: {save_path}')

        if val_miou > best_val_iou:
            best_val_iou = val_miou
            os.makedirs(cfg.checkpoints_dir, exist_ok=True)
            save_path = osp.join(cfg.checkpoints_dir, f"best_model_{run_tag}.pth")

            if isinstance(net, torch.nn.DataParallel):
                torch.save(net.module.state_dict(), save_path)
            else:
                torch.save(net.state_dict(), save_path)

            logging.info(f"New best model saved: {save_path} (mIoU={best_val_iou:.4f})")

    writer.close()


def eval_net(net, loader, device, n_val, cfg, criterion, loss_type='enhanced'):
    net.eval()

    total_loss = 0.0
    iou_sums = np.zeros(cfg.n_classes, dtype=np.float64)
    iou_counts = np.zeros(cfg.n_classes, dtype=np.float64)
    pa_correct = 0
    pa_total = 0

    val_loss_components_sum = {}

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc='Validation', unit='img', leave=False):
            images = batch['image'].to(device=device, dtype=torch.float32)
            masks = batch['mask'].to(device=device, dtype=torch.long)

            outputs = net(images)
            loss, main_out = compute_loss_with_ds(outputs, masks, criterion)
            total_loss += float(loss.item())

            if loss_type == 'enhanced' and hasattr(criterion, 'last_losses'):
                for k, v in criterion.last_losses.items():
                    val_loss_components_sum[k] = val_loss_components_sum.get(k, 0.0) + float(v)

            preds = torch.argmax(main_out, dim=1)

            for pred, true in zip(preds, masks):
                ious = calculate_iou_per_class(pred.cpu(), true.cpu(), cfg.n_classes)
                for i, iou in enumerate(ious):
                    if not np.isnan(iou):
                        iou_sums[i] += iou
                        iou_counts[i] += 1

            pa_correct += (preds == masks).sum().item()
            pa_total += masks.numel()

    class_iou = iou_sums / np.maximum(iou_counts, 1)
    avg_miou = np.nanmean(class_iou)
    avg_pa = pa_correct / pa_total if pa_total > 0 else 0.0
    avg_loss = total_loss / max(1, len(loader))

    if loss_type == 'enhanced' and len(val_loss_components_sum) > 0:
        val_loss_components_avg = {
            k: v / max(1, len(loader))
            for k, v in val_loss_components_sum.items()
        }
    else:
        val_loss_components_avg = {}

    return avg_loss, avg_miou, avg_pa, class_iou, val_loss_components_avg


# ----------------- 主入口 -----------------
if __name__ == '__main__':
    from unet.model import UNet, NestedUNet, U2NET

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = UNetConfig()

    # 给 cfg 增加默认字段，避免 config.py 里没写时报错
    if not hasattr(cfg, 'fixed_size'):
        cfg.fixed_size = (512, 512)
    if not hasattr(cfg, 'use_augment'):
        cfg.use_augment = True
    if not hasattr(cfg, 'seed'):
        cfg.seed = 42

    # 选择模型
    if cfg.model == 'UNet':
        net = UNet(cfg)
    elif cfg.model == 'NestedUNet':
        net = NestedUNet(cfg)
    elif cfg.model == 'U2Net':
        net = U2NET(cfg)
    else:
        raise ValueError(f"未知模型类型: {cfg.model}")

    # 多 GPU
    if torch.cuda.device_count() > 1:
        print(f"🔹 检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel 模式")
        net = torch.nn.DataParallel(net)
    else:
        print("⚠️ 仅检测到单 GPU，将使用单卡训练")

    net = net.to(device)

    # 加载模型
    if cfg.load:
        state_dict = torch.load(cfg.load, map_location=device)
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(state_dict)
        else:
            net.load_state_dict(state_dict)
        logging.info(f'✅ 模型已从 {cfg.load} 加载权重')

    try:
        train_net(net, cfg, device)
    except KeyboardInterrupt:
        save_path = 'INTERRUPTED.pth'
        if isinstance(net, torch.nn.DataParallel):
            torch.save(net.module.state_dict(), save_path)
        else:
            torch.save(net.state_dict(), save_path)
        logging.info(f'⚠️ 已保存中断时的模型权重：{save_path}')