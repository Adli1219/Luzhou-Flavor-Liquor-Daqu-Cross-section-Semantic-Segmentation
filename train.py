# train.py（只修改关键部分）
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import os.path as osp
import logging
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler

from utils.dataset import BasicDataset
from config import UNetConfig
from losses import CombinedLoss, EnhancedCombinedLoss, AutomaticClassWeights, calculate_iou_per_class, \
    multiclass_dice_coeff


def compute_loss_with_ds(outputs, targets, criterion):
    """兼容各类网络（UNet/UNet++/U2Net/U2NetP）的深监督输出"""
    # deep supervision: tuple 或 list
    if isinstance(outputs, (tuple, list)):
        # 对于EnhancedCombinedLoss，需要特殊处理
        if isinstance(criterion, EnhancedCombinedLoss):
            total_loss = 0
            all_loss_components = {}
            for o in outputs:
                loss = criterion(o, targets)
                total_loss += loss
                # 累加损失分量
                for k, v in criterion.last_losses.items():
                    all_loss_components[k] = all_loss_components.get(k, 0) + v
            loss = total_loss / len(outputs)
            # 平均损失分量
            for k in all_loss_components:
                all_loss_components[k] /= len(outputs)
            criterion.last_losses = all_loss_components
        else:
            loss = sum([criterion(o, targets) for o in outputs]) / len(outputs)
        main_out = outputs[-1]  # 主输出
        return loss, main_out

    # 单输出
    return criterion(outputs, targets), outputs


def train_net(net, cfg, device):
    dataset = BasicDataset(cfg.images_dir, cfg.masks_dir, cfg.scale)
    val_percent = cfg.validation / 100.0
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 自动计算类别权重
    auto_class_weights = True  # 可以添加到config中，这里先设为True
    if auto_class_weights:
        print("🔍 正在自动计算类别权重...")
        class_weights = AutomaticClassWeights.compute_from_dataloader(train_loader, n_classes=cfg.n_classes)
        print(f"📊 类别权重: {class_weights.tolist()}")
    else:
        # 手动设置权重（背景类权重低）
        class_weights = torch.tensor([0.2] + [1.0] * (cfg.n_classes - 1), dtype=torch.float32)

    class_weights = class_weights.to(device)

    # 选择损失函数类型
    loss_type = 'enhanced'  # 可以添加到config中，这里先设为'enhanced'
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

    writer = SummaryWriter(comment=f'LR_{cfg.lr}_BS_{cfg.batch_size}_SCALE_{cfg.scale}')

    if cfg.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay,
                                    nesterov=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.lr_decay_milestones, gamma=cfg.lr_decay_gamma
    )

    run_tag = f"{cfg.model}_{cfg.attention_type or 'noatt'}_{loss_type}"
    writer = SummaryWriter(comment=f'{run_tag}_LR{cfg.lr}_BS{cfg.batch_size}_SCALE{cfg.scale}')

    # CSV 日志（带元信息）
    log_path = f"training_log_{cfg.model}_{loss_type}.csv"
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([f"Model={cfg.model}", f"Attention={cfg.attention_type}", f"Classes={cfg.n_classes}",
                        f"Loss={loss_type}"])
            # 基础头
            header = ["epoch", "train_loss", "val_loss", "train_miou", "val_miou", "train_dice", "train_pa", "val_pa"]
            # 如果是Enhanced损失，添加损失分量头
            if loss_type == 'enhanced':
                header += ["train_lovasz", "train_dice_loss", "train_ce",
                           "val_lovasz", "val_dice_loss", "val_ce"]
            # 类别IoU头
            header += [f"train_iou_class{i}" for i in range(cfg.n_classes)]
            header += [f"val_iou_class{i}" for i in range(cfg.n_classes)]
            w.writerow(header)

    # AMP scaler
    scaler = GradScaler(enabled=True)

    best_val_iou = 0.0

    # 训练循环
    for epoch in range(cfg.epochs):
        net.train()
        epoch_loss = 0.0
        iou_sums = np.zeros(cfg.n_classes)
        iou_counts = np.zeros(cfg.n_classes)
        dice_total = 0.0
        dice_count = 0
        pa_correct = 0
        pa_total = 0

        pbar = tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{cfg.epochs}', unit='img')
        for batch in train_loader:
            images = batch['image'].to(device=device, dtype=torch.float32)
            masks = batch['mask'].to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            with autocast(enabled=True):
                outputs = net(images)
                loss, main_out = compute_loss_with_ds(outputs, masks, criterion)

            # backward with scaler
            scaler.scale(loss).backward()
            # 梯度裁剪（解缩放后裁剪）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss)
            preds = torch.argmax(main_out, dim=1)

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

            pbar.set_postfix(loss=float(loss))
            pbar.update(images.shape[0])
        pbar.close()

        # epoch 统计
        train_loss = epoch_loss / max(1, len(train_loader))
        train_class_iou = iou_sums / np.maximum(iou_counts, 1)
        train_miou = np.nanmean(train_class_iou)
        train_dice = dice_total / max(1, dice_count)
        train_pa = pa_correct / pa_total if pa_total > 0 else 0.0

        # 验证
        val_loss, val_miou, val_pa, val_class_iou = eval_net(net, val_loader, device, n_val, cfg, criterion)

        # 日志记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/train', train_miou, epoch)
        writer.add_scalar('IoU/val', val_miou, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('PixelAcc/train', train_pa, epoch)
        writer.add_scalar('PixelAcc/val', val_pa, epoch)

        # 如果是Enhanced损失，记录损失分量
        if loss_type == 'enhanced' and hasattr(criterion, 'last_losses'):
            for key, value in criterion.last_losses.items():
                writer.add_scalar(f'LossComponents/train_{key}', value, epoch)

        # CSV 写入
        csv_row = [
            epoch + 1, train_loss, val_loss,
            train_miou, val_miou,
            train_dice, train_pa, val_pa
        ]

        # 如果是Enhanced损失，添加损失分量
        if loss_type == 'enhanced' and hasattr(criterion, 'last_losses'):
            csv_row.extend([
                criterion.last_losses.get('lovasz', 0),
                criterion.last_losses.get('dice', 0),
                criterion.last_losses.get('ce', 0),
                0, 0, 0  # 验证集损失分量占位
            ])

        csv_row.extend(train_class_iou)
        csv_row.extend(val_class_iou)

        with open(log_path, "a", newline='') as f:
            w = csv.writer(f)
            w.writerow(csv_row)

        logging.info(
            f"Epoch {epoch + 1}: TrainLoss={train_loss:.4f}, Train_mIoU={train_miou:.4f}, Val_mIoU={val_miou:.4f}"
        )

        # scheduler step
        scheduler.step()

        # 打印详细信息
        print(f"\nEpoch [{epoch + 1}/{cfg.epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train mIoU: {train_miou:.4f} | Val mIoU: {val_miou:.4f}")
        print(f"  Train Dice: {train_dice:.4f} | Train PA: {train_pa:.4f} | Val PA: {val_pa:.4f}")

        # 如果是Enhanced损失，打印损失分量
        if loss_type == 'enhanced' and hasattr(criterion, 'last_losses'):
            print(f"  Loss Components: {criterion.last_losses}")

        print(f"  Train IoU per class: {[round(i, 4) for i in train_class_iou]}")
        print(f"  Val   IoU per class: {[round(i, 4) for i in val_class_iou]}")
        print("-" * 70)

        if epoch % 10 == 0:
            save_path = osp.join(cfg.checkpoints_dir, f"model_epoch_{epoch}.pth")
            torch.save(net.state_dict(), save_path)
            print(f'✅ 模型已保存到: {save_path}')

        # 保存最佳模型（处理 DataParallel）
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


def eval_net(net, loader, device, n_val, cfg, criterion):
    net.eval()
    total_loss = 0.0
    iou_sums = np.zeros(cfg.n_classes)
    iou_counts = np.zeros(cfg.n_classes)
    pa_correct = 0
    pa_total = 0
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc='Validation', unit='img', leave=False):
            images = batch['image'].to(device=device, dtype=torch.float32)
            masks = batch['mask'].to(device=device, dtype=torch.long)

            outputs = net(images)
            loss, main_out = compute_loss_with_ds(outputs, masks, criterion)
            total_loss += float(loss)

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
    return total_loss / max(1, len(loader)), avg_miou, avg_pa, class_iou


# ----------------- 主入口 -----------------
if __name__ == '__main__':
    import logging
    import torch
    from unet.model import UNet, NestedUNet, U2NET

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = UNetConfig()

    # 根据配置选择模型
    if cfg.model == 'UNet':
        net = UNet(cfg)
    elif cfg.model == 'NestedUNet':
        net = NestedUNet(cfg)
    elif cfg.model == 'U2Net':
        net = U2NET(cfg)
    else:
        raise ValueError(f"未知模型类型: {cfg.model}")

    # ✅ 多 GPU 支持
    if torch.cuda.device_count() > 1:
        print(f"🔹 检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel 模式")
        net = torch.nn.DataParallel(net)
    else:
        print("⚠️ 仅检测到单 GPU，将使用单卡训练")

    net = net.to(device)

    # ✅ 加载模型
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
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('⚠️ 已保存中断时的模型权重：INTERRUPTED.pth')