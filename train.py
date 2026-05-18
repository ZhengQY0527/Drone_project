import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import json
import logging
import time
from datetime import datetime
from config import Config
from dataset import get_loaders
from model import build_model

torch.backends.cudnn.benchmark = True

def setup_logger(log_dir):
    """配置双路日志：文件保留完整记录，终端只输出简要信息。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'train_{timestamp}.log')

    logger = logging.getLogger('LSEVGG')
    logger.setLevel(logging.DEBUG)

    # 文件处理器：记录所有 INFO 及以上信息
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-5s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    # 终端处理器：只输出 WARNING 及以上，避免干扰 tqdm 进度条
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    # 抑制第三方库日志噪音
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    return logger, log_path


def train():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    logger, log_path = setup_logger(cfg.log_dir)
    logger.info(f'日志文件: {log_path}')
    logger.info(f'设备: {cfg.device}')
    logger.info(f'超参数: epochs={cfg.epochs}, batch_size={cfg.batch_size}, lr={cfg.lr}, '
                f'momentum={cfg.momentum}, weight_decay={cfg.weight_decay}, '
                f'label_smoothing={cfg.label_smoothing}, warmup={cfg.warmup_epochs}, '
                f'scheduler={cfg.lr_scheduler}')
    logger.info(f'训练集: {cfg.train_dir}, 验证集: {cfg.val_dir}')

    # 构建训练/验证数据加载器和模型
    train_loader, val_loader, _ = get_loaders(cfg.batch_size)
    logger.info(f'数据加载完成: train_batches={len(train_loader)}, '
                f'val_batches={len(val_loader)}')
    model = build_model(num_classes=cfg.num_classes).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'模型参数量: {total_params:,}')
    # 使用带 label smoothing 的交叉熵，降低过拟合和过度自信预测
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    # 采用 SGD + momentum 作为优化器，适合图像分类任务
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr,
                          momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    # 余弦退火学习率调度器：在 warmup 之后逐步降低学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs,
                                  eta_min=1e-6)

    # AMP 混合精度：利用 RTX 4060 Tensor Core 加速训练并降低显存
    scaler = GradScaler()

    # 记录每轮指标，用于后续绘制训练曲线
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'lr': []}

    best_acc = 0.0
    no_improve = 0
    start_time = time.time()
    logger.info('开始训练...')

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        # Warmup 阶段：前几个 epoch 逐步升高学习率，避免训练一开始不稳定
        if epoch < cfg.warmup_epochs:
            lr = cfg.lr * (epoch+1) / cfg.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        # ── 训练阶段 ──
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.epochs}'):
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        train_acc = 100.*correct/total

        # ── 验证阶段 ──
        model.eval()
        val_loss, corr, tot = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
                with autocast():
                    outputs = model(imgs)
                    val_loss += criterion(outputs, labels).item()
                _, p = outputs.max(1)
                corr += p.eq(labels).sum().item()
                tot += labels.size(0)
        val_acc = 100.*corr/tot

        # Warmup 结束后，开始按余弦策略更新学习率
        if epoch >= cfg.warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # 终端输出
        print(f' Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.2f}%')

        # 日志文件输出（含时间戳与学习率）
        logger.info(f'Epoch {epoch+1:3d}/{cfg.epochs} | '
                    f'Train Loss: {train_loss/len(train_loader):.4f} | '
                    f'Train Acc: {train_acc:.2f}% | '
                    f'Val Loss: {val_loss/len(val_loader):.4f} | '
                    f'Val Acc: {val_acc:.2f}% | '
                    f'LR: {current_lr:.2e} | '
                    f'Time: {epoch_time:.1f}s')

        # 记录本轮指标到历史，便于事后绘制训练曲线
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # 保存完整检查点（含模型、优化器、调度器状态），便于恢复训练和推理
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
            }, os.path.join(cfg.save_dir, 'best_model.pth'))
            # 每次保存最佳模型时同步写出历史文件，防止中断丢失
            with open(os.path.join(cfg.log_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f'  -> 最佳模型已更新 (Val Acc: {best_acc:.2f}%)')
            no_improve = 0
        else:
            no_improve += 1
            # 连续若干轮验证集没有提升时提前停止，节省训练时间
            if no_improve >= cfg.early_stop_patience:
                logger.info(f'Early stopping at epoch {epoch+1} (no improvement for '
                            f'{cfg.early_stop_patience} epochs)')
                print('Early stopping!')
                break

    total_time = time.time() - start_time
    logger.info(f'训练完成 | 最佳验证准确率: {best_acc:.2f}% | '
                f'总耗时: {total_time:.1f}s ({total_time/60:.1f}min)')

    # 最终保存训练历史
    with open(os.path.join(cfg.log_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'训练完成，最佳验证准确率: {best_acc:.2f}%')
    print(f'日志文件: {log_path}')

if __name__ == '__main__':
    # 直接运行脚本时启动训练流程
    train()