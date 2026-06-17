"""
DOTA Baseline 对比训练: MobileNetV2 + VGG16
与 LSEVGG v3 统一训练配置，公平对比
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os, json, logging, time
from datetime import datetime

torch.backends.cudnn.benchmark = True

# ═══════════════════════════════
#  DOTA 配置
# ═══════════════════════════════
CLASS_NAMES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool'
]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 16
EPOCHS = 50
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 5
EARLY_STOP_PATIENCE = 15

# DOTA 归一化参数
MEAN = [0.385, 0.381, 0.375]
STD  = [0.130, 0.125, 0.127]

# ═══════════════════════════════
#  数据加载
# ═══════════════════════════════
def get_dota_loaders():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    train_ds = datasets.ImageFolder(root='./data_slices/train', transform=train_transform)
    val_ds   = datasets.ImageFolder(root='./data_slices/val',   transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

# ═══════════════════════════════
#  模型工厂
# ═══════════════════════════════
def build_mobilenetv2():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # MobileNetV2 最后是 classifier[1] = Linear(1280→1000)，替换为 15 类
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    return model

def build_vgg16():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # VGG16 最后是 classifier[6] = Linear(4096→1000)，替换为 15 类
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    return model

# ═══════════════════════════════
#  训练函数
# ═══════════════════════════════
def train_one_model(model, model_name, save_dir, log_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 日志
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(log_dir, f'{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)

    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'{model_name} 参数量: {total_params:,}')

    train_loader, val_loader = get_dota_loaders()
    logger.info(f'DOTA 数据: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}')

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scaler = GradScaler()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_acc = 0.0
    no_improve = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        # Warmup
        if epoch < WARMUP_EPOCHS:
            lr = LR * (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        # ── Train ──
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{EPOCHS}'):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        train_acc = 100. * correct / total

        # ── Val ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(imgs)
                    val_loss += criterion(outputs, labels).item()
                correct += outputs.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        val_acc = 100. * correct / total

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | '
                    f'Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | '
                    f'Val Acc: {val_acc:.2f}% | LR: {current_lr:.2e}')

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'best_acc': best_acc, 'epoch': epoch},
                       os.path.join(save_dir, f'{model_name}_best.pth'))
            with open(os.path.join(log_dir, f'{model_name}_history.json'), 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f'  -> 最佳模型更新 (Val Acc: {best_acc:.2f}%)')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

    total_time = time.time() - start_time
    logger.info(f'训练完成 | 最佳验证准确率: {best_acc:.2f}% | 总耗时: {total_time:.1f}s ({total_time/60:.1f}min)')
    with open(os.path.join(log_dir, f'{model_name}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return best_acc, total_params, total_time

# ═══════════════════════════════
#  主流程
# ═══════════════════════════════
if __name__ == '__main__':
    print("=" * 70)
    print("DOTA Baseline 对比实验")
    print(f"统一配置: batch={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}, cosine+{WARMUP_EPOCHS}warmup, label_smooth={LABEL_SMOOTHING}")
    print("=" * 70)

    results = {}

    # ── MobileNetV2 ──
    print("\n[1/2] 训练 MobileNetV2 ...")
    acc, params, t = train_one_model(
        build_mobilenetv2(), 'MobileNetV2',
        save_dir='./checkpoints_baseline', log_dir='./logs_baseline'
    )
    results['MobileNetV2'] = {'acc': acc, 'params': params, 'time': t}

    # ── VGG16 ──
    print("\n[2/2] 训练 VGG16 ...")
    acc, params, t = train_one_model(
        build_vgg16(), 'VGG16',
        save_dir='./checkpoints_baseline', log_dir='./logs_baseline'
    )
    results['VGG16'] = {'acc': acc, 'params': params, 'time': t}

    # ── 汇总对比 (加入 LSEVGG v3 的已知结果) ──
    print("\n" + "=" * 70)
    print("DOTA 15 类 Baseline 对比汇总")
    print("=" * 70)
    print(f"{'模型':<20} {'参数量':>12} {'最佳 Val Acc':>14} {'训练时间':>12}")
    print("-" * 60)

    # LSEVGG v3 已知结果
    print(f"{'LSEVGG v3':<20} {2720000:>12,} {'97.86%':>14} {'206.6 min':>12}")

    for name, r in results.items():
        print(f"{name:<20} {r['params']:>12,} {r['acc']:>13.2f}% {r['time']/60:>11.1f} min")

    print("-" * 60)
    print("LSEVGG v1 (121M): 98.63% — 仅作参考，不在本次公平对比范围")
