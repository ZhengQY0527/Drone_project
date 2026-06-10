import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import Config

cfg = Config()

# 数据集归一化参数：这里使用的是经验估计值。
# 如果你重新统计了切片数据集的 mean/std，可以在这里同步更新。
mean = [0.3678, 0.3808, 0.3435]
std  = [0.1453, 0.1356, 0.1321]

# 训练集增强：通过随机裁剪、翻转、旋转和颜色扰动提升泛化能力
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 验证/测试集只做确定性的尺寸统一和归一化，避免引入随机性
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def get_loaders(batch_size=64):
    # ImageFolder 会自动读取“类别子文件夹名”作为标签索引
    train_dataset = datasets.ImageFolder(root=cfg.train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=cfg.val_dir,   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(root=cfg.test_dir,  transform=eval_transform)

    # num_workers 和 pin_memory 主要用于加速 GPU 训练时的数据加载
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader