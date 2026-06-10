"""
NWPU-RESISC45 数据集划分脚本

将原始 45 个类别文件夹（每类 700 张 256×256 图像）按 60% / 20% / 20%
划分为 train / val / test，输出目录结构兼容 torchvision.datasets.ImageFolder。
"""

import os
import random
from shutil import copyfile

random.seed(42)

SRC = "./NWPU-RESISC45"
DST = "./data_slices_nwpu"

# 获取所有类别名（按字母排序，保证顺序可复现）
class_names = sorted([
    d for d in os.listdir(SRC)
    if os.path.isdir(os.path.join(SRC, d))
])
print(f"检测到 {len(class_names)} 个类别: {class_names}")

# 创建输出目录结构
for split in ['train', 'val', 'test']:
    for cls in class_names:
        os.makedirs(os.path.join(DST, split, cls), exist_ok=True)

# 逐类别划分
total_counts = {'train': 0, 'val': 0, 'test': 0}

for cls in class_names:
    cls_dir = os.path.join(SRC, cls)
    images = sorted(os.listdir(cls_dir))
    # 过滤非图像文件
    images = [im for im in images if im.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.bmp'))]
    random.shuffle(images)

    n = len(images)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split_name, imgs in splits.items():
        dst_cls_dir = os.path.join(DST, split_name, cls)
        for im in imgs:
            src_path = os.path.join(cls_dir, im)
            dst_path = os.path.join(dst_cls_dir, im)
            try:
                os.link(src_path, dst_path)       # 优先硬链接，节省磁盘
            except OSError:
                copyfile(src_path, dst_path)       # 失败时退化为复制
        total_counts[split_name] += len(imgs)

    print(f"  {cls:30s}  train={len(splits['train']):3d}  val={len(splits['val']):3d}  test={len(splits['test']):3d}")

print(f"\n划分完成！总计: train={total_counts['train']}, val={total_counts['val']}, test={total_counts['test']}")
print(f"数据已保存至: {DST}/")
