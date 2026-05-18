import os
import cv2
import numpy as np
from config import Config

def compute_mean_std(slice_root):
    """遍历切片数据集，统计 RGB 通道的均值和标准差。

    这个函数主要用于离线计算归一化参数：如果你更换了数据集或切片方式，
    应该重新统计 mean/std，再更新到 dataset.py 中。
    """
    cfg = Config()
    all_means, all_stds = [], []
    for split in ['train']:
        dir = os.path.join(slice_root, split)
        for cls in os.listdir(dir):
            cls_dir = os.path.join(dir, cls)
            for img_name in os.listdir(cls_dir):
                # 读取图像后转换为 RGB，并将像素归一化到 [0, 1]
                img = cv2.imread(os.path.join(cls_dir, img_name))
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                # 逐张图统计通道均值和标准差，最后再求整体平均
                all_means.append(img.mean(axis=(0,1)))
                all_stds.append(img.std(axis=(0,1)))
    mean = np.mean(all_means, axis=0)
    std  = np.mean(all_stds, axis=0)
    print(f"Mean: {mean}, Std: {std}")
    return mean, std