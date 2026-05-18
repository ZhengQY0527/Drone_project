import os
import cv2
import numpy as np
from config import Config
from sklearn.model_selection import train_test_split

def parse_dota_label(label_path):
    """解析 DOTA 标注文件。

    返回格式为 [(类别名, 多边形点列表), ...]，其中多边形点按 DOTA 的四点
    顺序保存。若目标被标记为 difficult，则直接跳过。
    """
    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9: continue
            x1,y1,x2,y2,x3,y3,x4,y4 = map(float, parts[:8])
            cls_name = parts[8]
            difficult = parts[9] if len(parts) > 9 else '0'
            if difficult == '1':
                continue
            objects.append((cls_name, [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]))
    return objects

def polygon_bbox(poly):
    """将四边形外接框转换为轴对齐矩形，便于后续裁剪图像区域。"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def crop_and_save(image_dir, label_dir, output_root, cfg):
    """遍历一个原始数据文件夹，将目标裁剪成分类样本并保存到输出目录。"""
    os.makedirs(output_root, exist_ok=True)
    # 先为每个类别创建子文件夹，保证 ImageFolder 读取时目录结构完整
    for cls in cfg.class_names:
        os.makedirs(os.path.join(output_root, cls), exist_ok=True)

    count = {cls:0 for cls in cfg.class_names}
    for img_file in os.listdir(image_dir):
        # 只处理常见图像格式，避免把其他文件误当成图片读入
        if not img_file.lower().endswith(('.jpg','.png','.tif')):
            continue
        img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_name + '.txt')
        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)
        if image is None: continue
        objs = parse_dota_label(label_path)

        for cls_name, poly in objs:
            # 只保留配置文件中定义的类别，其余标签直接忽略
            if cls_name not in cfg.class_names:
                continue
            x1,y1,x2,y2 = polygon_bbox(poly)
            # 对外接框做适度外扩，尽量保留目标周围上下文信息
            h,w = image.shape[:2]
            x1 = max(0, x1 - cfg.margin)
            y1 = max(0, y1 - cfg.margin)
            x2 = min(w, x2 + cfg.margin)
            y2 = min(h, y2 + cfg.margin)
            # 过滤过小区域，减少噪声样本对训练的干扰
            if (x2-x1)*(y2-y1) < cfg.min_area:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # 统一裁剪图尺寸，方便后续直接送入分类网络
            crop = cv2.resize(crop, (cfg.slice_size, cfg.slice_size))
            # 同一原图中的同类目标用序号区分，避免文件名重复覆盖
            save_name = f"{img_name}_{count[cls_name]:04d}.jpg"
            cv2.imwrite(os.path.join(output_root, cls_name, save_name), crop)
            count[cls_name] += 1

    # 打印每个类别裁剪出来的样本数量，便于检查数据分布是否异常
    total = sum(count.values())
    print(f"[{output_root}] 裁剪完成，各类别数量：")
    for cls in cfg.class_names:
        print(f"  {cls}: {count[cls]}")
    print(f"  总计: {total}")

def main():
    cfg = Config()
    # 先把训练集和验证集原图中的目标全部裁成切片，合并成一个总数据池
    all_data_dir = "./all_slices"   # 临时总池
    os.makedirs(all_data_dir, exist_ok=True)
    for cls in cfg.class_names:
        os.makedirs(os.path.join(all_data_dir, cls), exist_ok=True)

    # 分别处理原始训练集与验证集，方便后续统一重新划分 train/val/test
    print("裁剪 DOTA 训练集...")
    crop_and_save(cfg.raw_train_image_dir, cfg.raw_train_label_dir, all_data_dir, cfg)
    print("裁剪 DOTA 验证集...")
    crop_and_save(cfg.raw_val_image_dir, cfg.raw_val_label_dir, all_data_dir, cfg)

    # 按类别分别划分 train/val/test，避免类别分布被打乱
    print("划分数据集 train/val/test ...")
    random_seed = 42
    for cls in cfg.class_names:
        cls_dir = os.path.join(all_data_dir, cls)
        images = os.listdir(cls_dir)
        if len(images) == 0: continue
        # 先划出 30% 作为临时集合，再从中拆分验证集和测试集
        train_list, temp = train_test_split(images, test_size=0.3, random_state=random_seed)
        val_list, test_list = train_test_split(temp, test_size=1/3, random_state=random_seed)

        # 将文件放入目标目录；优先使用硬链接节省磁盘空间，失败时退化为复制
        for img, split in zip([train_list, val_list, test_list],
                              [cfg.train_dir, cfg.val_dir, cfg.test_dir]):
            dst_cls_dir = os.path.join(split, cls)
            os.makedirs(dst_cls_dir, exist_ok=True)
            for im in img:
                src = os.path.join(cls_dir, im)
                dst = os.path.join(dst_cls_dir, im)
                try:
                    os.link(src, dst)
                except OSError:
                    from shutil import copyfile
                    copyfile(src, dst)

    print("数据预处理完成！")
    # 如需清理临时总池，可以取消下一行注释；删除前请确认数据已经分拣完成
    # import shutil; shutil.rmtree(all_data_dir)

if __name__ == "__main__":
    main()