import os

class Config:
    # 原始 DOTA 数据路径：用于读取未切片的训练集和验证集图像及标注文件
    raw_train_image_dir = "./DOTA/train/images"      # 原DOTA训练图像文件夹
    raw_train_label_dir = "./DOTA/train/labelTxt"    # 原DOTA标注文件夹(TXT)
    raw_val_image_dir   = "./DOTA/val/images"
    raw_val_label_dir   = "./DOTA/val/labelTxt"

    # 切片后的数据保存根目录：预处理后会按 train/val/test 和类别名组织
    slice_root = "./data_slices_nwpu"
    train_dir = os.path.join(slice_root, "train")
    val_dir   = os.path.join(slice_root, "val")
    test_dir  = os.path.join(slice_root, "test")

    # NWPU-RESISC45 的 45 个类别
    class_names = [
        'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach',
        'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',
        'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway',
        'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection',
        'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park',
        'mountain', 'overpass', 'palace', 'parking_lot', 'railway',
        'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway',
        'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium',
        'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland'
    ]
    num_classes = len(class_names)

    # 图像切片与裁剪相关参数：用于把旋转框目标裁成统一大小的分类样本
    slice_size = 224                # 输出统一尺寸
    margin = 10                     # 裁剪时外扩像素（保留上下文）
    min_area = 32*32                # 最小切片面积（像素），过滤极小框

    # 训练超参数：根据显存和数据规模可继续调整
    batch_size = 16
    epochs = 50
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    label_smoothing = 0.1
    warmup_epochs = 5               # 必须 < epochs，约为 epochs 的 10%
    lr_scheduler = 'cosine'         # 'cosine' 或 'plateau'
    early_stop_patience = 15
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

    # 训练过程输出目录：保存最优模型和日志文件
    save_dir = "./checkpoints"
    log_dir = "./logs"