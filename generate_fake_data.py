import os
import numpy as np
from PIL import Image

cfg = {
    'class_names': [
        'plane', 'ship', 'storage-tank', 'baseball-diamond',
        'tennis-court', 'basketball-court', 'ground-track-field',
        'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
        'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool'
    ],
    'data_root': './data_slices'
}

def generate():
    for split in ['train', 'val', 'test']:
        for cls in cfg['class_names']:
            cls_dir = os.path.join(cfg['data_root'], split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            # 每个类生成 5 张假图片（训练/验证/测试各5张，后续可按需调整数量）
            for i in range(5):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                Image.fromarray(img).save(os.path.join(cls_dir, f'{cls}_{i:04d}.jpg'))

    print("假数据生成完毕，路径：", cfg['data_root'])

if __name__ == '__main__':
    generate()