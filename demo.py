"""
LSEVGG 推理 Demo — 输入一张图片，输出分类结果
用法: python demo.py <图片路径>
示例: python demo.py ./NWPU-RESISC45/airplane/airplane_001.jpg
"""

import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import build_model

# ── 配置 ──
CLASS_NAMES = [
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
NUM_CLASSES = len(CLASS_NAMES)
CHECKPOINT = './checkpoints_nwpu/best_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MEAN = [0.3678, 0.3808, 0.3435]
STD  = [0.1453, 0.1356, 0.1321]

# ── 图像预处理 ──
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ── 加载模型 ──
model = build_model(num_classes=NUM_CLASSES).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model.eval()
print(f"模型已加载 ({DEVICE})，共 {NUM_CLASSES} 个类别\n")

# ── 推理函数 ──
def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

    # Top-5 结果
    top5_prob, top5_idx = probs.topk(5, dim=1)

    print(f"图片: {image_path}")
    print(f"预测结果 (Top-5):")
    for i in range(5):
        cls = CLASS_NAMES[top5_idx[0, i].item()]
        pct = top5_prob[0, i].item() * 100
        bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        print(f"  {i+1}. {cls:25s}  {pct:5.1f}%  {bar}")

    print(f"\n最终预测: {CLASS_NAMES[top5_idx[0, 0].item()]} "
          f"(置信度 {top5_prob[0, 0].item()*100:.1f}%)")

# ── 入口 ──
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python demo.py <图片路径>")
        print("示例: python demo.py ./NWPU-RESISC45/airplane/airplane_001.jpg")
        sys.exit(1)

    predict(sys.argv[1])
