"""
Baseline 模型可视化 — 支持 MobileNetV2 / VGG16
用法: python evaluate_baseline.py MobileNetV2
"""
import torch, json, os, sys, numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, precision_recall_curve,
                              average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ═══════════════════════════════════════
#  配置
# ═══════════════════════════════════════
CLASS_NAMES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool'
]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN, STD = [0.385, 0.381, 0.375], [0.130, 0.125, 0.127]

OUTPUT_DIR = os.path.join('./figures_baseline', datetime.now().strftime('%Y%m%d_%H%M%S'))

# ═══════════════════════════════════════
#  样式
# ═══════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 11, 'axes.titlesize': 13, 'legend.fontsize': 7.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    'image.cmap': 'Blues',
})
import matplotlib.cm as cm
_T20 = plt.cm.tab20.colors if hasattr(plt.cm, 'tab20') else plt.cm.get_cmap('tab20').colors
_T20b = plt.cm.tab20b.colors if hasattr(plt.cm, 'tab20b') else plt.cm.get_cmap('tab20b').colors
COLORS = (list(_T20) + list(_T20b))[:NUM_CLASSES]

# ═══════════════════════════════════════
#  数据 & 模型加载
# ═══════════════════════════════════════
def load_model_and_data(model_name):
    """加载指定模型和 DOTA 测试集"""
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)
    ])
    test_ds = datasets.ImageFolder(root='./data_slices/test', transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # 构建模型
    if model_name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
        ckpt_path = './checkpoints_baseline/MobileNetV2_best.pth'
    elif model_name == 'VGG16':
        model = models.vgg16(weights=None)
        model.classifier[6] = torch.nn.Linear(4096, NUM_CLASSES)
        ckpt_path = './checkpoints_baseline/VGG16_best.pth'
    else:
        raise ValueError(f"未知模型: {model_name}，可选 MobileNetV2 / VGG16")

    model = model.to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"模型: {model_name} | 最佳 Val Acc: {ckpt.get('best_acc', 'N/A')} | Device: {DEVICE}")
    return model, test_loader

# ═══════════════════════════════════════
#  推理
# ═══════════════════════════════════════
def collect_predictions(model, test_loader):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(outputs.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())
    return (np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_probs))

# ═══════════════════════════════════════
#  入口
# ═══════════════════════════════════════
def evaluate_baseline(model_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, test_loader = load_model_and_data(model_name)
    all_preds, all_labels, all_probs = collect_predictions(model, test_loader)

    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3))

    n_cls = NUM_CLASSES

    # Fig 1: 混淆矩阵
    cm_raw = confusion_matrix(all_labels, all_preds)
    cm_norm = cm_raw.astype('float') / cm_raw.sum(axis=1, keepdims=True)
    fw = max(14, n_cls * 0.6)
    fh = max(8, n_cls * 0.35)
    af = max(5, 11 - n_cls * 0.35)
    fig, axes = plt.subplots(1, 2, figsize=(fw * 2, fh))
    for ax, data, fmt, title in [
        (axes[0], cm_raw, 'd', '(a) Confusion Matrix — Counts'),
        (axes[1], cm_norm, '.2f', '(b) Confusion Matrix — Row-normalized'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=0.1, linecolor='white',
                    vmin=0, vmax=data.max() if fmt == 'd' else 1,
                    annot_kws={'fontsize': af}, ax=ax, cbar_kws={'shrink': 0.6})
        ax.set_title(title, fontweight='bold', pad=10, fontsize=14)
        ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_confusion_matrix.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_confusion_matrix.pdf'))
    plt.close(fig)

    # Fig 2: P/R/F1 柱状图
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    precisions = [report[c]['precision'] for c in CLASS_NAMES]
    recalls = [report[c]['recall'] for c in CLASS_NAMES]
    f1s = [report[c]['f1-score'] for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(max(14, n_cls * 0.6), 6))
    x = np.arange(n_cls); width = 0.25
    ax.bar(x - width, precisions, width, label='Precision', color=COLORS[0], edgecolor='white', linewidth=0.3)
    ax.bar(x, recalls, width, label='Recall', color=COLORS[1], edgecolor='white', linewidth=0.3)
    ax.bar(x + width, f1s, width, label='F1-score', color=COLORS[2], edgecolor='white', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Score'); ax.set_title('Per-class Precision, Recall & F1-score', fontweight='bold')
    ax.legend(loc='lower right'); ax.set_ylim(0, 1.08)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_per_class_metrics.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_per_class_metrics.pdf'))
    plt.close(fig)

    # Fig 3: 训练曲线
    import glob
    history_files = glob.glob(f'./logs_baseline/{model_name}_history.json')
    if history_files:
        with open(history_files[0], 'r') as f:
            history = json.load(f)
        epochs = np.arange(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].plot(epochs, history['train_loss'], 'o-', color=COLORS[0], markersize=2, lw=1.2, label='Train')
        axes[0].plot(epochs, history['val_loss'], 's-', color=COLORS[1], markersize=2, lw=1.2, label='Validation')
        axes[0].set_title('(a) Loss', fontweight='bold'); axes[0].legend(); axes[0].grid(alpha=0.25, lw=0.5)
        axes[1].plot(epochs, history['train_acc'], 'o-', color=COLORS[0], markersize=2, lw=1.2, label='Train')
        axes[1].plot(epochs, history['val_acc'], 's-', color=COLORS[1], markersize=2, lw=1.2, label='Validation')
        best_epoch = np.argmax(history['val_acc'])
        axes[1].axvline(x=best_epoch + 1, color='gray', linestyle='--', lw=0.7)
        axes[1].set_title('(b) Accuracy', fontweight='bold'); axes[1].legend(); axes[1].grid(alpha=0.25, lw=0.5)
        axes[2].plot(epochs, history['lr'], 'D-', color=COLORS[3], markersize=2, lw=1.2)
        axes[2].set_title('(c) LR Schedule', fontweight='bold'); axes[2].set_yscale('log'); axes[2].grid(alpha=0.25, lw=0.5)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_training_curves.png'))
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_training_curves.pdf'))
        plt.close(fig)

    # Fig 4: ROC
    labels_bin = label_binarize(all_labels, classes=range(n_cls))
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, cls in enumerate(CLASS_NAMES):
        if labels_bin[:, i].sum() == 0: continue
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        ax.plot(fpr, tpr, color=COLORS[i], lw=0.8, label=f'{cls} (AUC={auc(fpr,tpr):.3f})')
    ax.plot([0,1],[0,1],'k--',lw=0.7,alpha=0.4)
    ax.set_xlim(0,1); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curves — One-vs-Rest', fontweight='bold')
    ax.legend(loc='lower right', ncol=3, fontsize=5.5); ax.grid(alpha=0.2, lw=0.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_roc_curves.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_roc_curves.pdf'))
    plt.close(fig)

    # Fig 5: PR 曲线
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, cls in enumerate(CLASS_NAMES):
        if labels_bin[:, i].sum() == 0: continue
        prec, rec, _ = precision_recall_curve(labels_bin[:, i], all_probs[:, i])
        ap = average_precision_score(labels_bin[:, i], all_probs[:, i])
        ax.plot(rec, prec, color=COLORS[i], lw=0.8, label=f'{cls} (AP={ap:.3f})')
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves', fontweight='bold')
    ax.legend(loc='lower left', ncol=3, fontsize=5.5); ax.grid(alpha=0.2, lw=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_pr_curves.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_pr_curves.pdf'))
    plt.close(fig)

    # Fig 6: 类别分布
    counts = np.bincount(all_labels, minlength=n_cls)
    fig, ax = plt.subplots(figsize=(max(12, n_cls*0.5), 5))
    bars = ax.bar(range(n_cls), counts, color=[COLORS[i] for i in range(n_cls)], edgecolor='white', lw=0.3)
    ax.set_xticks(range(n_cls)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Samples'); ax.set_title('Test Set Class Distribution', fontweight='bold')
    for b, c in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(counts)*0.01, str(c), ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_class_distribution.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_class_distribution.pdf'))
    plt.close(fig)

    # Fig 7: 置信度分布
    max_confs = all_probs.max(axis=1)
    correct_mask = all_preds == all_labels
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0, 1, 31)
    ax.hist(max_confs[correct_mask], bins=bins, alpha=0.7, color=COLORS[0],
            label=f'Correct (n={correct_mask.sum()})', edgecolor='white', lw=0.3)
    ax.hist(max_confs[~correct_mask], bins=bins, alpha=0.7, color=COLORS[1],
            label=f'Incorrect (n={(~correct_mask).sum()})', edgecolor='white', lw=0.3)
    ax.set_xlabel('Max Softmax Probability'); ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.25, lw=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig7_confidence_dist.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig7_confidence_dist.pdf'))
    plt.close(fig)

    # Fig 8: 热力图
    per_class_acc = []
    for i in range(n_cls):
        mask = all_labels == i
        per_class_acc.append((all_preds[mask]==i).mean() if mask.sum()>0 else 0)
    hm = np.column_stack([precisions, recalls, f1s, per_class_acc])
    fig, ax = plt.subplots(figsize=(6, max(8, n_cls*0.35)))
    sns.heatmap(hm, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                linewidths=0.5, linecolor='white',
                xticklabels=['Precision','Recall','F1','Accuracy'],
                yticklabels=CLASS_NAMES, annot_kws={'fontsize': 8},
                ax=ax, cbar_kws={'shrink': 0.6, 'label': 'Score'})
    ax.set_title('Per-Class Metrics Heatmap', fontweight='bold', pad=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig8_metrics_heatmap.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig8_metrics_heatmap.pdf'))
    plt.close(fig)

    # Fig 9: t-SNE (取 MobileNetV2 最后的 AvgPool 之前)
    # MobileNetV2 结构: features → AdaptiveAvgPool2d → classifier
    features_store = []
    def hook_fn(m, inp, out):
        features_store.append(out.detach().cpu().numpy())
    if model_name == 'MobileNetV2':
        # hook features 层输出
        handle = model.features.register_forward_hook(hook_fn)
    else:
        handle = model.features.register_forward_hook(hook_fn)

    with torch.no_grad():
        for imgs, _ in test_loader:
            model(imgs.to(DEVICE))
    handle.remove()
    feats = np.concatenate(features_store)
    if feats.ndim == 4: feats = feats.mean(axis=(2,3))  # GAP: [B,C,H,W]→[B,C]

    n_tsne = min(2000, len(feats))
    subset_idx = np.random.RandomState(42).choice(len(feats), n_tsne, replace=False)
    feats_2d = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=800).fit_transform(feats[subset_idx])
    labels_sub = all_labels[subset_idx]
    fig, ax = plt.subplots(figsize=(14, 11))
    for i, cls in enumerate(CLASS_NAMES):
        mask = labels_sub == i
        ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1], c=[COLORS[i]], label=cls, s=5, alpha=0.7, edgecolors='none', rasterized=True)
    ax.set_xlabel('t-SNE Dim 1'); ax.set_ylabel('t-SNE Dim 2')
    ax.set_title('t-SNE Feature Embedding', fontweight='bold')
    ax.legend(loc='lower left', ncol=3, fontsize=5.5, markerscale=1.5)
    ax.grid(alpha=0.15, lw=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig9_tsne_embedding.png'), dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig9_tsne_embedding.pdf'))
    plt.close(fig)

    # Fig 10: 可靠性曲线
    confs = all_probs.max(axis=1)
    correct_int = (all_preds == all_labels).astype(int)
    prob_true, prob_pred = calibration_curve(correct_int, confs, n_bins=15, strategy='uniform')
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0,1],[0,1],'k--',lw=0.7,alpha=0.4,label='Perfectly Calibrated')
    ax.plot(prob_pred, prob_true, 'o-', color=COLORS[0], markersize=4, lw=1.5, label=model_name)
    ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.15, color=COLORS[0])
    bin_c, bin_e = np.histogram(confs, bins=15, range=(0,1))
    ax2 = ax.twinx()
    ax2.bar((bin_e[:-1]+bin_e[1:])/2, bin_c, width=0.05, alpha=0.15, color='gray')
    ax2.set_ylabel('Samples per Bin', fontsize=9, color='gray')
    ax.set_xlim(0,1); ax.set_xlabel('Mean Predicted Confidence'); ax.set_ylabel('Observed Accuracy')
    ax.set_title('Reliability Diagram', fontweight='bold')
    ax.legend(loc='upper left'); ax.grid(alpha=0.2, lw=0.5); ax.set_aspect('equal')
    # ECE
    ece = 0.0
    for i in range(15):
        mask = (confs >= bin_e[i]) & (confs < bin_e[i+1])
        if i == 14: mask = (confs >= bin_e[i]) & (confs <= bin_e[i+1])
        n_b = mask.sum()
        if n_b > 0: ece += (n_b/len(confs))*abs(correct_int[mask].mean()-confs[mask].mean())
    ax.set_title(f'Reliability Diagram (ECE={ece:.4f})', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig10_reliability.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig10_reliability.pdf'))
    plt.close(fig)

    print(f"\n所有图表已保存至 {OUTPUT_DIR}/")

if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'MobileNetV2'
    evaluate_baseline(name)
