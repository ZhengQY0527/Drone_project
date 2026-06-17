import torch
import json
import os
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, precision_recall_curve,
                              average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
from config import Config
from dataset import get_loaders
from model import build_model

# ═══════════════════════════════════════════════════════════════
#  论文级出版质量可视化样式
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'image.cmap': 'Blues',
})

OUTPUT_DIR = os.path.join('./figures', datetime.now().strftime('%Y%m%d_%H%M%S'))
# 45 类扩展调色板（Tab20 + 额外颜色，共 45 种）
import matplotlib.cm as cm
_COLORS20 = plt.cm.tab20.colors if hasattr(plt.cm, 'tab20') else plt.cm.get_cmap('tab20').colors
_COLORS20b = plt.cm.tab20b.colors if hasattr(plt.cm, 'tab20b') else plt.cm.get_cmap('tab20b').colors
_COLORS_EXTRA = ['#2166AC', '#D6604D', '#4DAF4A', '#FF7F00', '#984EA3',
                 '#A65628', '#F781BF', '#66C2A5', '#FC8D62', '#8DA0CB',
                 '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3',
                 '#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E',
                 '#E6AB02', '#A6761D', '#666666', '#1F78B4', '#B2DF8A',
                 '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00',
                 '#CAB2D6', '#6A3D9A', '#FFFF99', '#B15928', '#8DD3C7',
                 '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462',
                 '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD', '#CCEBC5']
COLORS = list(_COLORS20) + list(_COLORS20b) + _COLORS_EXTRA
COLORS = COLORS[:45]  # 确保恰好 45 种


def evaluate():
    cfg = Config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 加载模型 ──
    _, _, test_loader = get_loaders(cfg.batch_size)
    model = build_model(cfg.num_classes).to(cfg.device)
    checkpoint = torch.load('./checkpoints/best_model.pth', map_location=cfg.device,
                            weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # ── 收集预测结果与概率 ──
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(cfg.device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(outputs.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    print(classification_report(all_labels, all_preds,
                                target_names=cfg.class_names, digits=3))

    # ═══════════════════════════════════════════════════════════
    #  Fig 1: 混淆矩阵（原始计数 + 行归一化）
    #  45 类时需要大尺寸 + 小字号，避免数字重叠
    # ═══════════════════════════════════════════════════════════
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    n_cls = cfg.num_classes
    # 根据类别数自适应调整图大小和字号
    figsize_w = max(22, n_cls * 0.55)
    figsize_h = max(10, n_cls * 0.38)
    annot_font = max(3.5, 9 - n_cls * 0.12)

    fig, axes = plt.subplots(1, 2, figsize=(figsize_w * 2, figsize_h))
    for ax, data, fmt, title in [
        (axes[0], cm, 'd', '(a) Confusion Matrix — Counts'),
        (axes[1], cm_norm, '.1f', '(b) Confusion Matrix — Row-normalized'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=cfg.class_names,
                    yticklabels=cfg.class_names,
                    linewidths=0.1, linecolor='white',
                    vmin=0, vmax=data.max() if fmt == 'd' else 1,
                    annot_kws={'fontsize': annot_font},
                    ax=ax, cbar_kws={'shrink': 0.5})
        ax.set_title(title, fontweight='bold', pad=10, fontsize=14)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_confusion_matrix.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_confusion_matrix.png'))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    #  Fig 2: 每类别 Precision / Recall / F1-score 分组柱状图
    # ═══════════════════════════════════════════════════════════
    report = classification_report(all_labels, all_preds,
                                   target_names=cfg.class_names,
                                   output_dict=True, zero_division=0)
    precisions = [report[cls]['precision'] for cls in cfg.class_names]
    recalls = [report[cls]['recall'] for cls in cfg.class_names]
    f1s = [report[cls]['f1-score'] for cls in cfg.class_names]

    #  45 类时自动加宽，字号缩小
    fig_w = max(18, n_cls * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    x = np.arange(len(cfg.class_names))
    width = 0.25
    bars_p = ax.bar(x - width, precisions, width, label='Precision',
                    color=COLORS[0], edgecolor='white', linewidth=0.3)
    bars_r = ax.bar(x, recalls, width, label='Recall',
                    color=COLORS[1], edgecolor='white', linewidth=0.3)
    bars_f = ax.bar(x + width, f1s, width, label='F1-score',
                    color=COLORS[2], edgecolor='white', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(cfg.class_names, rotation=90, ha='center', fontsize=6)
    ax.set_ylabel('Score')
    ax.set_title('Per-class Precision, Recall & F1-score', fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fancybox=True)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_per_class_metrics.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_per_class_metrics.png'))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    #  Fig 3: 训练曲线（Loss + Accuracy + LR）
    # ═══════════════════════════════════════════════════════════
    history_path = os.path.join(cfg.log_dir, 'history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        epochs = np.arange(1, len(history['train_loss']) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        axes[0].plot(epochs, history['train_loss'], 'o-', color=COLORS[0],
                     markersize=2, linewidth=1.2, label='Train')
        axes[0].plot(epochs, history['val_loss'], 's-', color=COLORS[1],
                     markersize=2, linewidth=1.2, label='Validation')
        axes[0].set_title('(a) Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cross-Entropy Loss')
        axes[0].legend(frameon=True, fancybox=True)
        axes[0].grid(alpha=0.25, linewidth=0.5)

        axes[1].plot(epochs, history['train_acc'], 'o-', color=COLORS[0],
                     markersize=2, linewidth=1.2, label='Train')
        axes[1].plot(epochs, history['val_acc'], 's-', color=COLORS[1],
                     markersize=2, linewidth=1.2, label='Validation')
        best_epoch = np.argmax(history['val_acc'])
        best_acc = history['val_acc'][best_epoch]
        axes[1].axvline(x=best_epoch + 1, color='gray', linestyle='--',
                       linewidth=0.7, alpha=0.7)
        axes[1].annotate(f'Best: {best_acc:.1f}%',
                        xy=(best_epoch + 1, best_acc),
                        xytext=(best_epoch + 1 + 2, best_acc - 5),
                        fontsize=7, color='gray',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
        axes[1].set_title('(b) Accuracy', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend(frameon=True, fancybox=True)
        axes[1].grid(alpha=0.25, linewidth=0.5)

        axes[2].plot(epochs, history['lr'], 'D-', color=COLORS[3],
                     markersize=2, linewidth=1.2)
        axes[2].set_title('(c) Learning Rate Schedule', fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(alpha=0.25, linewidth=0.5)

        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_training_curves.pdf'), format='pdf')
        fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_training_curves.png'))
        plt.close(fig)
    else:
        print(f"未找到训练历史文件 {history_path}，跳过训练曲线图。")

    # ═══════════════════════════════════════════════════════════
    #  Fig 4: ROC 曲线（One-vs-Rest），带 AUC
    # ═══════════════════════════════════════════════════════════
    labels_bin = label_binarize(all_labels, classes=range(cfg.num_classes))

    fig, ax = plt.subplots(figsize=(14, 12))
    for i, cls in enumerate(cfg.class_names):
        if labels_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], linewidth=0.8,
                label=f'{cls}  (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.7, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — One-vs-Rest', fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fancybox=True,
              ncol=3, fontsize=4.5, columnspacing=0.5)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_roc_curves.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_roc_curves.png'))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    #  Fig 5: Precision-Recall 曲线（One-vs-Rest），带 AP
    # ═══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 12))
    for i, cls in enumerate(cfg.class_names):
        if labels_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(labels_bin[:, i],
                                                       all_probs[:, i])
        ap = average_precision_score(labels_bin[:, i], all_probs[:, i])
        ax.plot(recall, precision, color=COLORS[i % len(COLORS)], linewidth=0.8,
                label=f'{cls}  (AP={ap:.3f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves — One-vs-Rest', fontweight='bold')
    ax.legend(loc='lower left', frameon=True, fancybox=True,
              ncol=3, fontsize=4.5, columnspacing=0.5)
    ax.grid(alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_pr_curves.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_pr_curves.png'))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    #  Fig 6: 每类别样本数量分布图（辅助分析类别不平衡）
    # ═══════════════════════════════════════════════════════════
    class_counts = np.bincount(all_labels, minlength=cfg.num_classes)
    fig_w6 = max(16, n_cls * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w6, 5))
    bar_colors = [COLORS[i % len(COLORS)] for i in range(cfg.num_classes)]
    bars = ax.bar(range(cfg.num_classes), class_counts, color=bar_colors,
                  edgecolor='white', linewidth=0.3)
    ax.set_xticks(range(cfg.num_classes))
    ax.set_xticklabels(cfg.class_names, rotation=90, ha='center', fontsize=6)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Test Set Class Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    for bar, count in zip(bars, class_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(class_counts) * 0.01,
                str(count), ha='center', va='bottom', fontsize=5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_class_distribution.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_class_distribution.png'))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    #  Fig 7: 预测置信度分布 —— 正确 vs 错误样本
    # ═══════════════════════════════════════════════════════════
    max_confs = all_probs.max(axis=1)
    correct_mask = all_preds == all_labels
    confs_correct = max_confs[correct_mask]
    confs_wrong = max_confs[~correct_mask]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0, 1, 31)
    ax.hist(confs_correct, bins=bins, alpha=0.7, color=COLORS[0],
            label=f'Correct  (n={len(confs_correct)})', edgecolor='white', linewidth=0.3)
    ax.hist(confs_wrong, bins=bins, alpha=0.7, color=COLORS[1],
            label=f'Incorrect  (n={len(confs_wrong)})', edgecolor='white', linewidth=0.3)
    ax.axvline(x=confs_correct.mean(), color=COLORS[0], linestyle='--', linewidth=1,
               label=f'Mean correct={confs_correct.mean():.3f}')
    ax.axvline(x=confs_wrong.mean(), color=COLORS[1], linestyle='--', linewidth=1,
               label=f'Mean incorrect={confs_wrong.mean():.3f}')
    ax.set_xlabel('Maximum Softmax Probability')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Prediction Confidence Distribution', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, fontsize=8)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig7_confidence_dist.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig7_confidence_dist.png'))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    #  Fig 8: 每类别指标热力图（Precision / Recall / F1 / Accuracy）
    # ═══════════════════════════════════════════════════════════
    per_class_acc = []
    for i in range(cfg.num_classes):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc.append((all_preds[mask] == i).mean())
        else:
            per_class_acc.append(0)

    metrics_heatmap = np.column_stack([precisions, recalls, f1s, per_class_acc])
    fig_h8 = max(12, n_cls * 0.38)
    fig, ax = plt.subplots(figsize=(6, fig_h8))
    sns.heatmap(metrics_heatmap, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=0.5, linecolor='white',
                xticklabels=['Precision', 'Recall', 'F1-score', 'Accuracy'],
                yticklabels=cfg.class_names,
                annot_kws={'fontsize': 7},
                ax=ax, cbar_kws={'shrink': 0.6, 'label': 'Score'})
    ax.set_title('Per-Class Metrics Heatmap', fontweight='bold', pad=10)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Class')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig8_metrics_heatmap.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig8_metrics_heatmap.png'))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    #  Fig 9: t-SNE 特征嵌入可视化（分类器倒数第二层输出）
    # ═══════════════════════════════════════════════════════════
    # 提取 penultimate 层特征
    features_store = []
    def hook_fn(module, input, output):
        features_store.append(output.detach().cpu().numpy())

    # LSEVGG GAP 层输出 512 维特征，作为 t-SNE 输入
    handle = model.gap.register_forward_hook(hook_fn)

    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(cfg.device)
            model(imgs)
    handle.remove()

    features = np.concatenate(features_store)
    # GAP 输出为 [B, 512, 1, 1]，展平为 [B, 512]
    if features.ndim == 4:
        features = features.reshape(features.shape[0], -1)

    # 512 维直接用 t-SNE，无需 PCA 降维
    n_tsne = min(3000, len(features))
    rng = np.random.RandomState(42)
    subset_idx = rng.choice(len(features), n_tsne, replace=False)
    features_sub = features[subset_idx]
    labels_sub = all_labels[subset_idx]

    features_tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                         max_iter=1000).fit_transform(features_sub)

    fig, ax = plt.subplots(figsize=(16, 12))
    for i, cls in enumerate(cfg.class_names):
        mask = labels_sub == i
        ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                   c=[COLORS[i % len(COLORS)]], label=cls, s=4, alpha=0.7,
                   edgecolors='none', rasterized=True)
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_title('t-SNE Visualization of Penultimate Layer Features', fontweight='bold')
    ax.legend(loc='lower left', frameon=True, fancybox=True,
              ncol=5, fontsize=4.5, markerscale=1.5, columnspacing=0.5)
    ax.grid(alpha=0.15, linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig9_tsne_embedding.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig9_tsne_embedding.png'), dpi=300)
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    #  Fig 10: 可靠性曲线（Reliability Diagram）—— 校准评估
    # ═══════════════════════════════════════════════════════════
    confs = all_probs.max(axis=1)
    correct_int = (all_preds == all_labels).astype(int)
    prob_true, prob_pred = calibration_curve(correct_int, confs,
                                              n_bins=15, strategy='uniform')

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.7, alpha=0.4, label='Perfectly Calibrated')
    ax.plot(prob_pred, prob_true, 'o-', color=COLORS[0], markersize=4,
            linewidth=1.5, label='LSEVGG')
    ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.15, color=COLORS[0])

    # 添加每个 bin 的样本量柱状图（背景）
    bin_counts, bin_edges = np.histogram(confs, bins=15, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax2 = ax.twinx()
    ax2.bar(bin_centers, bin_counts, width=0.05, alpha=0.15, color='gray',
            edgecolor='gray', linewidth=0.3)
    ax2.set_ylabel('Samples per Bin', fontsize=9, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)

    ax.set_xlim(0, 1)
    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Observed Accuracy')
    ax.set_title('Reliability Diagram (Calibration Curve)', fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig10_reliability.pdf'), format='pdf')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig10_reliability.png'))
    plt.close(fig)

    # ── 输出 ECE (Expected Calibration Error) ──
    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (confs >= bin_edges[i]) & (confs <= bin_edges[i + 1])
        n_b = mask.sum()
        if n_b > 0:
            ece += (n_b / len(confs)) * abs(correct_int[mask].mean() - confs[mask].mean())
    print(f'ECE (Expected Calibration Error): {ece:.4f}')

    print(f"所有图表已保存至 {OUTPUT_DIR}/ 目录。")


if __name__ == '__main__':
    evaluate()
