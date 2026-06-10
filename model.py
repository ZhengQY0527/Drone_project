import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力。

    全局池化 → FC→ReLU→FC→Sigmoid 学习通道权重，增强有效通道、抑制噪声。
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：Depthwise → BN → SiLU → Pointwise → BN → SiLU。

    SiLU (Swish) 相比 ReLU 保留负值小梯度，避免深层神经元"死亡"。
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride,
            padding=1, groups=in_ch, bias=False
        )
        self.bn_depth = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_point = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn_depth(self.depthwise(x)))
        x = self.act(self.bn_point(self.pointwise(x)))
        return x


class ResidualBlock(nn.Module):
    """残差块：两个 DSC + SE 注意力 + 跳跃连接。

    跳跃连接为梯度提供"高速通道"，即使网络很深，浅层也能收到有效梯度，
    从而更好地学习细粒度特征（如 small-vehicle 与 large-vehicle 的纹理差异）。
    """
    def __init__(self, in_ch, out_ch, use_se=True, se_reduction=16):
        super(ResidualBlock, self).__init__()
        self.dsc1 = DepthwiseSeparableConv(in_ch, out_ch)
        self.dsc2 = DepthwiseSeparableConv(out_ch, out_ch)
        self.se = SEBlock(out_ch, reduction=se_reduction) if use_se else nn.Identity()
        self.act = nn.SiLU(inplace=True)

        # 通道数变化时，用 1×1 卷积对齐跳跃连接的维度
        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.dsc1(x)          # 内含 BN→SiLU
        out = self.dsc2(out)        # 内含 BN→SiLU
        out = self.se(out)          # 通道注意力
        out = out + identity        # 残差连接
        out = self.act(out)         # 融合后激活
        return out


class LSEVGG(nn.Module):
    """LSEVGG v3：残差 SE-VGG + GAP 分类头。

    相比 v1 的改进：
    1. 残差跳跃连接 — 梯度直达浅层，细粒度特征学习能力显著提升
    2. SiLU (Swish) 激活 — 消除 ReLU 神经元死亡问题
    3. GAP + 轻量分类头 — 参数量从 121M → ~2.2M，大幅减少过拟合

    输入: [B, 3, 224, 224]    输出: [B, num_classes]
    """

    def __init__(self, num_classes=15):
        super(LSEVGG, self).__init__()

        # ── Stem：标准卷积建立浅层特征 ──
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        # ── Stage 1: 64ch, 224→112  ──
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, use_se=True),
            nn.MaxPool2d(2)
        )

        # ── Stage 2: 64→128ch, 112→56  ──
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, use_se=True),
            nn.MaxPool2d(2)
        )

        # ── Stage 3: 128→256ch, 56→28  ──
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, use_se=True),
            ResidualBlock(256, 256, use_se=True),
            nn.MaxPool2d(2)
        )

        # ── Stage 4: 256→512ch, 28→14  ──
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, use_se=True),
            ResidualBlock(512, 512, use_se=True),
            nn.MaxPool2d(2)
        )

        # ── Stage 5: 512ch, 14→7  ──
        self.stage5 = nn.Sequential(
            ResidualBlock(512, 512, use_se=True),
            ResidualBlock(512, 512, use_se=True),
            nn.MaxPool2d(2)
        )

        # ── GAP + 轻量分类头 ──
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def build_model(num_classes=15, pretrained_path=None):
    """构建 LSEVGG 模型，并可选择加载预训练权重。

    兼容 v1/v2/v3 的保存格式；只加载名称、形状均匹配的参数。
    """
    model = LSEVGG(num_classes=num_classes)
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items()
                         if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"已加载预训练权重 {pretrained_path}（匹配 {len(filtered_dict)} 个参数）")
    return model


if __name__ == '__main__':
    model = build_model(num_classes=15)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"输入: {x.shape} → 输出: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
