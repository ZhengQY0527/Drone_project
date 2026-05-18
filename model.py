import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。

    通过全局平均池化提取通道统计信息，再学习每个通道的重要性权重，
    从而增强有效特征、抑制无关特征。
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # squeeze：把空间维度压缩成 1x1 的通道描述符
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # excitation：两层全连接学习通道之间的依赖关系
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b: batch size, c: channels
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        # 将通道权重逐元素作用到原特征图上
        return x * y


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：Depthwise Conv + Pointwise Conv。

    先按通道做空间卷积，再用 1x1 卷积融合通道，参数量和计算量都更小。
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise：每个输入通道单独做 3x3 卷积
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride,
            padding=1, groups=in_ch, bias=False
        )
        self.bn_depth = nn.BatchNorm2d(in_ch)
        # Pointwise：用 1x1 卷积完成通道混合并调整通道数
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_point = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_depth(self.depthwise(x)))
        x = self.relu(self.bn_point(self.pointwise(x)))
        return x


class LSEVGG(nn.Module):
    """LSEVGG：轻量级 SE-VGG 风格网络。

    结构上保留了 VGG 的分阶段堆叠方式，同时用深度可分离卷积降低开销，
    再通过 SE 模块增强通道表达能力。num_classes 可按数据集配置。
    """
    def __init__(self, num_classes=15):
        super(LSEVGG, self).__init__()

        # Block 1：先用标准卷积建立浅层特征，再进入轻量卷积和注意力模块
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(2, 2)
        )

        # Block 2：逐步提升通道数，扩大表征能力
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.MaxPool2d(2, 2)
        )

        # Block 3：继续加深网络，让模型捕获更复杂的局部结构
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256),
            nn.MaxPool2d(2, 2)
        )

        # Block 4：中高层语义特征提取
        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.MaxPool2d(2, 2)
        )

        # Block 5：进一步压缩空间尺寸，提升语义抽象程度
        self.block5 = nn.Sequential(
            DepthwiseSeparableConv(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.MaxPool2d(2, 2)
        )

        # 自适应池化到固定特征尺寸，再接全连接分类头
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        # 初始化所有层参数，保证训练初期更稳定
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
        # 按顺序经过五个特征提取阶段
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # 压缩空间维度，保留通道语义信息
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # 输出每个类别的 logits，供交叉熵损失使用
        x = self.classifier(x)
        return x


def build_model(num_classes=15, pretrained_path=None):
    """构建 LSEVGG 模型，并可选择加载预训练权重。

    若分类类别数不同，只加载形状匹配的参数，避免分类头维度不一致报错。
    """
    model = LSEVGG(num_classes=num_classes)
    if pretrained_path:
        # 兼容训练保存的 checkpoint 或纯 state_dict 文件
        state_dict = torch.load(pretrained_path, map_location='cpu')
        if 'model_state_dict' in state_dict:   # 兼容训练保存的检查点
            state_dict = state_dict['model_state_dict']
        # 只保留名称和形状都匹配的参数，其余参数按当前模型初始化值保留
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items()
                         if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"已加载预训练权重 {pretrained_path}（匹配 {len(filtered_dict)} 个参数）")
    return model


if __name__ == '__main__':
    # 直接运行本文件时，快速检查模型输出形状是否正确
    model = build_model(num_classes=15)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"输入: {x.shape} → 输出: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")