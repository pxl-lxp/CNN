import torch
import torch.nn as nn
from torchsummary import summary
"""
ResNet18 网络层级结构说明
输入：(N, 3, 224, 224)
1. 初始卷积块 conv1
    Conv2d(3→64, 7×7, stride=2, padding=3)
    BN
    ReLU
    MaxPool(3×3, stride=2, padding=1)
    输出：(N, 64, 56, 56)
2. layer1（2个BasicBlock，通道64，尺寸56×56，stride=1）
    BasicBlock1: conv(3×3,64→64) + BN + ReLU + conv(3×3,64→64) + BN + 残差连接
    BasicBlock2: conv(3×3,64→64) + BN + ReLU + conv(3×3,64→64) + BN + 残差连接
    输出：(N, 64, 56, 56)
3. layer2（2个BasicBlock，通道128，尺寸28×28，stride=2，第一个block带downsample）
    BasicBlock1: stride=2，下采样；1×1卷积downsample对齐通道和尺寸
    BasicBlock2: 正常残差块
    输出：(N, 128, 28, 28)
4. layer3（2个BasicBlock，通道256，尺寸14×14，stride=2，第一个block带downsample）
    BasicBlock1: stride=2，下采样
    BasicBlock2: 正常残差块
    输出：(N, 256, 14, 14)
5. layer4（2个BasicBlock，通道512，尺寸7×7，stride=2，第一个block带downsample）
    BasicBlock1: stride=2，下采样
    BasicBlock2: 正常残差块
    输出：(N, 512, 7, 7)
6. 分类头
    AdaptiveAvgPool2d → (N,512,1,1)
    flatten → (N,512)
    Linear(512 → num_classes)
    输出：(N, num_classes)
BasicBlock说明：
expansion=1，两层3×3卷积，残差shortcut；通道/尺寸不匹配时用downsample（1×1conv+BN）
"""

# ===============================基础残差块======================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 是否对原图加 1x1 卷积 , 改变通道数
        self.downsample = downsample

    def forward(self, x):
        # 记录原数据
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 是否需要 1x1 卷积
        if self.downsample is not None:
            identity = self.downsample(x)
        # 残差连接
        out += identity
        out = self.relu(out)
        return out
# ==============================ResNet 主体=====================================
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # 初始层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def _make_layer(self, block, out_channels, block_nums, stride=1):
        # 是否下采样 , 改变通道数
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        # 添加层
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, block_nums):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
# =============================构建 resnet18====================================
def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18().to(device)
    summary(model, input_size=(3, 224, 224))





