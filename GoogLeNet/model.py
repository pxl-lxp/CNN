import torch
from torchsummary import summary
import torch.nn as nn
"""
基础卷积单元 BasicConv2d
功能：卷积层 + BatchNorm2d + ReLU激活
设计要点：卷积不使用bias，BN自带偏移，减少参数量
入参：in_channels输入通道，out_channels输出通道，**kwargs传入kernel_size/stride/padding
输入张量：[N, in_channels, H, W]
输出张量：[N, out_channels, H_out, W_out]
"""
# 先不做bn
# **kwargs:接受任意数量的关键字参数， 打包成字典
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias = False, **kwargs)
        # 加回 bn 层 , 因为验证集 acc 始终保持 0.5 上不去
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.relu = nn.ReLU(inplace=True) # inplace=True原地操作， 不新建副本

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
"""
Inception模块
功能：4分支并行多尺度特征提取，最后通道维度拼接
分支1：1×1卷积
分支2：1×1降维卷积 → 3×3卷积(padding=1保持尺寸)
分支3：1×1降维卷积 → 5×5卷积(padding=2保持尺寸)
分支4：3×3最大池化(stride=1保尺寸) → 1×1通道投影卷积
参数说明：
    in_channels：输入特征通道
    ch1x1：分支1输出通道
    ch3x3red：分支2的1×1降维通道
    ch3x3：分支2的3×3输出通道
    ch5x5red：分支3的1×1降维通道
    ch5x5：分支3的5×5输出通道
    pool_proj：分支4池化后1×1输出通道
输入：[N, in_channels, H, W]
输出：[N, ch1x1+ch3x3+ch5x5+pool_proj, H, W]
"""

# 特点:每个分支 输入形状 和 输出形状 完全一样
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 分支1
        self.branch1=BasicConv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)
        # 分支2
        self.branch2=nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1),
            BasicConv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, padding=1),
        )
        # 分支3
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1),
            BasicConv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, padding=2),
        )
        # 分支4
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat((branch1, branch2, branch3, branch4), dim=1)
"""
GoogLeNet 网络主体（移除辅助分类器版本）
整体流水线：
1.输入预处理阶段：7×7卷积+池化+1×1+3×3卷积，特征图 224→112→56→28
2.Inception3组：2层Inception，特征图28×28，输出通道480，下采样到14×14
3.Inception4组：5层Inception，特征图14×14，输出通道832，下采样到7×7
4.Inception5组：2层Inception，特征图7×7，输出通道1024
5.分类输出头：全局平均池化→flatten→dropout→全连接输出类别
输入张量：[N, 3, 224, 224]
输出张量：[N, num_classes]
"""

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        # 块1
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)# 224x224x3 -> 112x112x64
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True) # 112x112x64 -> 56x56x64
        # 块2
        self.conv2 = BasicConv2d(64, 64, kernel_size=1) # 56x56x64 -> 56x56x64
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1) # 56x56x64 -> 56x56x192
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)  # 56x56x192 -> 28x28x192
        # Inception1
        """
            路线1: 28x28x192 -> 28x28x64
            路线2: 28x28x192 -> 28x28x96 -> 28x28x128
            路线3: 28x28x192 -> 28x28x16 -> 28x28x32
            路线4: 28x28x192 -> 28x28x192 -> 28x28x32
        """
        self.inception1a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception1b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(3, 2, ceil_mode=True)
        # Inception2
        self.inception2a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception2b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception2c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception2d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception2e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(3, 2, ceil_mode=True)
        # Inception3
        self.inception3a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception3b = Inception(832, 384, 192, 384, 48, 128, 128)
        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',# 从输入通道总数计算标准差
                    nonlinearity='relu'# 非线性激活函数选什么， 用来计算增益 gain
                )
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # 预处理
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        # inception1
        x = self.inception1a(x)
        x = self.inception1b(x)
        x = self.pool3(x)
        # inception2
        x = self.inception2a(x)
        x = self.inception2b(x)
        x = self.inception2c(x)
        x = self.inception2d(x)
        x = self.inception2e(x)
        x = self.pool4(x)
        # inception3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.avgpool(x)
        # 检测头
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet(10).to(device)
    summary(model, (3, 224, 224))



