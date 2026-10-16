import torch
import torch.nn as nn

# 定义基础模块

def autopad(k, p=None):
    """todo:自动求padding, 保证尺寸不变"""
    if p is None:
        p = k //2
    return p

class Conv(nn.Module):
    """todo:Conv2d + BN + ReLU"""
    # c1:输入通道数 c2:输出通道数 k:卷积核尺寸 s:步幅 p:填充 g:groups
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MaxPool(nn.Module):
    """todo:MaxPool"""
    def __init__(self, k=2, s=2):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=k, stride=s)
    def forward(self, x):
        return self.pool(x)

class AdaptiveAvgPool(nn.Module):
    """todo:Adaptive avg pool"""
    def __init__(self, h, w):
        super(AdaptiveAvgPool, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((h, w))
    def forward(self, x):
        return self.avgpool(x)

class Flatten(nn.Module):
    """todo:Flatten"""
    def forward(self, x):
        return torch.flatten(x, 1)

class Linear(nn.Module):
    """todo:Linear"""
    def __init__(self, c1, c2):
        super(Linear, self).__init__()
        self.linear = nn.Linear(c1, c2)
    def forward(self, x):
        return self.linear(x)

class ReLU(nn.Module):
    """todo:ReLU"""
    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x)

class Dropout(nn.Module):
    """todo:Dropout"""
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.dropout = nn.Dropout(p)
    def forward(self, x):
        return self.dropout(x)


