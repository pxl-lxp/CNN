import torch
import torch.nn as nn
from torchsummary import summary
"""
    输入：1×28×28 单通道灰度图像
    Conv1：5×5卷积，输出6通道，步长1，填充2，输出特征图 6×28×28
    Sigmoid 激活函数
    AvgPool1：2×2平均池化，步长2，输出 6×14×14
    Conv2：5×5卷积，输出16通道，步长1，填充0，输出特征图 16×10×10
    Sigmoid 激活函数
    AvgPool2：2×2平均池化，步长2，输出 16×5×5
    Flatten展平，维度变为 400
    全连接层FC1：400映射到120
    Sigmoid 激活函数
    全连接层FC2：120映射到84
    Sigmoid 激活函数
    全连接层FC3：84映射到10，手写数字10分类输出
"""

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.FC1 = nn.Linear(in_features=400, out_features=120)
        self.FC2 = nn.Linear(in_features=120, out_features=84)
        self.FC3 = nn.Linear(in_features=84, out_features=10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.sigmoid(self.FC1(x))
        x = self.sigmoid(self.FC2(x))
        x = self.FC3(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))

