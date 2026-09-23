import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

"""
     todo:此处仍使用灰度图,不考虑rgb输入层: [1, 227, 227]
    ========== AlexNet 网络结构 ==========
    输入层: [3, 227, 227]

    第1层 卷积层: 11×11卷积, 96输出通道, stride=4, padding=0
    输出: [96, 55, 55]

    第2层 最大池化层: 3×3池化核, stride=2
    输出: [96, 27, 27]

    第3层 卷积层: 5×5卷积, 256输出通道, stride=1, padding=2
    输出: [256, 27, 27]

    第4层 最大池化层: 3×3池化核, stride=2
    输出: [256, 13, 13]

    第5层 卷积层: 3×3卷积, 384输出通道, stride=1, padding=1
    输出: [384, 13, 13]

    第6层 卷积层: 3×3卷积, 384输出通道, stride=1, padding=1
    输出: [384, 13, 13]

    第7层 卷积层: 3×3卷积, 256输出通道, stride=1, padding=1
    输出: [256, 13, 13]

    第8层 最大池化层: 3×3池化核, stride=2
    输出: [256, 6, 6]

    Flatten展平: 256*6*6 = 9216

    第9层 全连接层: 4096维, ReLU + Dropout
    第10层 全连接层: 4096维, ReLU + Dropout
    第11层 全连接层: 1000维, Softmax(分类输出)
"""

class AlexNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(AlexNet, self).__init__()
        # 特征提取backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(3,2),
        )
        # 展平
        self.flatten = nn.Flatten()

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
    # 定义向前传播
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet(10)
    model.to(device)
    summary(model, (1,227, 227))

