import torch
import torch.nn as nn
from torchsummary import summary

"""
==========================VGG16=========================
    输入：[batch, 3, 224, 224]
    块1：2层3×3卷积(64) + 最大池化  输出 [64,112,112]
    块2：2层3×3卷积(128) + 最大池化 输出 [128,56,56]
    块3：3层3×3卷积(256) + 最大池化 输出 [256,28,28]
    块4：3层3×3卷积(512) + 最大池化 输出 [512,14,14]
    块5：3层3×3卷积(512) + 最大池化 输出 [512,7,7]

    展平：512×7×7 = 25088
    全连接层：25088→4096→4096→类别数，中间ReLU+Dropout
========================================================
             仿照标准写法，使用make_layers
             
"""

def make_layers(cfg, in_channels=3):

    # 层级列表，往里面加层
    layers = []
    in_chan = in_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(2, 2)]
        else:
            conv2d = nn.Conv2d(in_chan, v, kernel_size=3, padding=1)
            layers +=[conv2d, nn.ReLU(inplace=True)]
            in_chan = v

    return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5, in_channels=3):
        super(VGG16, self).__init__()
        cfg = [
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M'
        ]
        self.features = make_layers(cfg, in_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),

        )

        # 不初始化效果极差，模型再较短的轮次内根本无法收敛，所以需要初始化权重，ReLU激活函数一般使用凯明初始化
        for m in self.modules():
            if isinstance(m ,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std= 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16(in_channels=1).to(device)
    summary(model, input_size=(1, 224, 224))




