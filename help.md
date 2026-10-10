# LeNet
nn模块与F模块 

nn 和 F 的本质区别：有没有“状态”

| pxl|`torch.nn（nn）` |`torch.nn.functional（F）`|
|:---|----|----|
| 形式|类，需要实例化：nn.ReLU()|函数，直接调用：F.relu(x)|
| 状态|有状态（可保存参数、缓冲区）|无状态（纯函数，输入→输出）|
| 是否注册到模型|是，model.parameters() 能找到| 否，不会出现在 state_dict|
| 典型例子|nn.Conv2d、nn.Linear、nn.BatchNorm2d|F.relu、F.max_pool2d、F.dropout|

## nn 是“层”，F 是“运算”。
# AlexNet
写成特征提取模块 ***backbone*** 和检测头两部分
# VGG16
采用 `make_layers(cfg, in_channels)` 构建层级结构

直接传入层列表即可，因为**VGG**的块结构很规整


# GoogLeNet
创建 **训练** 数据加载器 时需要`shuffle=True`

**验证** 数据加载器 时需要`shuffle=False`

否则出现验证集 ***acc*** 始终保持 **0.5**,(二分类)
# ResNet

核心是 BasicBlock：两层 3×3 卷积 + 残差连接。

通道或尺寸变化时用 downsample（1×1 卷积 + BN）对齐。

分类头必须用 AdaptiveAvgPool2d((1,1))，不能用 AvgPool2d。

resnet18(num_classes=37) 修改类别数。

# 训练技巧
数据增强（仅训练集）：RandomResizedCrop、RandomHorizontalFlip、ColorJitter。

验证集只做 Resize + Normalize。

优化器：Adam(lr=1e-4, weight_decay=1e-4) 或 SGD(lr=0.01, momentum=0.9)。

学习率调度：CosineAnnealingLR。

过拟合缓解：增强、weight_decay、Dropout、早停。

  
# git
### 先修改 ***.gitignore***

`git status`

`git add .`

`git commit -m "添加什么"`

`git push -u origin main`

推送不了的话，设置成自己端口

`git config --global http.proxy http://127.0.0.1:7890`

`git config --global https.proxy http://127.0.0.1:7890`
