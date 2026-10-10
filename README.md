# CNN 自学记录

这是我学习卷积神经网络（CNN）过程中的代码与笔记，主要通过动手实现经典网络、训练模型和整理问题来加深理解。项目会随着学习持续更新。

## 学习内容

- **LeNet**：从基础卷积网络入门，练习模型搭建和数据处理。
- **AlexNet**：了解更深层的网络结构，并拆分特征提取部分与分类头。
- **VGG16**：练习使用配置列表构建重复的网络模块。
- **GoogLeNet**：学习 Inception 结构，并实践训练集与验证集的数据加载。
- **ResNet**：学习残差连接、BasicBlock 和下采样。

## 目录说明

每个网络目录主要包含模型定义、训练和测试代码；部分目录还包含数据预处理或预测脚本。学习过程中的补充笔记整理在 [`help.md`](./help.md)。

## 环境配置

项目使用 Conda 环境 `detection`，基于 Python 3.8 和 PyTorch 2.4.1（CUDA 11.8）。安装了兼容 CUDA 11.8 的 NVIDIA 驱动后，可通过以下命令创建并激活环境：

```bash
conda env create -f environment.yml
conda activate detection
```

依赖及版本见 [`environment.yml`](./environment.yml)。

## 学习记录

目前的学习重点包括：

- 理解 PyTorch 中 `nn` 模块与 `torch.nn.functional` 的使用区别。
- 熟悉模型训练、验证、数据增强和学习率调度等基本流程。
- 通过实现经典 CNN，逐步理解网络结构及其设计思路。

## 后续计划

- 持续完善现有模型与实验记录。
- 整理不同网络的结构特点和训练效果。
- 在后续学习中补充新的模型与实践。
