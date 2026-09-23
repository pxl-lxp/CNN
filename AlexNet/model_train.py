import copy
import time
import pandas as pd
import matplotlib.pyplot as plt

import torch                     # torch库
import torch.nn as nn            # nn
import torch.optim as optim      # 优化器
from torchvision.transforms.v2 import Compose
from tqdm import tqdm            # 封装后显示进度条

import torch.utils.data as data
from  torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import AlexNet #模型


def get_dataloader(dataset, batch_size, num_workers, pin_memory):
    # 划分训练集和验证集
    train_set, val_set = data.random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)])
    # 创建加载器对象
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

def train(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    model = model.to(device)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义最优模型
    best_model_wts = copy.deepcopy(model.state_dict())
    # 初始化参数
    best_acc = 0.0    # 准确率
    train_losses = [] # 训练损失列表
    val_losses = []   # 验证损失列表
    train_accs = []   # 训练准确率列表
    val_accs = []     # 验证准确率列表
    start_time = time.time()   #开始时间

    #训练+验证
    for epoch in range(epochs):
        # 初始化参数
        # 训练样本数 验证样本数   训练损失     验证损失     训练正确数量     验证正确数量
        train_num, val_num, train_loss, val_loss, train_corrects, val_corrects = 0, 0, 0.0, 0.0, 0, 0

        # ================================分批训练========================================
        # 开启训练模式
        model.train()
        # 封装后可以打印进度条
        with tqdm(train_loader,
                  desc=f"Epoch {epoch + 1}/{epochs} [训练]",
                  ncols=120,
                  leave=False) as train_pdar:
            for input, target in train_pdar:
                # 设置到设备
                input, target = input.to(device), target.to(device)
                # 向前传播
                output = model(input)
                # 计算损失
                loss = criterion(output, target)
                # 清零累计梯度
                optimizer.zero_grad()
                # 反向传播
                loss.backward()
                # 优化学习率和梯度
                optimizer.step()
                # 本批损失总和
                train_loss += loss.item() * input.size(0) # 第一维 batch_size, shape = torch.Size([32, 1, 28, 28])
                # 预测类别
                pred = torch.argmax(output, dim=1)
                # 正确数量
                train_corrects += torch.sum(pred == target).item()
                # 训练数量
                train_num += input.size(0)

                # 实时更新进度条上的 loss acc
                train_pdar.set_postfix({
                    "loss": f"{train_loss / train_num:.4f}",
                    "acc": f"{train_corrects / train_num:.4f}"
                })

        # ================================分批验证========================================
        # 开启验证模式
        model.eval()
        with tqdm(val_loader,
                  desc=f"Epoch {epoch + 1}/{epochs} [验证]",
                  ncols=120,
                  leave=False) as val_pdar:
            with torch.no_grad():
                for input, target in val_pdar:
                    # 设置到设备
                    input, target = input.to(device), target.to(device)

                    # 向前传播
                    output = model(input)
                    # 计算损失
                    loss = criterion(output, target)
                    # 验证总损失
                    val_loss += loss.item() * input.size(0)  # 第一维 batch_size, shape = torch.Size([32, 1, 224, 224])
                    # 预测类别
                    pred = torch.argmax(output, dim=1)
                    # 验证正确数量
                    val_corrects += torch.sum(pred == target).item()
                    # 验证数量
                    val_num += input.size(0)
                    # 实时更新进度条上的 loss acc
                    val_pdar.set_postfix({
                        "loss": f"{val_loss / val_num:.4f}",
                        "acc": f"{val_corrects / val_num:.4f}"
                    })

        # ==============================统计本轮损失和精度======================================
        # 加入列表
        train_losses.append(train_loss / float(train_num))
        val_losses.append(val_loss / float(val_num))
        train_accs.append(train_corrects / float(train_num))
        val_accs.append(val_corrects / float(val_num))
        tqdm.write(f"Epoch {epoch + 1}/{epochs} 完成 | 训练loss{train_loss / float(train_num):.4f} | 训练acc{train_corrects / float(train_num):.4f} | 验证loss{val_loss / float(val_num):.4f} | 验证acc{val_corrects / float(val_num):.4f}")

        if val_accs[-1] > best_acc:
            best_acc = val_accs[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
    # 保存模型
    torch.save(best_model_wts, f"./weights/AleNet.pth")
    #打印耗时和acc
    time_used = time.time() - start_time
    print(f"训练和验证耗时{time_used // 60:.0f}m{time_used % 60:.0f}s")
    print(f"最佳准确率: {best_acc * 100:.2f}%")

    # 保存训练数据
    train_process = pd.DataFrame(data={
        "Epoch": range(1, epochs + 1),
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
    })
    train_process.to_csv("./results/train.csv", index=False)
    return train_process

def matplot_process(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_process["Epoch"], train_process.train_losses, 'ro-', label="train loss")
    plt.plot(train_process["Epoch"], train_process.val_losses, 'bs-', label="validation loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.subplot(122)
    plt.plot(train_process["Epoch"], train_process.train_accs, 'ro-', label="train accuracy")
    plt.plot(train_process["Epoch"], train_process.val_accs, 'bs-', label="validation accuracy")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

if __name__ == '__main__':
    # 数据集,套用之前的FashionMNIST
    dataset = FashionMNIST(root='../LeNet/data',
                              train=True,
                              transform=Compose([transforms.Resize(227),transforms.ToTensor()]),
                              download=True
                              )
    # print(len(dataset))
    # 数据加载器
    train_loader, val_loader = get_dataloader(  dataset, 32, num_workers=4, pin_memory=True)

    # for step, (x, y) in enumerate(train_loader):
    #     if step>0:
    #         break
    #     print(x.shape)

    model = AlexNet(10)
    train_process = train(model, train_loader, val_loader, 20)
    matplot_process(train_process)