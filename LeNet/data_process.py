from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_data():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([
                                  transforms.Resize(size=224),
                                  transforms.ToTensor()]),
                              download=True
                              )
    # transform=transforms.Compose([transforms.Resize(size=244),  transforms.ToTensor()])
    # 依次执行Compose容器中的操作
    return train_data

if __name__ == '__main__':
    train_data = get_data()
    train_loader = DataLoader(train_data,
                              batch_size=64,
                              shuffle=True,
                              num_workers=0)
    # print(train_loader.dataset)

    for step,(train_x,train_y) in enumerate(train_loader):
        if step > 0:
            break
        # print(step, train_x, train_y)
        # 转维度和类型
        batch_x = train_x.squeeze(1).numpy()
        batch_y = train_y.numpy()
        # 类别字典
        class_names = train_data.classes

        print(step, batch_x.shape, batch_y.shape)
        # 绘图
        plt.figure(figsize=(12,5))
        for i in range(batch_x.shape[0]):
            plt.subplot(4, 16, i + 1)
            plt.imshow(batch_x[i, :, :], cmap='gray')
            plt.title(class_names[int(batch_y[i])], size=10)


        plt.show()