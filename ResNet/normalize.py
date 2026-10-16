import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 只做Resize+ToTensor，不能加Normalize！！
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 只加载训练集计算均值方差
train_dataset = datasets.ImageFolder("./data/train", transform=transform)

# batch尽量开大一点加快计算，不要shuffle
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

mean = torch.zeros(3)
std = torch.zeros(3)
total_pixel = 0

print("数据集mean、std...")
for images, _ in train_loader:
    # images shape: [B,3,H,W]
    b, c, h, w = images.shape
    pixel_num = b * h * w
    total_pixel += pixel_num

    # 按通道求和
    mean += images.sum(dim=[0, 2, 3])
    std += (images ** 2).sum(dim=[0, 2, 3])

mean = mean / total_pixel
std = torch.sqrt(std / total_pixel - mean ** 2)

print(f"\n计算结果：")
print(f"mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
print(f"std  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
