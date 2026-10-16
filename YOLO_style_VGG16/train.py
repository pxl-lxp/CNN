import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from models import VGG16
import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="VGG16 训练脚本")
    # 模型配置
    parser.add_argument("--cfg", type=str, default="VGG16.yaml", help="模型配置文件路径")
    parser.add_argument("--num-classes", type=int, default=37, help="类别数， 会覆盖yaml中的nc")
    # 数据配置
    parser.add_argument("--data", type=str, default="../ResNet/data", help="dataset")
    parser.add_argument("--img-size", type=int, default=224, help="input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader num_workers")
    # 训练配置
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减系数")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="优化器")
    # 保存配置
    parser.add_argument("--save-dir", type=str, default="weights", help="权重保存路径")
    parser.add_argument("--save-name", type=str, default="VGG16.pth", help="权重文件名")
    parser.add_argument("--result-dir", type=str, default="results", help="结果保存路径")

    return parser.parse_args()

def get_dataloader(args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4830, 0.4509, 0.3972],
                             std=[0.2649, 0.2601, 0.2686], )
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4830, 0.4509, 0.3972],
                             std=[0.2649, 0.2601, 0.2686], )
    ])

    train_set = datasets.ImageFolder(os.path.join(args.data, "train"),
                                     transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(args.data, "val"),
                                   transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True)
    return train_loader, val_loader, train_set.classes

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="[训练]", ncols=120, leave=False)
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        # 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += imgs.size(0)

        pbar.set_postfix({"loss": f"{total_loss / total:.4f}",
                          "acc": f"{correct / total:.4f}"})
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

def plot_curve(train_process, save_path):
    """绘制 loss 和 acc 曲线并保存"""
    plt.figure(figsize=(12, 4))

    # 左图：loss
    plt.subplot(121)
    plt.plot(train_process["Epoch"], train_process["train_loss"], 'ro-', label="train loss")
    plt.plot(train_process["Epoch"], train_process["val_loss"], 'bs-', label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    # 右图：accuracy
    plt.subplot(122)
    plt.plot(train_process["Epoch"], train_process["train_acc"], 'ro-', label="train acc")
    plt.plot(train_process["Epoch"], train_process["val_acc"], 'bs-', label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"曲线已保存：{save_path}")

def main():
    args = parse_args()
    print("配置参数:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # 修改 yaml 中的 nc
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["nc"] = args.num_classes
    tmp_cfg = "_tmp_vgg16.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("+"*30)
    print("device:", device)

    # 数据
    train_loader, val_loader, classes = get_dataloader(args)
    print(f"类别数：{len(classes)}，训练批次数：{len(train_loader)}")

    # 模型
    model = VGG16(tmp_cfg).to(device)

    # 优化器
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    # 添加列表保存数值, 用于绘图
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        # 保存数据到列表
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        scheduler.step()

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name))

    os.remove(tmp_cfg)
    print(f"训练完成，最佳验证 acc: {best_acc * 100:.2f}%")

    # 保存训练数据到 CSV
    train_process = pd.DataFrame({
        "Epoch": range(1, args.epochs + 1),
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
    })
    csv_path = os.path.join(args.result_dir, "train.csv")
    train_process.to_csv(csv_path, index=False)
    print(f"训练数据已保存：{csv_path}")
    # 绘图并保存
    png_path = os.path.join(args.result_dir, "train_curve.png")
    plot_curve(train_process, png_path)

if __name__ == "__main__":
    main()
