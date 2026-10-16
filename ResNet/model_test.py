import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from model import resnet18


def get_dataloader(batch_size, num_workers, pin_memory):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4860, 0.4517, 0.4161], std=[0.2583, 0.2510, 0.2540])
    ])
    test_loader = DataLoader(datasets.ImageFolder("./data/test", transform=transform),)

    return test_loader

def model_eval(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型放到设备
    model.to(device)

    test_acc, test_corrects, test_num = 0.0, 0 , 0
    model.eval()

    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to(device), label.to(device)
            output = model(input)
            pred = torch.argmax(output, dim = 1)
            test_corrects += torch.sum(pred == label).item()
            test_num +=input.size(0)


    test_acc = test_corrects/test_num
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':

    test_loader = get_dataloader(batch_size=16, num_workers=0, pin_memory=False)
    # 数据集位猫狗二分类
    model = resnet18(37)

    state_dict = torch.load('./weights/ResNet18_37_class.pth', weights_only=True)
    model.load_state_dict(state_dict)
    model_eval(model, test_loader)
# Test Accuracy: 0.7150

# Test Accuracy: 0.5655 37classes
