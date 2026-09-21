import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from LeNet.model import LeNet


def get_dataloader(test_set, batch_size, num_workers, pin_memory):

    # 创建加载器对象
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

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
    test_set = FashionMNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())
    test_loader = get_dataloader(test_set, batch_size=1, num_workers=0, pin_memory=False)
    model = LeNet()
    state_dict = torch.load('./weights/LeNet.pth', weights_only=True)
    model.load_state_dict(state_dict)
    model_eval(model, test_loader)


