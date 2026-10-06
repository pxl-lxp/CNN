from PIL import Image
import torch
from torchvision import transforms
from model import GoogLeNet


# ============配置==============
MODEL_PATH = "./weights/GoogLeNet.pth"
IMG_SIZE = 224
CLASSES = ["cat","dog",]
MEAN=[0.4860, 0.4517, 0.4161]
STD=[0.2583, 0.2510, 0.2540]

# todo:换图片预测需要修改的地方

image_path = "./5.jpg"
image_class = "dog"
# ==============================

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    # 实例化模型
    model = GoogLeNet(num_classes=len(CLASSES))
    # 加载模型权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    # 把模型放到设备中
    model.to(device)
    # 验证模式
    model.eval()
    # 设置变换
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # 变换图片
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    # 预测
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output).item()
        print(f"预测的类别是{CLASSES[prediction]}")
        if CLASSES[prediction] == image_class:
            print("预测正确")

if __name__ == "__main__":
    main()