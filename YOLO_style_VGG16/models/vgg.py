import yaml
import torch
import torch.nn as nn
from .common import (Conv, MaxPool, AdaptiveAvgPool, Flatten, Linear, ReLU, Dropout)

# 模块名 -> 类
MODULE_MAP = {
    "Conv":Conv,
    "MaxPool":MaxPool,
    "AdaptiveAvgPool":AdaptiveAvgPool,
    "Flatten":Flatten,
    "Linear":Linear,
    "ReLU":ReLU,
    "Dropout":Dropout,
}

def parse_model(cfg, ch=3):
    """依据 yaml 配置模型"""
    # 类别数
    nc = cfg['nc']
    # 宽度倍率
    width = cfg.get("width_multiple", 1.0)

    layers = []
    in_ch = ch
    # 连接 backbone 和 head
    all_layers = cfg["backbone"] + cfg["head"]
    for i, (f, n, m, args) in enumerate(all_layers):
        module = MODULE_MAP[m]
        args = list(args)

        # 卷积层 : 通道数按宽度倍率缩放
        if m == "Conv":
            args[0] = max(1, int(args[0] * width))
            args = [in_ch] + args
            in_ch = args[1]

        elif m == "Linear":
            if args[0] == "nc":
                args[0] = nc
            args = [in_ch] + args
            in_ch = args[1]

        layer = module(*args)
        layers.append(layer)

    return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, cfg_path="../VGG16.yaml"):
        super(VGG16, self).__init__()
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.model = parse_model(cfg)

    def forward(self, x):
        return self.model(x)

    def info(self):
        """todo:打印每层信息"""
        print("+" * 60)
        for i, layer in enumerate(self.model):
            print(f"{i:3d}  {layer.__class__.__name__:3s}     {layer}")
            print("+"*60)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VGG16().to(device)
    model.info()