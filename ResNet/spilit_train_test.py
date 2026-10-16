import os
import shutil
import random

# ===================== 配置区，修改这里 =====================
src_root = r".\sorted_data"   # 原始数据集路径，里面每个子文件夹是一个类别
dst_root = r".\data"         # 划分之后输出的文件夹
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

random.seed(13)  # 固定随机种子，每次划分结果一致
# ==========================================================

# 自动获取所有类别（src_root 下的所有子文件夹）
classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
classes.sort()   # 排序，保证每次运行顺序一致
print(f"检测到 {len(classes)} 个类别：")
for c in classes:
    print(f"  {c}")
print()

# 创建输出文件夹
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(dst_root, split, cls), exist_ok=True)

# 遍历每个类别进行划分
for cls in classes:
    src_dir = os.path.join(src_root, cls)
    img_list = [f for f in os.listdir(src_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    random.shuffle(img_list)
    total = len(img_list)

    # 计算切分索引
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_imgs = img_list[:train_end]
    val_imgs = img_list[train_end:val_end]
    test_imgs = img_list[val_end:]

    # 复制文件
    for name in train_imgs:
        shutil.copy(os.path.join(src_dir, name), os.path.join(dst_root, "train", cls, name))
    for name in val_imgs:
        shutil.copy(os.path.join(src_dir, name), os.path.join(dst_root, "val", cls, name))
    for name in test_imgs:
        shutil.copy(os.path.join(src_dir, name), os.path.join(dst_root, "test", cls, name))

    print(f"[{cls}] total:{total} | train:{len(train_imgs)} val:{len(val_imgs)} test:{len(test_imgs)}")

print("\n数据集划分完成！")
print(f"输出目录：{os.path.abspath(dst_root)}")