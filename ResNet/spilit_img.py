import os
import shutil

# ===================== 配置区 =====================
# 源文件夹：你存放那堆图片的文件夹
src_dir = "images"
# 目标文件夹：分类后输出到哪里（会自动创建）
dst_dir = "sorted_data"
# 操作方式：True 为复制（保留原图，更安全），False 为移动（节省磁盘空间）
use_copy = False
# =================================================

# 创建目标根目录
os.makedirs(dst_dir, exist_ok=True)

count_dict = {}
total = 0

# 遍历源文件夹
for fname in os.listdir(src_dir):
    # 过滤出图片文件
    if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    # 解析文件名获取类别
    # 如果文件名是 "Abyssinian_1.jpg" -> rsplit 后得到 ["Abyssinian", "1.jpg"]
    # 如果文件名是 "american_bulldog_100.jpg" -> rsplit 后得到 ["american_bulldog", "100.jpg"]
    parts = fname.rsplit('_', 1)
    if len(parts) < 2:
        print(f"[跳过] 文件名不符合格式: {fname}")
        continue

    breed = parts[0]

    # 创建对应的类别文件夹
    breed_dir = os.path.join(dst_dir, breed)
    os.makedirs(breed_dir, exist_ok=True)

    # 复制/移动图片
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(breed_dir, fname)

    if use_copy:
        shutil.copy(src_path, dst_path)
    else:
        shutil.move(src_path, dst_path)

    # 统计数量
    count_dict[breed] = count_dict.get(breed, 0) + 1
    total += 1

# ===================== 打印结果 =====================
print(f"\n处理完成，共处理 {total} 张图片")
print(f"共生成 {len(count_dict)} 个类别文件夹：")
for breed, count in sorted(count_dict.items()):
    print(f"  {breed}: {count} 张")
print(f"\n输出目录：{os.path.abspath(dst_dir)}")