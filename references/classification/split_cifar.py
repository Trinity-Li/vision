import os
import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image
from tqdm import tqdm
import numpy as np

# ================= 配置区域 =================
# 原始压缩包所在的目录 (注意：torchvision会自动查找该目录下的压缩包)
# AutoDL通常在 /root/autodl-pub/cifar-10/
SOURCE_ROOT = '/root/autodl-tmp/cifar10_source'

# 输出路径
OUTPUT_ROOT = './cifar10_split_data'

# 选择数据集 'cifar10' 或 'cifar100'
DATASET_TYPE = 'cifar10'


# ===========================================

def save_images(dataset, indices, save_dir):
    """
    读取dataset中的特定indices，保存为 ImageFolder 格式
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"正在处理并保存数据到: {save_dir} ...")

    # 遍历索引
    for idx in tqdm(indices):
        img, target = dataset[idx]  # 获取图片对象和标签索引

        # 获取类别名称 (例如 'airplane', 'bird' 等)
        # CIFAR10/100 dataset.classes 存放了类别名列表
        class_name = dataset.classes[target]

        # 创建类别文件夹
        class_dir = os.path.join(save_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # 保存图片 (文件名带上原始索引以防重名)
        save_path = os.path.join(class_dir, f"{idx}.png")
        img.save(save_path)


def main():
    print(f"准备处理 {DATASET_TYPE} 数据集...")

    # ================= 第一部分：处理训练集 (切分 A/B) =================
    # 1. 加载训练集 (train=True)
    if DATASET_TYPE == 'cifar10':
        train_dataset = CIFAR10(root=SOURCE_ROOT, train=True, download=False)
    elif DATASET_TYPE == 'cifar100':
        train_dataset = CIFAR100(root=SOURCE_ROOT, train=True, download=False)

    total_train_size = len(train_dataset)
    print(f"原始训练集大小: {total_train_size}")

    # 2. 生成随机索引并切分
    # np.random.seed(42) # 如果需要固定切分结果，取消注释
    indices = np.random.permutation(total_train_size)
    split_point = total_train_size // 2
    indices_A = indices[:split_point]
    indices_B = indices[split_point:]

    # 3. 保存
    print(f"切分方案: Set A ({len(indices_A)}) | Set B ({len(indices_B)})")

    path_A = os.path.join(OUTPUT_ROOT, 'train_set_A')
    save_images(train_dataset, indices_A, path_A)

    path_B = os.path.join(OUTPUT_ROOT, 'eval_set_B')
    save_images(train_dataset, indices_B, path_B)

    # ================= 第二部分：处理官方测试集 (全部解压) =================
    print("\n正在处理官方 Test 集...")

    # 4. 加载测试集 (关键点：train=False)
    if DATASET_TYPE == 'cifar10':
        test_dataset = CIFAR10(root=SOURCE_ROOT, train=False, download=False)
    elif DATASET_TYPE == 'cifar100':
        test_dataset = CIFAR100(root=SOURCE_ROOT, train=False, download=False)

    total_test_size = len(test_dataset)
    print(f"官方测试集大小: {total_test_size}")

    # 5. 对于测试集，我们需要所有的图片，不需要随机切分
    # 使用 np.arange 生成从 0 到 9999 的所有索引
    test_indices = np.arange(total_test_size)

    # 6. 保存到单独的文件夹
    path_test = os.path.join(OUTPUT_ROOT, 'official_test')
    save_images(test_dataset, test_indices, path_test)

    print("\n全部完成！")
    print(f"Set A 路径: {os.path.abspath(path_A)}")
    print(f"Set B 路径: {os.path.abspath(path_B)}")
    print(f"Test Set 路径: {os.path.abspath(path_test)}")


if __name__ == '__main__':
    main()