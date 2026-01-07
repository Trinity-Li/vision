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

    # 1. 加载数据集 (download=False 表示使用本地数据)
    # 注意：root参数应该指向包含 .tar.gz 的文件夹
    if DATASET_TYPE == 'cifar10':
        dataset = CIFAR10(root=SOURCE_ROOT, train=True, download=False)
    elif DATASET_TYPE == 'cifar100':
        dataset = CIFAR100(root=SOURCE_ROOT, train=True, download=False)
    else:
        raise ValueError("不支持的数据集类型")

    total_size = len(dataset)
    print(f"原始训练集大小: {total_size}")

    # 2. 生成随机索引并切分
    # 设置随机种子保证可复现 (可选)
    # np.random.seed(42)
    indices = np.random.permutation(total_size)

    split_point = total_size // 2
    indices_A = indices[:split_point]  # 前一半
    indices_B = indices[split_point:]  # 后一半

    print(f"切分方案: Set A ({len(indices_A)} 张) | Set B ({len(indices_B)} 张)")

    # 3. 保存 Set A (用于训练 ResNet18)
    path_A = os.path.join(OUTPUT_ROOT, 'train_set_A')
    save_images(dataset, indices_A, path_A)

    # 4. 保存 Set B (用于评估)
    path_B = os.path.join(OUTPUT_ROOT, 'eval_set_B')
    save_images(dataset, indices_B, path_B)

    print("\n完成！数据已按 ImageFolder 结构保存。")
    print(f"Set A 路径: {os.path.abspath(path_A)}")
    print(f"Set B 路径: {os.path.abspath(path_B)}")


if __name__ == '__main__':
    main()