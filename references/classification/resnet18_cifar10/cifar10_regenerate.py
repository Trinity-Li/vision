import os
import pickle
import numpy as np
from PIL import Image
import tarfile
import shutil

# ================= 配置 =================
# 你的压缩包路径
TAR_PATH = '/root/autodl-tmp/data/cifar-10-python.tar.gz'

# 你的 .pt 文件里记录的目标根目录名称 (根据报错信息推断)
# 报错路径: .../cifar10_split_data/eval_set_B/truck/36717.png
# 我们这里先生成到一个标准目录，之后再通过代码映射路径
OUTPUT_ROOT = '/root/autodl-tmp/eval/references/classification/cifar10_images'

# CIFAR-10 类别名
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# ========================================

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def main():
    print(f"🚀 开始从 {TAR_PATH} 提取图片...")

    # 1. 解压 tar.gz
    extract_tmp = './temp_cifar_extract'
    if not os.path.exists(extract_tmp):
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            tar.extractall(path=extract_tmp)

    data_dir = os.path.join(extract_tmp, 'cifar-10-batches-py')

    # 准备输出目录
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    # 2. 遍历所有 batch 文件 (包含训练集和测试集)
    # CIFAR-10 训练集有 5 个 batch，测试集 1 个
    batches = [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch']

    global_idx = 0  # 全局索引计数器

    # 注意：CIFAR-10 的图片索引通常是连续的。
    # 我们需要将所有图片按顺序保存，以便 .pt 文件里的路径能对上。
    # 但 .pt 文件里的路径结构是 `eval_set_B/truck/36717.png`
    # 这暗示了图片是按类别分文件夹存放的。

    # 创建类别文件夹
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_ROOT, cls), exist_ok=True)

    print("📸 正在转换二进制数据为 PNG 图片...")

    # 这里的关键是：必须按照 .pt 文件预期的顺序生成 ID
    # 通常 CIFAR-10 原始顺序是：data_batch_1 -> 5, 然后 test_batch

    # 为了保险起见，我们把所有图片都解压出来。
    # 但是 .pt 文件里的 36717 这个 ID 说明它是从原始训练集(50000张)里切出来的。

    # 处理训练集
    train_batches = [f'data_batch_{i}' for i in range(1, 6)]
    idx_counter = 0

    for batch_name in train_batches:
        batch_path = os.path.join(data_dir, batch_name)
        d = unpickle(batch_path)
        labels = d[b'labels']
        data = d[b'data']
        filenames = d[b'filenames']  # 虽然有文件名，但我们通常用索引命名

        for i in range(len(labels)):
            label = labels[i]
            img_flat = data[i]

            # Reshape: 3072 -> 3, 32, 32 -> 32, 32, 3 (RGB)
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))
            img = np.dstack((img_R, img_G, img_B))

            # 保存图片
            # 命名格式：直接用索引，例如 36717.png
            save_name = f"{idx_counter}.png"
            class_name = CLASSES[label]

            save_path = os.path.join(OUTPUT_ROOT, class_name, save_name)
            Image.fromarray(img).save(save_path)

            idx_counter += 1

    print(f"✅ 已处理完 50,000 张训练集图片 (Index 0 - 49999)")
    print(f"📂 图片保存在: {OUTPUT_ROOT}")

    # 清理临时文件
    shutil.rmtree(extract_tmp)


if __name__ == "__main__":
    main()