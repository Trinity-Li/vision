import torch
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

# ================= 配置区域 =================
# 指向你的其中一个 .pt 文件 (例如 Ratio 0.0 或 0.2 的文件)
PT_FILE_PATH = 'eval_set_B_features.pt'

# CIFAR-10 官方类别名称 (顺序必须固定，不能乱)
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# ===========================================

class FilelistDataset(Dataset):
    """(复制你原本的 Dataset 类以确保加载逻辑一致)"""

    def __init__(self, pt_path):
        self.samples = []
        print(f"📂 读取文件: {pt_path}")
        try:
            data = torch.load(pt_path, map_location='cpu')
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return

        # 兼容处理
        items = []
        if isinstance(data, dict) and 'ids' in data:
            ids = data['ids']
            lbls = data['labels']
            for i in range(len(ids)):
                items.append({'id': ids[i], 'label': int(lbls[i])})
        elif isinstance(data, list):
            items = data

        for item in items:
            raw_id = item.get('id')
            label = int(item.get('label'))
            if not raw_id: continue

            # 路径清洗
            real_path = raw_id
            if '_copy_' in real_path:
                real_path = real_path.split('_copy_')[0]

            self.samples.append((real_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def check_alignment():
    # 1. 加载数据集
    if not os.path.exists(PT_FILE_PATH):
        print(f"❌ 找不到文件: {PT_FILE_PATH}")
        return

    dataset = FilelistDataset(PT_FILE_PATH)
    total_len = len(dataset)
    print(f"✅ 数据集加载成功，共 {total_len} 张图片")

    # 2. 随机抽取 16 张图片进行验证
    indices = random.sample(range(total_len), 16)

    # 设置绘图
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f"Label Verification: {PT_FILE_PATH}", fontsize=16)

    mismatch_count = 0

    print("\n🔍 开始详细核对 (Path vs Label):")
    print("-" * 60)
    print(f"{'Index':<8} | {'Label ID':<8} | {'Class Name':<12} | {'File Path Keyword'}")
    print("-" * 60)

    for i, idx in enumerate(indices):
        path, label_id = dataset[idx]
        class_name = CIFAR10_CLASSES[label_id]

        # 尝试加载图片
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 图片损坏: {path}")
            img = Image.new('RGB', (32, 32), color='gray')

        # --- 自动逻辑检查 ---
        # 很多数据集的文件路径里会包含类别名，例如 ".../airplane/001.png"
        # 我们检查路径里是否包含当前的 class_name
        is_suspicious = False
        if class_name not in path and str(label_id) not in path:
            # 注意：有些数据集路径可能是 .../class_0/... 或 .../airplane/...
            # 如果路径完全没有包含类别信息，这个检查可能不适用，但通常 CIFAR 解压后会按文件夹分类
            pass

        print(f"{idx:<8} | {label_id:<8} | {class_name:<12} | ...{path[-30:]}")

        # 绘图
        ax = axes[i // 4, i % 4]
        ax.imshow(img)
        ax.set_title(f"Label: {label_id}\n({class_name})", color='green' if not is_suspicious else 'red')
        ax.axis('off')

    # 保存结果图
    save_path = 'verify_labels.png'
    plt.tight_layout()
    plt.savefig(save_path)
    print("-" * 60)
    print(f"\n📸 验证图片已保存至: {save_path}")
    print("👉 请打开这张图片，用肉眼检查图片内容是否与标题一致！")


if __name__ == "__main__":
    check_alignment()