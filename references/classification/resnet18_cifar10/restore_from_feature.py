import os
import torch
import pickle
import numpy as np
from PIL import Image
import re

# ================= 配置区域 (请确认路径) =================

# 1. 你刚刚解压好的数据目录 (里面应该有 data_batch_1 等文件)
# 注意：请确保路径最后是 'cifar-10-batches-py'
DATA_DIR = '/root/autodl-tmp/data/cifar-10-batches-py'

# 2. Set B 的特征文件路径 (用来区分 A 和 B)
PT_FILE_PATH = '/root/autodl-tmp/eval/references/classification/eval_set_B_features.pt'

# 3. 输出根目录 (生成的图片将放在这里)
# 这一步会生成 /root/autodl-tmp/eval/references/classification/cifar10_split_data/eval_set_B/...
OUTPUT_ROOT = '/root/autodl-tmp/eval/references/classification/cifar10_split_data'

# CIFAR-10 类别名
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# =======================================================

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_set_b_indices(pt_path):
    print(f"🔍 正在读取特征索引: {pt_path}")
    try:
        data = torch.load(pt_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 读取 .pt 文件失败: {e}")
        return set()

    # 兼容 list 或 dict 格式
    items = []
    if isinstance(data, dict) and 'ids' in data:
        items = data['ids']
    elif isinstance(data, list):
        items = [x['id'] for x in data]  # 假设 list 里是 dict

    indices_b = set()
    for item in items:
        # 处理可能的 dict 结构
        if isinstance(item, dict):
            path_str = item.get('id', '')
        elif isinstance(item, str):
            path_str = item
        else:
            continue

        # 从路径字符串中提取数字 ID
        # 例如 ".../truck/36717.png" -> 36717
        filename = os.path.basename(path_str)
        if '_copy_' in filename:
            filename = filename.split('_copy_')[0]

        match = re.search(r'(\d+)', filename)
        if match:
            indices_b.add(int(match.group(1)))

    print(f"✅ 成功解析 Set B 索引: 共 {len(indices_b)} 个唯一 ID")
    return indices_b


def main():
    # 0. 检查输入目录是否存在
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误: 找不到数据目录: {DATA_DIR}")
        print("   请检查路径是否正确？")
        return

    # 1. 获取 B 集合的 ID
    indices_b = get_set_b_indices(PT_FILE_PATH)
    if not indices_b:
        print("❌ 警告: 未找到 B 集合索引，所有图片将被放入 Set A (或程序出错)")
        # 这里不强制退出，方便调试，但你要注意输出结果

    # 2. 创建输出目录
    dir_a = os.path.join(OUTPUT_ROOT, 'eval_set_A')
    dir_b = os.path.join(OUTPUT_ROOT, 'eval_set_B')

    for d in [dir_a, dir_b]:
        for cls in CLASSES:
            os.makedirs(os.path.join(d, cls), exist_ok=True)

    print(f"🚀 开始转换图片...")
    print(f"   源目录: {DATA_DIR}")
    print(f"   目标目录: {OUTPUT_ROOT}")

    # 3. 遍历 5 个 batch 文件
    batch_files = [f'data_batch_{i}' for i in range(1, 6)]

    global_idx = 0
    count_a = 0
    count_b = 0

    for batch_name in batch_files:
        batch_path = os.path.join(DATA_DIR, batch_name)
        if not os.path.exists(batch_path):
            print(f"⚠️ 跳过缺失文件: {batch_path}")
            continue

        d = unpickle(batch_path)
        labels = d[b'labels']
        data = d[b'data']

        # 处理当前 batch 的图片
        for i in range(len(labels)):
            label = labels[i]

            # 还原 RGB 图像
            img_flat = data[i]
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))
            img = np.dstack((img_R, img_G, img_B))

            class_name = CLASSES[label]
            filename = f"{global_idx}.png"

            # === 分类逻辑 ===
            if global_idx in indices_b:
                target_dir = dir_b
                count_b += 1
            else:
                target_dir = dir_a
                count_a += 1

            save_path = os.path.join(target_dir, class_name, filename)
            Image.fromarray(img).save(save_path)

            global_idx += 1

        print(f"   ✅ 已处理 {batch_name} (累计: {global_idx})")

    print("-" * 50)
    print("🎉 任务完成！")
    print(f"📊 统计: Set A: {count_a} 张 | Set B: {count_b} 张 | 总计: {count_a + count_b}")
    print(f"📂 请修改代码中的路径指向: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()