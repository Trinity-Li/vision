import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保你的当前目录下有 resnet.py 文件
from resnet import resnet18_modified

# ================= 配置区域 =================
# 1. 模型权重路径 (绝对路径)
CHECKPOINT_PATH = '/root/autodl-tmp/vision/references/classification/result_final_test/checkpoint.pth'

# 2. 验证集路径 (你的 Set B)
VAL_DATA_PATH = '/root/autodl-tmp/vision/references/classification/cifar10_split_data/eval_set_B'

# 3. 结果保存文件名
OUTPUT_FILE = 'eval_set_B_features.pt'

# 4. 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===========================================

def main():
    print(f"当前使用设备: {device}")

    # --- 检查路径是否存在 ---
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"找不到模型权重文件: {CHECKPOINT_PATH}")
    if not os.path.exists(VAL_DATA_PATH):
        raise FileNotFoundError(f"找不到验证集文件夹: {VAL_DATA_PATH}")

    # --- 1. 准备数据预处理 ---
    # 必须与训练时保持一致 (Resize -> CenterCrop -> ToTensor -> Normalize)
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        # 使用 CIFAR-10 的标准均值和方差，或者 ImageNet 的 (通常 ResNet 预训练用 ImageNet)
        # 这里假设你训练时使用的是默认参数
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"正在加载数据集: {VAL_DATA_PATH}")
    # ImageFolder 会自动读取 eval_set_B 下的各个类别子文件夹
    val_dataset = datasets.ImageFolder(root=VAL_DATA_PATH, transform=transform)

    # shuffle=False 保证读取顺序固定，以便我们记录路径
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    print(f"共发现图片: {len(val_dataset)} 张")

    # --- 2. 加载模型 ---
    print("正在加载模型结构...")
    # 注意：这里 num_classes 必须与你训练时的设置一致 (CIFAR-10 是 10)
    model = resnet18_modified(num_classes=10)

    print(f"正在加载权重: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu',weights_only=False)

    # 兼容处理：检查权重字典的结构
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 加载权重
    model.load_state_dict(state_dict)

    # --- 3. 修改模型用于特征提取 ---
    # 将全连接层替换为直通层 (Identity)，直接输出 AvgPool 后的 512维向量
    model.fc = torch.nn.Identity()

    model.to(device)
    model.eval()  # 开启评估模式

    # --- 4. 提取特征 ---
    results = []

    # 获取所有图片的原始路径
    # dataset.samples 是一个列表，每项是 (path, class_index)
    all_samples = val_dataset.samples

    print("开始提取特征...")

    current_idx = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)

            # 前向传播 (输出 512维 特征)
            features = model(images)

            # 转移回 CPU
            features = features.cpu()

            # 存入结果列表
            batch_size = images.size(0)
            for i in range(batch_size):
                # 获取对应的图片路径和标签
                img_path, label_idx = all_samples[current_idx]

                # 获取对应的类别名称 (例如 'airplane')
                class_name = val_dataset.classes[label_idx]

                results.append({
                    "id": img_path,  # 图片绝对路径
                    "feature": features[i],  # 512维 Tensor
                    "label": label_idx,  # 类别索引 (0-9)
                    "class_name": class_name  # 类别名称 (可选)
                })

                current_idx += 1

    # --- 5. 保存结果 ---
    print(f"正在保存结果到 {OUTPUT_FILE} ...")
    torch.save(results, OUTPUT_FILE)
    print("全部完成！")

    # --- 6. 简单验证 ---
    print("\n--- 结果示例 ---")
    if len(results) > 0:
        sample = results[0]
        print(f"图片路径: {sample['id']}")
        print(f"特征维度: {sample['feature'].shape}")
        print(f"类别索引: {sample['label']} ({sample['class_name']})")


if __name__ == '__main__':
    main()