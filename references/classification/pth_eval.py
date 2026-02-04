import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys

# ================= 配置区域 =================
# 你的 Checkpoint 路径
CHECKPOINT_PATH = '/root/autodl-tmp/eval/references/classification/checkpoint.pth'

# 数据集路径 (你之前确定的可用路径)
DATA_ROOT = '/root/autodl-tmp/data'

# 硬件
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128


# ===========================================

def get_cifar_resnet18():
    """重建你训练时用的魔改版 ResNet-18 结构"""
    model = torchvision.models.resnet18(weights=None)
    # 1. 修改第一层卷积 (7x7 -> 3x3)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 2. 移除池化
    model.maxpool = nn.Identity()
    # 3. 修改全连接层 (10类)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def main():
    print(f"🔍 正在检查 Checkpoint: {CHECKPOINT_PATH}")

    # 1. 检查文件是否存在
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ 错误: 文件不存在！")
        return

    # 2. 准备数据 (仅测试集)
    print("📚 加载 CIFAR-10 测试集...")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    try:
        test_dataset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 3. 初始化模型结构
    model = get_cifar_resnet18().to(DEVICE)

    # 4. 加载权重
    try:
        print("📥 正在加载权重参数...")
        # 针对 PyTorch 2.6+ 的安全加载修复
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

        # 兼容处理：有些 checkpoint 保存的是 {'model': state_dict}，有些直接是 state_dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 加载
        msg = model.load_state_dict(state_dict, strict=True)
        print(f"✅ 权重加载成功! ({msg})")

    except Exception as e:
        print(f"❌ 权重加载极其失败: {e}")
        print("原因可能是：")
        print("1. 文件损坏 (请检查文件大小)")
        print("2. 架构不匹配 (比如你用标准 ResNet 权重加载到了魔改 ResNet 上)")
        return

    # 5. 开始评估
    print("🚀 开始评估准确率...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print("-" * 30)
    print(f"🏆 测试集最终准确率: {acc:.2f}%")
    print("-" * 30)

    # 简单的结果判定
    if acc < 15.0:
        print("⚠️ 警告: 准确率极低 (接近随机猜测 10%)。模型可能损坏或未训练。")
    elif acc > 80.0:
        print("✅ 状态: 优秀。这是一个高质量的 Checkpoint。")
    else:
        print("ℹ️ 状态: 正常。模型已学习，但可能未收敛或性能一般。")


if __name__ == "__main__":
    main()