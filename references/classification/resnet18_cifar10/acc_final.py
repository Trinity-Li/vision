import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
import glob
import re
import csv

# --- 导入同级目录下的官方模块 ---
try:
    import utils
    from train import train_one_epoch, evaluate
except ImportError:
    print("❌ 错误：请将此脚本放在 'references/classification/' 目录下运行！")
    sys.exit(1)

# ==========================================
# 1. 配置区域
# ==========================================
TRAIN_FILES_PATTERN = 'eval_set_B*.pt'  # 训练集索引文件模式
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 你的 Checkpoint 路径 (如果需要从之前的权重开始，请填这里)
# 如果留空，将默认使用 ImageNet 预训练权重进行初始化
PRETRAINED_CHECKPOINT = '/root/autodl-tmp/eval/references/classification/checkpoint.pth'


class Args:
    def __init__(self):
        self.batch_size = 128
        self.epochs = 15  # 针对 CIFAR-10 结构调整后，训练可能需要稍多几轮
        self.lr = 0.001  # SGD 初始学习率
        self.momentum = 0.9
        self.weight_decay = 5e-4  # CIFAR-10 常用 weight_decay
        self.print_freq = 50
        self.workers = 8
        self.output_dir = '.'

        # 官方代码占位参数
        self.clip_grad_norm = None
        self.model_ema = False
        self.lr_warmup_epochs = 0
        self.lr_warmup_method = 'linear'
        self.lr_warmup_decay = 0.01
        self.label_smoothing = 0.0
        self.mixup_alpha = 0.0
        self.cutmix_alpha = 0.0
        self.distributed = False
        self.rank = 0


# ==========================================
# 2. CIFAR-10 专用 Transforms
# ==========================================
def get_cifar_transforms(is_train=True):
    # CIFAR-10 标准均值和方差
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 保持 32x32
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ==========================================
# 3. 自定义数据集 (同前)
# ==========================================
class FilelistDataset(Dataset):
    def __init__(self, pt_path, transform=None):
        self.samples = []
        self.transform = transform

        # ========================================================
        # 1. 定义新数据的根目录 (即你刚才运行 restore 脚本生成的目录)
        # ========================================================
        SPLIT_DATA_ROOT = '/root/autodl-tmp/eval/references/classification/cifar10_split_data'

        # CIFAR-10 类别名 (用于构建路径)
        CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # 2. 自动判断是 Set A 还是 Set B
        # 根据 .pt 文件名来猜。文件名里通常包含 'set_B' 或 'set_A'
        if 'set_B' in os.path.basename(pt_path):
            subset_folder = 'eval_set_B'
        else:
            subset_folder = 'eval_set_A'

        print(f"📂 读取索引: {os.path.basename(pt_path)} -> 指向目录: {subset_folder}")

        # 3. 加载 .pt 文件
        try:
            data = torch.load(pt_path, map_location='cpu')
        except Exception as e:
            print(f"❌ .pt 文件加载失败: {e}")
            return

        # 4. 提取数据列表 (兼容不同格式)
        items = []
        if isinstance(data, dict):
            # 你的文件可能是 {'ids': [...], 'labels': [...]} 或者是 {'features': ...}
            if 'ids' in data:
                ids = data['ids']
                lbls = data['labels']
            elif 'features' in data:  # 兼容你代码里的这种写法
                ids = data['ids']
                lbls = data['labels']
            else:
                print("❌ 字典格式无法识别，缺少 'ids'")
                return

            for i in range(len(ids)):
                items.append({'id': ids[i], 'label': int(lbls[i])})

        elif isinstance(data, list):
            items = data

        # 5. 核心循环：构建新路径
        for item in items:
            raw_id = item.get('id')
            label = int(item.get('label'))
            if not raw_id: continue

            # --- 关键修改开始 ---
            # 原始可能是: "/.../23182_r0.2_copy90.png"
            base_name = os.path.basename(raw_id)

            # 使用正则提取开头的数字 ID
            # 逻辑：找到字符串开头的第一串数字
            match = re.match(r'^(\d+)', base_name)

            if match:
                clean_id = match.group(1)  # 拿到 "23182"
                filename = f"{clean_id}.png"  # 强制变成 "23182.png"
            else:
                # 如果没匹配到数字，回退到原始文件名（防止报错，虽然不太可能）
                filename = base_name

            # --- 关键修改结束 ---

            # --- C. 获取类别名 ---
            class_name = CIFAR_CLASSES[label]  # 拿到 "truck"

            # --- D. 拼装成正确的新路径 ---
            # 新路径 = 根目录 / eval_set_B / truck / 36717.png
            real_path = os.path.join(SPLIT_DATA_ROOT, subset_folder, class_name, filename)

            self.samples.append((real_path, label))

        print(f"   ✅ 成功索引 {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. 获取路径和标签
        path, label = self.samples[idx]

        # 2. 检查文件是否存在 (这一步非常关键)
        if not os.path.exists(path):
            # 详细报错，告诉你到底缺哪个文件
            raise FileNotFoundError(
                f"🔥 图片缺失！\n"
                f"  试图读取: {path}\n"
                f"  原始索引: {self.samples[idx]}\n"
                f"  请检查 /root/autodl-tmp/eval/references/classification/cifar10_split_data 下是否有文件？"
            )

        # 3. 安全打开图片
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"🔥 图片损坏无法打开: {path}\n  错误信息: {e}")

        if self.transform:
            img = self.transform(img)

        return img, label

# ==========================================
# 4. 模型修改函数
# ==========================================
# ==========================================
# 4. 模型修改函数 (修正版)
# ==========================================
def get_cifar_resnet18(pretrained_path=''):
    # 1. 无论如何，先初始化一个标准 ResNet
    # 如果有自定义权重，这里可以用 weights=None 纯净初始化
    # 如果没有，用 'DEFAULT' 加载 ImageNet 权重作为迁移学习的基础
    init_weights = None if pretrained_path else 'DEFAULT'
    model = torchvision.models.resnet18(weights=init_weights)

    # 2. 【关键】先进行架构修改，把坑位调整好
    # 修改第一层卷积：适应 32x32 输入 (7x7 -> 3x3)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 移除第一层池化
    model.maxpool = nn.Identity()

    # 修改全连接层：适应 10 类
    model.fc = nn.Linear(model.fc.in_features, 10)

    # 3. 【关键】坑位对齐了，再加载你的 Checkpoint
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"  -> 加载指定 Checkpoint: {pretrained_path}")
        # 修复之前的 pickle 报错，允许加载非权重数据
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        # 提取 state_dict
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

        # 加载权重
        try:
            model.load_state_dict(state_dict, strict=True)
            print("  ✅ 权重加载成功！")
        except RuntimeError as e:
            # 如果 strict=True 失败，尝试 strict=False 并打印忽略的键
            print(f"  ⚠️ 严格加载失败，尝试非严格加载: {e}")
            model.load_state_dict(state_dict, strict=False)
    else:
        if not pretrained_path:
            print("  -> 使用 ImageNet 预训练权重 (迁移学习模式)")

    return model

# ==========================================
# 5. 主程序
# ==========================================
def main():
    args = Args()
    print(f"🔧 CIFAR-10 全局微调 (Arch Modified) | Device: {DEVICE}")

    # --- 准备测试集 ---
    CIFAR_ROOT ='/root/autodl-tmp/data'
    print(f"📚 加载测试集 (Root: {CIFAR_ROOT})...")
    try:
        test_dataset = torchvision.datasets.CIFAR10(
            root=CIFAR_ROOT, train=False, download=True, transform=get_cifar_transforms(is_train=False)
        )
    except:
        print("  ⚠️ 本地加载失败，尝试自动下载...")
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=get_cifar_transforms(is_train=False)
        )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # --- 搜索训练文件 ---
    train_files = glob.glob(TRAIN_FILES_PATTERN)
    train_files.sort(key=lambda f: float(re.search(r'(\d+\.\d+)', f).group(1)) if re.search(r'(\d+\.\d+)', f) else 0.0)

    results = []

    for train_file in train_files:
        filename = os.path.basename(train_file)
        match = re.search(r'_(\d+\.\d+)\.pt', filename)
        ratio = match.group(1) if match else "Original"

        print("\n" + "=" * 60)
        print(f"🚀 [任务] 文件: {filename} | Ratio: {ratio}")
        print("=" * 60)

        # 1. 准备数据 (使用 CIFAR Transforms)
        dataset_train = FilelistDataset(train_file, transform=get_cifar_transforms(is_train=True))
        if len(dataset_train) == 0: continue
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        # 2. 准备模型 (应用结构修改)
        model = get_cifar_resnet18(PRETRAINED_CHECKPOINT)
        model.to(DEVICE)

        # 3. 优化器 (Global Finetune)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 12], gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        # ... (模型初始化和加载权重之后) ...

        # === 新增：起跑线检查 ===
        print("\n🔍 [Sanity Check] 训练前先测一次，确认起点是否为 88%...")
        initial_metrics = evaluate(model, criterion, test_loader, device=DEVICE)
        print(f"🏁 起跑线成绩: Acc@1 {initial_metrics:.2f}%\n")
        # ========================



        # 4. 训练
        best_acc = 0.0
        for epoch in range(args.epochs):
            train_one_epoch(model, criterion, optimizer, loader_train, DEVICE, epoch, args)
            lr_scheduler.step()

            # 评估
            eval_metrics = evaluate(model, criterion, test_loader, device=DEVICE)
            # 直接赋值，因为 eval_metrics 已经是计算好的准确率了
            acc1 = eval_metrics
            best_acc = max(best_acc, acc1)

        print(f"✨ 结果: Ratio {ratio} -> Best Acc: {best_acc:.2f}%")
        results.append([filename, ratio, f"{best_acc:.2f}"])

    # --- 保存 ---
    with open('finetune_cifar_arch_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Filename', 'Ratio', 'Best Acc'])
        w.writerows(results)
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()