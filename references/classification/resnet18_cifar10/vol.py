import torch
import csv
import os
import sys


# ==========================================
# 1. 核心计算函数 (基于文档公式)
# ==========================================
def calculate_class_volume(features):
    """
    根据文档计算类别体积 Vol(Z)。

    参数:
    features (torch.Tensor): 形状必须为 (512, n)，即 (特征维度 p x 样本数 m)

    返回:
    float: 该类别的感知流形体积
    """
    # 数据类型防御：确保使用 float32 或 float64 避免精度丢失
    if features.dtype != torch.float32 and features.dtype != torch.float64:
        features = features.float()

    p, m = features.shape  # p=512, m=样本数

    # 边界情况处理：如果样本数太少，无法计算体积
    if m <= 1:
        return 0.0

    # --- 步骤 1: 特征收集与均值计算 ---
    # Z_mean = (1/m) Σ z_i
    # dim=1 表示沿样本方向求均值, keepdim=True 保持形状为 (512, 1) 以便广播
    z_mean = torch.mean(features, dim=1, keepdim=True)

    # --- 步骤 2: 特征中心化处理 ---
    # Ẑ = Z - Z_mean
    z_centered = features - z_mean

    # --- 步骤 3: 构建正则化协方差矩阵 ---
    # 计算 (1/m) Ẑ Ẑ^T
    # 结果是一个 p x p (512 x 512) 的矩阵
    cov_term = torch.mm(z_centered, z_centered.t()) / m

    # Σ_adjusted = I + (1/m) Ẑ Ẑ^T
    identity = torch.eye(p, device=features.device)
    sigma_adjusted = identity + cov_term

    # --- 步骤 4: 类别体积计算 ---
    # Vol(Z) = (1/2) log₂ det( Σ_adjusted )
    # 使用 logdet 避免直接计算 det 导致的数值溢出
    log_det_val = torch.logdet(sigma_adjusted)

    # 换底公式: log2(x) = ln(x) / ln(2)
    # Vol = 0.5 * (ln_det / ln_2)
    vol_z = 0.5 * (log_det_val / torch.log(torch.tensor(2.0)))

    return vol_z.item()


# ==========================================
# 2. 主程序逻辑
# ==========================================
if __name__ == "__main__":
    # --- 配置路径 ---
    # 优先使用你提供的完整路径
    file_path = 'autodl-tmp/eval/references/classification/eval_set_B_features.pt'

    # 智能检查：如果完整路径不存在，尝试在当前目录下查找文件名
    if not os.path.exists(file_path):
        filename = os.path.basename(file_path)
        if os.path.exists(filename):
            print(f"提示: 在当前目录下找到了 {filename}，将使用该文件。")
            file_path = filename
        else:
            print(f"错误: 找不到文件 {file_path}")
            print("请确认 .pt 文件是否在正确的位置。")
            sys.exit(1)

    print(f"正在加载数据: {file_path} ...")

    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"加载失败: {e}")
        sys.exit(1)

    # --- 数据分组 (Grouping) ---
    print(f"数据加载成功，共 {len(data)} 条样本。正在按类别分组...")

    class_buckets = {}  # 结构: { label_id: {'features': [], 'name': 'xxx'} }

    # 遍历数据列表 (List of Dicts)
    for item in data:
        # 根据你的数据结构提取字段
        label = item.get('label')
        feat = item.get('feature')
        name = item.get('class_name', str(label))  # 如果没有 class_name，用 label 代替

        if label is None or feat is None:
            continue

        if label not in class_buckets:
            class_buckets[label] = {'features': [], 'name': name}

        class_buckets[label]['features'].append(feat)

    # --- 循环计算 (Calculation) ---
    print("-" * 65)
    print(f"{'Class Name':<20} {'ID':<10} {'Samples':<15} {'Vol(Z)':<15}")
    print("-" * 65)

    results_list = []

    # 排序是为了输出整齐，按 label ID 升序
    for label in sorted(class_buckets.keys()):
        info = class_buckets[label]
        feat_list = info['features']
        class_name = info['name']
        n = len(feat_list)

        if n == 0:
            continue

        # 关键步骤 1: 堆叠 (Stack) -> (n, 512)
        # 关键步骤 2: 转置 (.t()) -> (512, n) 以符合公式 Z ∈ R^{p×m}
        features_matrix = torch.stack(feat_list).t()

        # 计算体积
        vol = calculate_class_volume(features_matrix)

        # 记录结果
        results_list.append({
            'class_name': class_name,
            'label_id': label,
            'sample_count': n,
            'volume': vol
        })

        print(f"{class_name:<20} {label:<10} {n:<15} {vol:.4f}")

    # --- 保存结果到 CSV ---
    csv_filename = 'class_volumes_result.csv'
    print("-" * 65)
    print(f"正在保存结果到 {csv_filename} ...")

    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['Class Name', 'Label ID', 'Sample Count', 'Volume (Vol Z)'])

            # 写入数据
            for row in results_list:
                writer.writerow([
                    row['class_name'],
                    row['label_id'],
                    row['sample_count'],
                    f"{row['volume']:.6f}"
                ])
        print(f"✅ 保存成功！文件位于: {os.path.abspath(csv_filename)}")
    except Exception as e:
        print(f"❌ 保存 CSV 失败: {e}")