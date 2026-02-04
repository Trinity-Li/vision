import torch
import os
import sys
import glob
import csv
import re  # 用于提取文件名中的数字


# ==========================================
# 1. 核心计算函数 (计算全局体积)
# ==========================================
def calculate_global_volume(features):
    """
    计算特征矩阵的感知流形体积。
    features: (512, m) 的张量
    """
    if features.dtype != torch.float32 and features.dtype != torch.float64:
        features = features.float()

    p, m = features.shape
    if m <= 1: return 0.0

    # 标准计算步骤
    z_mean = torch.mean(features, dim=1, keepdim=True)
    z_centered = features - z_mean
    cov_term = torch.mm(z_centered, z_centered.t()) / m
    identity = torch.eye(p, device=features.device)
    sigma_adjusted = identity + cov_term
    log_det_val = torch.logdet(sigma_adjusted)
    vol_z = 0.5 * (log_det_val / torch.log(torch.tensor(2.0)))

    return vol_z.item()


# ==========================================
# 2. 辅助函数：加载数据与提取特征
# ==========================================
def load_features_from_file(file_path):
    """兼容 List[Dict] 和 Dict[Tensor] 两种格式"""
    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"  ❌ 文件损坏: {e}")
        return None

    # 格式 A: 紧凑格式 (Dict)
    if isinstance(data, dict) and 'features' in data:
        return data['features']  # (N, 512)

    # 格式 B: 松散格式 (List)
    elif isinstance(data, list):
        # 提取 feature 字段并堆叠
        feat_list = [item['feature'] for item in data if 'feature' in item]
        if feat_list:
            return torch.stack(feat_list)

    return None


# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    # --- 配置 ---
    # 1. 原始文件路径 (请修改为你实际的原始文件路径)
    ORIGINAL_FILE = 'autodl-tmp/eval/references/classification/eval_set_B_features.pt'

    # 2. 扰动文件搜索模式
    PERTURBED_PATTERN = 'eval_set_B_perturbed_*.pt'

    # 3. 输出结果文件
    OUTPUT_CSV = 'global_volume_comparison.csv'

    # --- 搜集所有文件 ---
    tasks = []

    # 1. 添加原始文件 (如果存在)
    if os.path.exists(ORIGINAL_FILE):
        tasks.append({
            'path': ORIGINAL_FILE,
            'type': 'Original',
            'ratio': 0.0  # 原始数据删除率为 0%
        })
    else:
        # 尝试在当前目录找
        base_name = os.path.basename(ORIGINAL_FILE)
        if os.path.exists(base_name):
            tasks.append({'path': base_name, 'type': 'Original', 'ratio': 0.0})
        else:
            print(f"⚠️ 警告: 未找到原始文件 {ORIGINAL_FILE}")

    # 2. 添加扰动文件
    perturbed_files = glob.glob(PERTURBED_PATTERN)
    for p_file in perturbed_files:
        # 从文件名中提取比例 (例如 ..._0.2.pt -> 0.2)
        # 使用正则寻找 0.x 的数字
        match = re.search(r'_(\d+\.\d+)\.pt', p_file)
        ratio = float(match.group(1)) if match else -1.0

        tasks.append({
            'path': p_file,
            'type': 'Perturbed',
            'ratio': ratio
        })

    # 3. 按比例排序 (0.0 -> 0.1 -> 0.2 ...)
    tasks.sort(key=lambda x: x['ratio'])

    print(f"📋 找到 {len(tasks)} 个文件任务，准备开始...\n")
    print(f"{'Type':<10} {'Ratio':<8} {'Samples':<10} {'Global Volume':<15}")
    print("-" * 50)

    results = []

    # --- 循环处理 ---
    for task in tasks:
        file_path = task['path']
        ratio = task['ratio']
        dtype = task['type']

        # 1. 加载特征
        features = load_features_from_file(file_path)

        if features is None:
            continue

        n_samples = features.shape[0]

        # 2. 核心：转置并计算
        # features 是 (N, 512)，计算需要 (512, N)
        z_input = features.t()

        vol = calculate_global_volume(z_input)

        # 3. 打印与记录
        print(f"{dtype:<10} {ratio:<8} {n_samples:<10} {vol:.4f}")

        results.append([dtype, ratio, n_samples, f"{vol:.6f}", os.path.basename(file_path)])

    # --- 保存 CSV ---
    print("-" * 50)
    print(f"💾 正在保存对比报告到 {OUTPUT_CSV} ...")

    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Type', 'Deletion Ratio', 'Sample Count', 'Global Volume', 'Filename'])
        writer.writerows(results)

    print("✅ 完成！")