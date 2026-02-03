import torch
import csv
import os
import sys
import glob  # 新增：用于查找文件


# ==========================================
# 1. 核心计算函数 (保持不变)
# ==========================================
def calculate_class_volume(features):
    """
    根据文档计算类别体积 Vol(Z)。
    features: (512, n) 的张量
    """
    if features.dtype != torch.float32 and features.dtype != torch.float64:
        features = features.float()

    p, m = features.shape
    if m <= 1: return 0.0  # 样本太少无法计算

    # 步骤 1-4 (文档公式实现)
    z_mean = torch.mean(features, dim=1, keepdim=True)
    z_centered = features - z_mean
    cov_term = torch.mm(z_centered, z_centered.t()) / m
    identity = torch.eye(p, device=features.device)
    sigma_adjusted = identity + cov_term
    log_det_val = torch.logdet(sigma_adjusted)
    vol_z = 0.5 * (log_det_val / torch.log(torch.tensor(2.0)))

    return vol_z.item()


# ==========================================
# 2. 批量处理主程序
# ==========================================
if __name__ == "__main__":

    # --- 配置区域 ---
    # 1. 设置要搜索的文件模式 (支持通配符 *)
    # 例如：计算当前目录下所有以 eval_set_B_perturbed 开头的 .pt 文件
    SEARCH_PATTERN = 'eval_set_B_perturbed_*.pt'

    # 或者指定某个文件夹下的所有 pt 文件：
    # SEARCH_PATTERN = 'autodl-tmp/eval/data/*.pt'

    # 2. 输出结果文件名
    OUTPUT_CSV = 'batch_volumes_report.csv'

    # --- 开始搜索 ---
    files = glob.glob(SEARCH_PATTERN)
    # 按文件名排序，保证处理顺序 (0.1, 0.2, ...)
    files.sort()

    if not files:
        print(f"❌ 未找到符合模式 '{SEARCH_PATTERN}' 的文件！请检查路径。")
        sys.exit(1)

    print(f"📂 找到 {len(files)} 个文件，准备开始计算...\n")

    # 用于存储所有文件的汇总结果
    all_results = []

    # --- 外层循环：遍历文件 ---
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"正在处理: {filename} ...")

        try:
            data = torch.load(file_path, map_location='cpu')
        except Exception as e:
            print(f"  ❌ 加载失败，跳过: {e}")
            continue

        # 内存分组
        class_buckets = {}
        for item in data:
            label = item.get('label')
            feat = item.get('feature')
            name = item.get('class_name', str(label))

            if label is None or feat is None: continue
            if label not in class_buckets:
                class_buckets[label] = {'features': [], 'name': name}
            class_buckets[label]['features'].append(feat)

        # --- 内层循环：遍历类别计算 ---
        file_success_count = 0
        for label in sorted(class_buckets.keys()):
            info = class_buckets[label]
            feat_list = info['features']
            class_name = info['name']
            n = len(feat_list)

            if n == 0: continue

            # 堆叠与转置 (n, 512) -> (512, n)
            features_matrix = torch.stack(feat_list).t()

            # 计算
            vol = calculate_class_volume(features_matrix)

            # 收集结果 (增加了 filename 字段)
            all_results.append({
                'filename': filename,
                'class_name': class_name,
                'label_id': label,
                'sample_count': n,
                'volume': vol
            })
            file_success_count += 1

        print(f"  ✅ 完成，计算了 {file_success_count} 个类别。")

    # --- 保存汇总结果到 CSV ---
    print("-" * 60)
    print(f"正在保存汇总结果到 {OUTPUT_CSV} ...")

    try:
        with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 表头增加了一列 'Filename'
            header = ['Filename', 'Class Name', 'Label ID', 'Sample Count', 'Volume (Vol Z)']
            writer.writerow(header)

            for row in all_results:
                writer.writerow([
                    row['filename'],
                    row['class_name'],
                    row['label_id'],
                    row['sample_count'],
                    f"{row['volume']:.6f}"
                ])
        print(f"🎉 批量计算完成！结果已保存。")
    except Exception as e:
        print(f"❌ 保存 CSV 失败: {e}")