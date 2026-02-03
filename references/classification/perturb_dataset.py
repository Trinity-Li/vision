import torch
import random
import copy
import os
import sys


# ==========================================
# 1. 核心处理函数 (保持不变)
# ==========================================
def perturb_dataset(data, drop_ratio=0.1):
    """
    对数据集进行扰动：
    1. 按类别随机删除 drop_ratio 比例的样本。
    2. 在该类剩余样本中随机选一张，复制并填补删除的空缺。
    """
    print(f"\n⚡ [处理中] 目标删除比例: {drop_ratio * 100:.0f}%")

    # 1. 按类别分组
    class_buckets = {}
    for item in data:
        label = item['label']
        if label not in class_buckets:
            class_buckets[label] = []
        class_buckets[label].append(item)

    final_data = []

    # 2. 遍历每个类别
    for label in sorted(class_buckets.keys()):
        items = class_buckets[label]
        original_count = len(items)

        # 计算数量
        n_drop = int(original_count * drop_ratio)
        n_keep = original_count - n_drop

        if n_keep < 1:
            # 极少数情况：如果比例太高导致一张不剩，强制保留一张用于复制
            n_keep = 1
            n_drop = original_count - 1
            print(f"  ! 警告: 类别 {label} 样本过少，强制保留1张。")

        # A. 随机打乱并截取
        random.shuffle(items)
        kept_items = items[:n_keep]

        # B. 选种子
        seed_item = random.choice(kept_items)

        # C. 复制填补
        duplicates = []
        for i in range(n_drop):
            dup = copy.deepcopy(seed_item)
            # 修改 ID 以示区分
            if 'id' in dup and isinstance(dup['id'], str):
                base_id = os.path.splitext(dup['id'])[0]
                ext = os.path.splitext(dup['id'])[1]
                dup['id'] = f"{base_id}_r{drop_ratio}_copy{i}{ext}"
            duplicates.append(dup)

        # 合并
        new_class_list = kept_items + duplicates
        assert len(new_class_list) == original_count
        final_data.extend(new_class_list)

    return final_data


# ==========================================
# 2. 主程序 (修改支持批量保存)
# ==========================================
if __name__ == "__main__":
    # --- 配置区域 ---
    input_path = 'autodl-tmp/eval/references/classification/eval_set_B_features.pt'

    # 在这里定义你想要生成的比例列表 (0.1 代表 10%, 0.5 代表 50%)
    RATIOS_TO_GENERATE = [0.2, 0.4, 0.6, 0.8, 1.0]

    # 输出文件名的前缀
    output_prefix = 'eval_set_B_perturbed'

    # --- 1. 加载源数据 (只加载一次) ---
    if not os.path.exists(input_path):
        input_path = os.path.basename(input_path)  # 尝试当前目录

    print(f"正在加载源数据: {input_path} ...")
    try:
        source_data = torch.load(input_path, map_location='cpu')
        print(f"✅ 源数据加载成功，共 {len(source_data)} 条样本。")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        sys.exit(1)

    # --- 2. 循环生成并保存 ---
    for ratio in RATIOS_TO_GENERATE:
        # 为了不污染源数据，每次处理前确保使用深拷贝是不够的，
        # 因为 perturb_dataset 内部已经做了处理，
        # 但为了保险起见，我们传进去的数据列表本身会在函数内被切片读取，
        # 只要不修改原列表里的对象引用即可。
        # 上面的 perturb_dataset 实现是安全的（使用了 copy.deepcopy 生成新元素）。

        try:
            # 执行扰动
            new_data = perturb_dataset(source_data, drop_ratio=ratio)

            # 构造带比例的文件名，例如: eval_set_B_perturbed_0.2.pt
            save_name = f"{output_prefix}_{ratio}.pt"

            # 保存
            print(f"💾 正在保存到: {save_name} ...")
            torch.save(new_data, save_name)
            print(f"✅ 完成比例 {ratio} 的生成。\n")

        except Exception as e:
            print(f"❌ 处理比例 {ratio} 时出错: {e}")
            continue

    print("🎉 所有任务执行完毕！")