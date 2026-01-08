import torch


def check():
    file_path = 'eval_set_B_features.pt'
    print(f"正在加载 {file_path} ...")

    try:
        data = torch.load(file_path, weights_only=False)
    except FileNotFoundError:
        print("错误：找不到文件，请确认文件名或路径正确。")
        return

    # 1. 检查列表长度
    total = len(data)
    print(f"\n[√] 加载成功！列表长度: {total}")
    if total != 25000:
        print(f"    [!] 警告: 预期 25000 张，实际只有 {total} 张")

    # 2. 检查第一条数据
    if total > 0:
        sample = data[0]

        # 检查 Key
        keys = list(sample.keys())
        print(f"[√] 包含的字段: {keys}")

        # 检查 Feature
        feat = sample['feature']
        if torch.is_tensor(feat) and feat.shape[0] == 512:
            print(f"[√] 特征向量格式正确: Tensor {feat.shape}")
        else:
            print(f"    [!] 错误: 特征向量格式不对: {type(feat)} {feat.shape if torch.is_tensor(feat) else ''}")

        # 检查 ID
        print(f"[√] 样本 ID 示例: {sample['id']}")

        # 检查 Label
        print(f"[√] 样本 Label 示例: {sample['label']}")

    print("\n检查完成。")


if __name__ == '__main__':
    check()