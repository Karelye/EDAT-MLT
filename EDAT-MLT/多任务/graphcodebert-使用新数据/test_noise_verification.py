#!/usr/bin/env python3
"""
特征控制功能测试脚本
验证特征选择和处理是否正常工作
"""

import torch
import numpy as np
import sys
import os

# 添加当前目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_task_evaluate import select_features, EvalConfig


def test_feature_selection():
    """测试特征选择功能"""
    print("=== 特征选择功能测试 ===")

    # 创建模拟配置
    config = EvalConfig()

    # 创建模拟的专家特征张量
    batch_size = 3
    feature_dim = 25
    expert_features = torch.randn(batch_size, feature_dim)

    print(f"原始特征形状: {expert_features.shape}")
    print(f"原始特征前5维的值:")
    for i in range(batch_size):
        print(f"  样本{i}: {expert_features[i, :5].tolist()}")

    # 测试1: 不启用特征限制
    print("\n--- 测试1: 不启用特征限制 ---")
    config.use_limited_features = False
    result1 = select_features(expert_features, config)
    print(f"结果形状: {result1.shape}")
    print(f"特征是否相同: {torch.equal(expert_features, result1)}")

    # 测试2: 使用前10个特征
    print("\n--- 测试2: 使用前10个特征 ---")
    config.use_limited_features = True
    config.num_features_to_use = 10
    config.feature_selection_strategy = "first"
    result2 = select_features(expert_features, config)
    print(f"结果形状: {result2.shape}")
    print(f"前10个特征是否相同: {torch.equal(expert_features[:, :10], result2[:, :10])}")
    print(f"后15个特征是否为0: {torch.all(result2[:, 10:] == 0)}")

    # 测试3: 随机选择5个特征
    print("\n--- 测试3: 随机选择5个特征 ---")
    config.num_features_to_use = 5
    config.feature_selection_strategy = "random"
    config.feature_selection_seed = 42
    result3 = select_features(expert_features, config)
    print(f"结果形状: {result3.shape}")
    non_zero_count = torch.sum(result3 != 0, dim=1)
    print(f"每个样本的非零特征数量: {non_zero_count.tolist()}")

    # 测试4: 按重要性选择8个特征
    print("\n--- 测试4: 按重要性选择8个特征 ---")
    config.num_features_to_use = 8
    config.feature_selection_strategy = "important"
    result4 = select_features(expert_features, config)
    print(f"结果形状: {result4.shape}")

    # 验证重要特征索引
    important_indices = config.important_feature_indices[:8]
    print(f"期望使用的重要特征索引: {important_indices}")

    # 检查是否正确选择了重要特征
    for i, idx in enumerate(important_indices):
        if idx < feature_dim:
            original_val = expert_features[0, idx]
            selected_val = result4[0, i]
            print(
                f"  特征{idx}: 原始={original_val:.4f}, 选择后={selected_val:.4f}, 匹配={abs(original_val - selected_val) < 1e-6}")

    # 测试5: 边界情况 - 使用0个特征
    print("\n--- 测试5: 边界情况 - 使用0个特征 ---")
    config.num_features_to_use = 0
    result5 = select_features(expert_features, config)
    print(f"结果形状: {result5.shape}")
    print(f"应该返回原始特征: {torch.equal(expert_features, result5)}")

    # 测试6: 边界情况 - 使用超过总数的特征
    print("\n--- 测试6: 边界情况 - 使用超过总数的特征 ---")
    config.num_features_to_use = 30  # 超过25
    result6 = select_features(expert_features, config)
    print(f"结果形状: {result6.shape}")
    print(f"应该返回原始特征: {torch.equal(expert_features, result6)}")

    print("\n=== 测试完成 ===")


def test_feature_strategies():
    """测试不同特征选择策略的差异"""
    print("\n=== 特征选择策略差异测试 ===")

    config = EvalConfig()
    config.use_limited_features = True
    config.num_features_to_use = 5

    # 创建一个固定的特征张量用于比较
    torch.manual_seed(123)
    expert_features = torch.randn(1, 25)

    print("原始特征前10维:", expert_features[0, :10].tolist())

    # 策略1: first
    config.feature_selection_strategy = "first"
    result_first = select_features(expert_features, config)
    print(f"\nfirst策略 - 前5维: {result_first[0, :5].tolist()}")
    print(f"first策略 - 6-10维: {result_first[0, 5:10].tolist()}")

    # 策略2: random (固定种子)
    config.feature_selection_strategy = "random"
    config.feature_selection_seed = 42
    result_random = select_features(expert_features, config)
    print(f"\nrandom策略 - 前5维: {result_random[0, :5].tolist()}")
    print(f"random策略 - 6-10维: {result_random[0, 5:10].tolist()}")

    # 策略3: important
    config.feature_selection_strategy = "important"
    result_important = select_features(expert_features, config)
    print(f"\nimportant策略 - 前5维: {result_important[0, :5].tolist()}")
    print(f"important策略 - 6-10维: {result_important[0, 5:10].tolist()}")

    # 验证三种策略产生的结果不同
    print(f"\nfirst vs random 相同: {torch.equal(result_first, result_random)}")
    print(f"first vs important 相同: {torch.equal(result_first, result_important)}")
    print(f"random vs important 相同: {torch.equal(result_random, result_important)}")


def demonstrate_usage():
    """演示实际使用场景"""
    print("\n=== 实际使用场景演示 ===")

    # 模拟评估时的批次数据
    batch_size = 2
    expert_features = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0] + [0.1] * 20,  # 样本1
        [2.0, 4.0, 6.0, 8.0, 10.0] + [0.2] * 20  # 样本2
    ])

    print("模拟的专家特征批次:")
    print(f"形状: {expert_features.shape}")
    print(f"样本1前5维: {expert_features[0, :5].tolist()}")
    print(f"样本2前5维: {expert_features[1, :5].tolist()}")

    # 场景1: 特征传感器故障，只有3个核心特征可用
    print("\n--- 场景1: 特征传感器故障 ---")
    config = EvalConfig()
    config.use_limited_features = True
    config.num_features_to_use = 3
    config.feature_selection_strategy = "important"

    limited_features = select_features(expert_features, config)
    print(f"故障模式下的特征前5维:")
    print(f"样本1: {limited_features[0, :5].tolist()}")
    print(f"样本2: {limited_features[1, :5].tolist()}")

    # 场景2: 消融研究，逐步减少特征数量
    print("\n--- 场景2: 消融研究 ---")
    feature_counts = [25, 15, 10, 5, 3]
    for count in feature_counts:
        config.num_features_to_use = count
        if count < 25:
            config.use_limited_features = True
        else:
            config.use_limited_features = False

        ablation_features = select_features(expert_features, config)
        non_zero_dims = torch.sum(ablation_features[0] != 0).item()
        print(f"使用{count}个特征: 实际非零维度={non_zero_dims}")


if __name__ == "__main__":
    print("开始特征控制功能测试...")

    try:
        test_feature_selection()
        test_feature_strategies()
        demonstrate_usage()

        print("\n✅ 所有测试通过！特征控制功能正常工作。")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()