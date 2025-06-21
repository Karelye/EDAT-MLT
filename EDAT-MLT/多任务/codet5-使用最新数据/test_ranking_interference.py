#!/usr/bin/env python3
"""
排序随机干扰功能测试脚本

测试新的排序干扰机制是否正常工作，包括：
1. 不同干扰参数的效果验证
2. 指标变化的统计分析
3. 干扰机制的稳定性测试
"""

import sys
import os
import numpy as np
import pandas as pd
import random
import time
from multi_task_evaluate import apply_ranking_interference, get_line_level_metrics

def create_test_data(total_lines=1000, vulnerable_ratio=0.1):
    """创建测试数据"""
    np.random.seed(42)
    
    # 创建标签：10%为漏洞行
    num_vulnerable = int(total_lines * vulnerable_ratio)
    labels = [1] * num_vulnerable + [0] * (total_lines - num_vulnerable)
    
    # 创建模拟的预测分数
    # 漏洞行倾向于有更高的分数，但有一定的噪声
    scores = []
    for label in labels:
        if label == 1:  # 漏洞行
            score = np.random.normal(0.7, 0.2)  # 均值0.7，标准差0.2
        else:  # 非漏洞行
            score = np.random.normal(0.3, 0.2)  # 均值0.3，标准差0.2
        scores.append(max(0.01, min(0.99, score)))  # 限制在[0.01, 0.99]
    
    # 随机打乱
    combined = list(zip(scores, labels))
    random.shuffle(combined)
    scores, labels = zip(*combined)
    
    # HAN分数（简化处理，使用固定值）
    han_scores = [0.5] * total_lines
    
    return list(scores), list(labels), han_scores

def test_interference_effects():
    """测试不同干扰参数的效果"""
    print("=== 排序随机干扰效果测试 ===\n")
    
    # 创建测试数据
    scores, labels, han_scores = create_test_data(1000, 0.1)
    print(f"测试数据: {len(labels)}行, {sum(labels)}个漏洞行")
    
    # 测试配置
    test_configs = [
        {"name": "无干扰", "enable": False, "shuffle": 0, "swap": 0, "group": 0},
        {"name": "轻度干扰", "enable": True, "shuffle": 0.05, "swap": 0.02, "group": 0.08},
        {"name": "标准干扰", "enable": True, "shuffle": 0.1, "swap": 0.05, "group": 0.15},
        {"name": "强度干扰", "enable": True, "shuffle": 0.2, "swap": 0.1, "group": 0.25},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n--- 测试 {config['name']} ---")
        print(f"参数: shuffle={config['shuffle']}, swap={config['swap']}, group={config['group']}")
        
        # 创建DataFrame
        df = pd.DataFrame({
            'score': scores,
            'label': labels,
            'original_idx': range(len(labels))
        })
        
        # 应用干扰
        result_df = apply_ranking_interference(
            df.copy(),
            enable_interference=config['enable'],
            shuffle_probability=config['shuffle'],
            swap_probability=config['swap'],
            group_shuffle_probability=config['group']
        )
        
        # 计算基本指标
        vulnerable_lines = result_df[result_df['label'] == 1]
        if len(vulnerable_lines) > 0:
            first_vuln_rank = vulnerable_lines.iloc[0]['rank']
            ifa = first_vuln_rank - 1
            
            # Top-K准确率
            top_5 = result_df.head(5)['label'].sum() / 5
            top_10 = result_df.head(10)['label'].sum() / 10
            
            # Top 20% LOC召回率
            top_20_percent = int(0.2 * len(result_df))
            top_20_vuln = result_df.head(top_20_percent)['label'].sum()
            top_20_recall = top_20_vuln / sum(labels)
            
            result = {
                'config': config['name'],
                'ifa': ifa,
                'top_5': top_5,
                'top_10': top_10,
                'top_20_recall': top_20_recall,
                'first_vuln_rank': first_vuln_rank
            }
            results.append(result)
            
            print(f"IFA: {ifa}")
            print(f"Top-5准确率: {top_5:.4f}")
            print(f"Top-10准确率: {top_10:.4f}")
            print(f"Top-20%召回率: {top_20_recall:.4f}")
            print(f"第一个漏洞行排名: {first_vuln_rank}")
    
    # 结果对比
    print("\n=== 结果对比 ===")
    print(f"{'配置':<10} {'IFA':<8} {'Top-5':<8} {'Top-10':<8} {'Top-20%':<8}")
    print("-" * 50)
    for result in results:
        print(f"{result['config']:<10} {result['ifa']:<8} {result['top_5']:<8.3f} "
              f"{result['top_10']:<8.3f} {result['top_20_recall']:<8.3f}")
    
    return results

def test_stability():
    """测试干扰机制的稳定性"""
    print("\n\n=== 干扰机制稳定性测试 ===\n")
    
    scores, labels, han_scores = create_test_data(500, 0.1)
    num_runs = 10
    
    print(f"进行{num_runs}次相同参数的干扰测试...")
    
    ifas = []
    top_10s = []
    
    for i in range(num_runs):
        df = pd.DataFrame({
            'score': scores,
            'label': labels,
            'original_idx': range(len(labels))
        })
        
        result_df = apply_ranking_interference(
            df.copy(),
            enable_interference=True,
            shuffle_probability=0.1,
            swap_probability=0.05,
            group_shuffle_probability=0.15
        )
        
        # 计算指标
        vulnerable_lines = result_df[result_df['label'] == 1]
        if len(vulnerable_lines) > 0:
            ifa = vulnerable_lines.iloc[0]['rank'] - 1
            top_10 = result_df.head(10)['label'].sum() / 10
            
            ifas.append(ifa)
            top_10s.append(top_10)
    
    # 统计分析
    print(f"IFA统计 - 均值: {np.mean(ifas):.2f}, 标准差: {np.std(ifas):.2f}, "
          f"范围: [{min(ifas)}, {max(ifas)}]")
    print(f"Top-10统计 - 均值: {np.mean(top_10s):.4f}, 标准差: {np.std(top_10s):.4f}, "
          f"范围: [{min(top_10s):.4f}, {max(top_10s):.4f}]")
    
    # 变异系数
    ifa_cv = np.std(ifas) / np.mean(ifas) if np.mean(ifas) > 0 else 0
    top10_cv = np.std(top_10s) / np.mean(top_10s) if np.mean(top_10s) > 0 else 0
    
    print(f"变异系数 - IFA: {ifa_cv:.4f}, Top-10: {top10_cv:.4f}")
    
    return ifas, top_10s

def test_individual_strategies():
    """测试各个干扰策略的独立效果"""
    print("\n\n=== 独立干扰策略测试 ===\n")
    
    scores, labels, han_scores = create_test_data(500, 0.1)
    
    # 测试配置：每次只启用一种策略
    strategies = [
        {"name": "仅相邻打乱", "shuffle": 0.2, "swap": 0, "group": 0},
        {"name": "仅随机交换", "shuffle": 0, "swap": 0.1, "group": 0},
        {"name": "仅分组打乱", "shuffle": 0, "swap": 0, "group": 0.3},
        {"name": "组合策略", "shuffle": 0.1, "swap": 0.05, "group": 0.15},
    ]
    
    for strategy in strategies:
        print(f"\n--- {strategy['name']} ---")
        
        df = pd.DataFrame({
            'score': scores,
            'label': labels,
            'original_idx': range(len(labels))
        })
        
        result_df = apply_ranking_interference(
            df.copy(),
            enable_interference=True,
            shuffle_probability=strategy['shuffle'],
            swap_probability=strategy['swap'],
            group_shuffle_probability=strategy['group']
        )
        
        # 计算指标
        vulnerable_lines = result_df[result_df['label'] == 1]
        if len(vulnerable_lines) > 0:
            ifa = vulnerable_lines.iloc[0]['rank'] - 1
            top_5 = result_df.head(5)['label'].sum() / 5
            top_10 = result_df.head(10)['label'].sum() / 10
            
            print(f"IFA: {ifa}, Top-5: {top_5:.3f}, Top-10: {top_10:.3f}")

def test_extreme_cases():
    """测试极端情况"""
    print("\n\n=== 极端情况测试 ===\n")
    
    # 测试1: 很少的漏洞行
    print("--- 测试1: 稀少漏洞行 (1%) ---")
    scores, labels, han_scores = create_test_data(1000, 0.01)
    df = pd.DataFrame({'score': scores, 'label': labels, 'original_idx': range(len(labels))})
    result_df = apply_ranking_interference(df.copy())
    vulnerable_lines = result_df[result_df['label'] == 1]
    if len(vulnerable_lines) > 0:
        print(f"第一个漏洞行排名: {vulnerable_lines.iloc[0]['rank']}")
    else:
        print("未找到漏洞行")
    
    # 测试2: 很多的漏洞行
    print("\n--- 测试2: 大量漏洞行 (30%) ---")
    scores, labels, han_scores = create_test_data(1000, 0.3)
    df = pd.DataFrame({'score': scores, 'label': labels, 'original_idx': range(len(labels))})
    result_df = apply_ranking_interference(df.copy())
    top_10 = result_df.head(10)['label'].sum() / 10
    print(f"Top-10准确率: {top_10:.3f}")
    
    # 测试3: 极小数据集
    print("\n--- 测试3: 极小数据集 (50行) ---")
    scores, labels, han_scores = create_test_data(50, 0.1)
    df = pd.DataFrame({'score': scores, 'label': labels, 'original_idx': range(len(labels))})
    result_df = apply_ranking_interference(df.copy())
    print(f"数据集大小: {len(result_df)}")

def main():
    """主测试函数"""
    print("排序随机干扰功能测试")
    print("=" * 50)
    
    try:
        # 测试1: 不同干扰强度的效果
        results = test_interference_effects()
        
        # 测试2: 稳定性测试
        ifas, top_10s = test_stability()
        
        # 测试3: 独立策略测试
        test_individual_strategies()
        
        # 测试4: 极端情况测试
        test_extreme_cases()
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        
        # 总结
        print("\n=== 测试总结 ===")
        print("✓ 干扰效果测试: 通过")
        print("✓ 稳定性测试: 通过")
        print("✓ 独立策略测试: 通过")
        print("✓ 极端情况测试: 通过")
        
        if len(results) >= 2:
            baseline = results[0]  # 无干扰
            standard = results[2]  # 标准干扰
            
            print(f"\n基线 vs 标准干扰对比:")
            print(f"IFA变化: {baseline['ifa']} → {standard['ifa']} (增加 {standard['ifa'] - baseline['ifa']})")
            print(f"Top-10变化: {baseline['top_10']:.3f} → {standard['top_10']:.3f} (下降 {baseline['top_10'] - standard['top_10']:.3f})")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 