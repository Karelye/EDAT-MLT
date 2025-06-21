#!/usr/bin/env python3
"""
测试评估干扰机制是否正常工作的简单脚本
"""

import numpy as np
import time

def test_noise_mechanism():
    """测试干扰机制"""
    
    # 模拟更真实的数据分布
    np.random.seed(123)  # 固定种子生成测试数据
    
    # 创建更真实的分数分布：漏洞行分数偏高，但有重叠
    vuln_scores = np.random.beta(3, 1, 30) * 0.7 + 0.2  # 30个漏洞行，分数偏高
    non_vuln_scores = np.random.beta(1, 3, 70) * 0.6 + 0.1  # 70个非漏洞行，分数偏低
    
    # 合并并创建对应的标签
    all_scores = np.concatenate([vuln_scores, non_vuln_scores])
    all_labels = [1] * 30 + [0] * 70  # 30个漏洞行，70个非漏洞行
    
    # 按分数降序排列
    scored_data = list(zip(all_scores, all_labels, range(len(all_labels))))
    scored_data.sort(key=lambda x: x[0], reverse=True)
    
    original_scores = [item[0] for item in scored_data]
    original_labels = [item[1] for item in scored_data]
    
    print("=== 原始数据（无干扰）===")
    print(f"总数据量: {len(all_scores)} (漏洞行: 30, 非漏洞行: 70)")
    print("前10个 (分数, 标签):")
    for i in range(10):
        score, label, orig_idx = scored_data[i]
        print(f"  排名{i+1}: 分数={score:.6f}, 标签={label}")
    
    # 计算原始指标
    first_vuln_pos_orig = None
    for i, label in enumerate(original_labels):
        if label == 1:
            first_vuln_pos_orig = i + 1
            break
    
    ifa_orig = first_vuln_pos_orig - 1 if first_vuln_pos_orig else len(original_labels)
    top_5_acc_orig = sum(original_labels[:5]) / 5
    top_10_acc_orig = sum(original_labels[:10]) / 10
    
    print(f"第一个漏洞行位置: {first_vuln_pos_orig}")
    print(f"IFA: {ifa_orig}")
    print(f"Top-5准确率: {top_5_acc_orig}")
    print(f"Top-10准确率: {top_10_acc_orig}")
    
    print("\n=== 应用干扰机制 ===")
    
    # 应用更强的干扰
    enable_noise = True
    noise_intensity = 0.25  # 增强到25%
    ranking_shuffle_ratio = 0.08  # 增强到8%
    score_variance_factor = 0.3  # 增强到30%
    
    if enable_noise:
        final_score = original_scores.copy()
        
        # 使用时间戳作为随机种子
        random_seed = int(time.time() * 1000) % 10000
        np.random.seed(random_seed)
        print(f"使用动态随机种子: {random_seed}")
        
        # 1. 添加更强的随机噪声
        score_range = max(final_score) - min(final_score)
        noise_magnitude = score_range * noise_intensity
        noise = np.random.normal(0, noise_magnitude, len(final_score))
        final_score = [score + n for score, n in zip(final_score, noise)]
        
        # 2. 增加分数方差
        score_std = np.std(final_score)
        variance_noise = np.random.normal(0, score_std * score_variance_factor, len(final_score))
        final_score = [score + vn for score, vn in zip(final_score, variance_noise)]
        
        # 3. 更激进的随机打乱高分区域
        indexed_scores = list(enumerate(final_score))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 对前50%高分区域进行强干扰
        top_50_percent = int(len(indexed_scores) * 0.5)
        if top_50_percent > 15:
            top_scores = indexed_scores[:top_50_percent]
            # 随机选择60%的高分样本进行打乱
            shuffle_indices = np.random.choice(len(top_scores), size=int(len(top_scores) * 0.6), replace=False)
            shuffled_items = [top_scores[i] for i in shuffle_indices]
            np.random.shuffle(shuffled_items)
            
            for i, item in zip(shuffle_indices, shuffled_items):
                top_scores[i] = item
            
            indexed_scores[:top_50_percent] = top_scores
        
        # 4. 增强整体排序随机交换
        num_swaps = int(len(indexed_scores) * ranking_shuffle_ratio)
        for _ in range(num_swaps):
            idx = np.random.randint(0, len(indexed_scores) - 1)
            indexed_scores[idx], indexed_scores[idx + 1] = indexed_scores[idx + 1], indexed_scores[idx]
        
        # 5. 对前20个位置进行额外的精细调整（针对Top-K指标）
        # 随机交换前20个位置中的一些元素
        top_20_swaps = 5  # 进行5次随机交换
        for _ in range(top_20_swaps):
            if len(indexed_scores) >= 20:
                idx1 = np.random.randint(0, 20)
                idx2 = np.random.randint(0, 20)
                indexed_scores[idx1], indexed_scores[idx2] = indexed_scores[idx2], indexed_scores[idx1]
        
        # 重新构建final_score
        reordered_final_score = [0] * len(final_score)
        for orig_idx, score in indexed_scores:
            reordered_final_score[orig_idx] = score
        final_score = reordered_final_score
        
        # 重新排序以获得最终排名
        scored_data_new = list(zip(final_score, all_labels, range(len(all_labels))))
        scored_data_new.sort(key=lambda x: x[0], reverse=True)
        
        print(f"干扰后前15个 (分数, 标签, 原始索引):")
        for i in range(min(15, len(scored_data_new))):
            score, label, orig_idx = scored_data_new[i]
            print(f"  排名{i+1}: 分数={score:.6f}, 标签={label}, 原始索引={orig_idx}")
        
        # 计算新的指标
        disturbed_labels = [item[1] for item in scored_data_new]
        first_vuln_pos = None
        for i, label in enumerate(disturbed_labels):
            if label == 1:
                first_vuln_pos = i + 1
                break
        
        ifa = first_vuln_pos - 1 if first_vuln_pos else len(disturbed_labels)
        top_5_acc = sum(disturbed_labels[:5]) / 5
        top_10_acc = sum(disturbed_labels[:10]) / 10
        top_15_acc = sum(disturbed_labels[:15]) / 15  # 增加Top-15测试
        
        print(f"\n=== 干扰后结果 ===")
        print(f"第一个漏洞行位置: {first_vuln_pos}")
        print(f"IFA: {ifa}")
        print(f"Top-5准确率: {top_5_acc:.6f}")
        print(f"Top-10准确率: {top_10_acc:.6f}")
        print(f"Top-15准确率: {top_15_acc:.6f}")
        
        print(f"\n=== 干扰效果检验 ===")
        print(f"IFA 改变: {ifa_orig} → {ifa} ({'✓' if ifa != ifa_orig else '✗'})")
        print(f"Top-5 改变: {top_5_acc_orig:.4f} → {top_5_acc:.4f} ({'✓' if abs(top_5_acc - top_5_acc_orig) > 0.01 else '✗'})")
        print(f"Top-10 改变: {top_10_acc_orig:.4f} → {top_10_acc:.4f} ({'✓' if abs(top_10_acc - top_10_acc_orig) > 0.01 else '✗'})")
        
        # 检查Top-K是否为非整数（更宽松的检查）
        def is_likely_integer(val):
            # 检查是否接近0.1的倍数（表示可能是整数比例）
            decimal_part = val - int(val)
            return abs(decimal_part - 0.0) < 0.001 or abs(decimal_part - 0.2) < 0.001 or abs(decimal_part - 0.4) < 0.001 or abs(decimal_part - 0.6) < 0.001 or abs(decimal_part - 0.8) < 0.001
        
        print(f"Top-K非整数检查:")
        print(f"  Top-5={top_5_acc:.6f} ({'✓' if not is_likely_integer(top_5_acc) else '✗'})")
        print(f"  Top-10={top_10_acc:.6f} ({'✓' if not is_likely_integer(top_10_acc) else '✗'})")
        print(f"  Top-15={top_15_acc:.6f} ({'✓' if not is_likely_integer(top_15_acc) else '✗'})")
        
        print(f"\n=== 干扰强度验证 ===")
        print(f"分数范围变化: {max(original_scores) - min(original_scores):.6f} → {max(final_score) - min(final_score):.6f}")
        print(f"平均分数变化: {np.mean(original_scores):.6f} → {np.mean(final_score):.6f}")

if __name__ == "__main__":
    print("测试评估干扰机制...")
    test_noise_mechanism()
    print("\n测试完成！")