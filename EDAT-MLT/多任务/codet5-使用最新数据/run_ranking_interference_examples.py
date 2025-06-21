#!/usr/bin/env python3
"""
排序随机干扰功能使用示例

展示如何在不同场景下使用新的排序随机干扰功能，包括：
1. 基础使用方法
2. 参数调节示例
3. 批量实验运行
4. 结果分析脚本
"""

import os
import subprocess
import json
import time

def run_basic_examples():
    """基础使用示例"""
    print("=== 基础使用示例 ===\n")
    
    examples = [
        {
            "name": "禁用干扰（获得真实性能）",
            "cmd": "python multi_task_evaluate.py --disable_ranking_interference",
            "description": "获得模型的真实性能，不添加任何干扰"
        },
        {
            "name": "标准干扰（推荐设置）", 
            "cmd": "python multi_task_evaluate.py --enable_ranking_interference",
            "description": "使用默认的干扰参数，模拟真实环境"
        },
        {
            "name": "轻度干扰（保守评估）",
            "cmd": "python multi_task_evaluate.py --shuffle_probability 0.05 --swap_probability 0.02 --group_shuffle_probability 0.08",
            "description": "较小的干扰强度，适用于初步验证"
        },
        {
            "name": "强干扰（压力测试）",
            "cmd": "python multi_task_evaluate.py --shuffle_probability 0.2 --swap_probability 0.1 --group_shuffle_probability 0.25",
            "description": "较大的干扰强度，测试模型鲁棒性"
        }
    ]
    
    for example in examples:
        print(f"示例: {example['name']}")
        print(f"说明: {example['description']}")
        print(f"命令: {example['cmd']}")
        print("-" * 60)

def run_parameter_study():
    """参数研究示例"""
    print("\n=== 参数调节研究示例 ===\n")
    
    # 生成参数组合
    shuffle_probs = [0.05, 0.1, 0.15, 0.2]
    swap_probs = [0.02, 0.05, 0.08, 0.1]
    group_probs = [0.08, 0.15, 0.2, 0.25]
    
    print("参数网格搜索示例:")
    print("以下命令可以用于系统性地测试不同参数组合的效果:\n")
    
    count = 0
    for shuffle in shuffle_probs:
        for swap in swap_probs:
            for group in group_probs:
                count += 1
                if count <= 8:  # 只显示前8个示例
                    cmd = (f"python multi_task_evaluate.py "
                          f"--shuffle_probability {shuffle} "
                          f"--swap_probability {swap} "
                          f"--group_shuffle_probability {group}")
                    print(f"实验 {count}: {cmd}")
    
    print(f"\n总共可以生成 {len(shuffle_probs) * len(swap_probs) * len(group_probs)} 种参数组合")
    print("建议使用脚本自动化运行并记录结果")

def generate_batch_script():
    """生成批量实验脚本"""
    print("\n=== 生成批量实验脚本 ===\n")
    
    script_content = '''#!/bin/bash
# 批量排序干扰实验脚本
# 自动运行多种配置并记录结果

echo "开始批量排序干扰实验..."
mkdir -p ranking_interference_results

# 实验配置
configs=(
    "--disable_ranking_interference"
    "--shuffle_probability 0.05 --swap_probability 0.02 --group_shuffle_probability 0.08"
    "--shuffle_probability 0.1 --swap_probability 0.05 --group_shuffle_probability 0.15" 
    "--shuffle_probability 0.2 --swap_probability 0.1 --group_shuffle_probability 0.25"
)

config_names=(
    "no_interference"
    "light_interference"
    "standard_interference"
    "heavy_interference"
)

# 运行实验
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    name="${config_names[$i]}"
    
    echo "运行实验: $name"
    echo "参数: $config"
    
    # 运行评估并重定向输出
    python multi_task_evaluate.py $config > "ranking_interference_results/${name}.log" 2>&1
    
    echo "完成实验: $name"
    echo "结果保存到: ranking_interference_results/${name}.log"
    echo "---"
done

echo "所有实验完成！"
echo "结果文件位于 ranking_interference_results/ 目录"
'''
    
    script_filename = "batch_ranking_interference.sh"
    
    print(f"生成批量实验脚本: {script_filename}")
    print("脚本内容预览:")
    print("-" * 40)
    print(script_content[:500] + "...")
    print("-" * 40)
    
    # 实际创建脚本文件
    try:
        with open(script_filename, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 设置执行权限（Unix系统）
        os.chmod(script_filename, 0o755)
        
        print(f"✓ 脚本已创建: {script_filename}")
        print(f"使用方法: bash {script_filename}")
        
    except Exception as e:
        print(f"✗ 创建脚本失败: {e}")

def generate_analysis_script():
    """生成结果分析脚本"""
    print("\n=== 生成结果分析脚本 ===\n")
    
    analysis_content = '''#!/usr/bin/env python3
"""
排序干扰实验结果分析脚本
"""

import re
import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_log_file(log_file):
    """解析日志文件，提取评估指标"""
    if not os.path.exists(log_file):
        return None
    
    metrics = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取IFA
    ifa_match = re.search(r'第一个漏洞行排名: (\\d+), IFA: (\\d+)', content)
    if ifa_match:
        metrics['ifa'] = int(ifa_match.group(2))
    
    # 提取Top-K准确率
    top_5_match = re.search(r'Top-5: 前5行中有(\\d+)个漏洞行, 准确率=(\\d+\\.\\d+)', content)
    if top_5_match:
        metrics['top_5'] = float(top_5_match.group(2))
    
    top_10_match = re.search(r'Top-10: 前10行中有(\\d+)个漏洞行, 准确率=(\\d+\\.\\d+)', content)
    if top_10_match:
        metrics['top_10'] = float(top_10_match.group(2))
    
    # 提取F1分数
    f1_match = re.search(r'F1分数: (\\d+\\.\\d+)', content)
    if f1_match:
        metrics['f1'] = float(f1_match.group(1))
    
    return metrics

def analyze_results():
    """分析实验结果"""
    results_dir = "ranking_interference_results"
    
    if not os.path.exists(results_dir):
        print("结果目录不存在，请先运行批量实验")
        return
    
    # 配置映射
    config_names = {
        "no_interference": "无干扰",
        "light_interference": "轻度干扰", 
        "standard_interference": "标准干扰",
        "heavy_interference": "强干扰"
    }
    
    results = []
    
    for config_file, config_name in config_names.items():
        log_file = os.path.join(results_dir, f"{config_file}.log")
        metrics = parse_log_file(log_file)
        
        if metrics:
            metrics['config'] = config_name
            results.append(metrics)
            print(f"✓ 解析 {config_name}: {metrics}")
        else:
            print(f"✗ 无法解析 {config_name}")
    
    if not results:
        print("没有有效的结果数据")
        return
    
    # 创建对比表格
    df = pd.DataFrame(results)
    print("\\n=== 实验结果对比 ===")
    print(df.to_string(index=False))
    
    # 保存结果
    df.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
    print(f"\\n结果已保存到: {os.path.join(results_dir, 'summary.csv')}")

if __name__ == "__main__":
    analyze_results()
'''
    
    analysis_filename = "analyze_ranking_interference.py"
    
    print(f"生成结果分析脚本: {analysis_filename}")
    
    try:
        with open(analysis_filename, 'w', encoding='utf-8') as f:
            f.write(analysis_content)
        
        print(f"✓ 分析脚本已创建: {analysis_filename}")
        print(f"使用方法: python {analysis_filename}")
        
    except Exception as e:
        print(f"✗ 创建分析脚本失败: {e}")

def show_feature_combination_examples():
    """展示特征控制与排序干扰的组合使用"""
    print("\n=== 特征控制与排序干扰组合示例 ===\n")
    
    combinations = [
        {
            "name": "限制特征 + 标准干扰",
            "cmd": "python multi_task_evaluate.py --use_limited_features --num_features_to_use 15 --enable_ranking_interference",
            "description": "使用15个特征并添加标准干扰，测试减少特征对性能的影响"
        },
        {
            "name": "随机特征 + 轻度干扰", 
            "cmd": "python multi_task_evaluate.py --use_limited_features --num_features_to_use 10 --feature_selection_strategy random --shuffle_probability 0.05",
            "description": "随机选择10个特征并添加轻度干扰"
        },
        {
            "name": "重要特征 + 无干扰",
            "cmd": "python multi_task_evaluate.py --use_limited_features --num_features_to_use 20 --feature_selection_strategy important --disable_ranking_interference",
            "description": "使用20个重要特征且不添加干扰，获得最佳性能基线"
        }
    ]
    
    for combo in combinations:
        print(f"组合: {combo['name']}")
        print(f"说明: {combo['description']}")
        print(f"命令: {combo['cmd']}")
        print("-" * 80)

def main():
    """主函数"""
    print("排序随机干扰功能使用示例")
    print("=" * 60)
    
    # 基础示例
    run_basic_examples()
    
    # 参数研究
    run_parameter_study()
    
    # 生成批量脚本
    generate_batch_script()
    
    # 生成分析脚本
    generate_analysis_script()
    
    # 特征组合示例
    show_feature_combination_examples()
    
    print("\\n" + "=" * 60)
    print("使用示例完成！")
    print("\\n推荐的实验流程:")
    print("1. 运行基础示例，了解功能")
    print("2. 使用批量脚本进行系统性实验")
    print("3. 使用分析脚本处理结果") 
    print("4. 根据需要调节参数并重复实验")

if __name__ == "__main__":
    main() 