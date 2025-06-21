#!/usr/bin/env python3
"""
特征控制实验批量运行脚本
用于快速运行多种特征配置的实验
"""

import subprocess
import sys
import os
import time
from datetime import datetime


def run_experiment(config_name, args, output_file=None):
    """运行单个实验"""
    cmd = ["python", "multi_task_evaluate.py"] + args

    print(f"\n{'=' * 60}")
    print(f"开始实验: {config_name}")
    print(f"命令: {' '.join(cmd)}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                        text=True, check=True)
            print(f"✅ 实验完成，结果保存到: {output_file}")
        else:
            result = subprocess.run(cmd, check=True)
            print(f"✅ 实验完成")

        elapsed = time.time() - start_time
        print(f"⏱️  耗时: {elapsed:.1f} 秒")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ 实验失败 (退出码: {e.returncode})")
        print(f"⏱️  耗时: {elapsed:.1f} 秒")
        return False
    except KeyboardInterrupt:
        print(f"\n🛑 实验被用户中断")
        return False


def run_ablation_study():
    """运行消融研究实验"""
    print("🔬 开始消融研究实验...")

    # 创建结果目录
    results_dir = "feature_ablation_results"
    os.makedirs(results_dir, exist_ok=True)

    # 不同特征数量的实验配置
    experiments = [
        ("baseline_all_features", [], f"{results_dir}/baseline_all_25_features.txt"),
        ("important_15_features", ["--use_limited_features", "--num_features", "15", "--feature_strategy", "important"],
         f"{results_dir}/important_15_features.txt"),
        ("important_10_features", ["--use_limited_features", "--num_features", "10", "--feature_strategy", "important"],
         f"{results_dir}/important_10_features.txt"),
        ("important_5_features", ["--use_limited_features", "--num_features", "5", "--feature_strategy", "important"],
         f"{results_dir}/important_5_features.txt"),
        ("important_3_features", ["--use_limited_features", "--num_features", "3", "--feature_strategy", "important"],
         f"{results_dir}/important_3_features.txt"),
    ]

    success_count = 0
    for config_name, args, output_file in experiments:
        if run_experiment(config_name, args, output_file):
            success_count += 1
        else:
            print(f"⚠️  跳过后续实验...")
            break

    print(f"\n📊 消融研究完成: {success_count}/{len(experiments)} 个实验成功")
    return success_count == len(experiments)


def run_strategy_comparison():
    """运行特征选择策略比较实验"""
    print("📋 开始特征选择策略比较实验...")

    # 创建结果目录
    results_dir = "feature_strategy_results"
    os.makedirs(results_dir, exist_ok=True)

    # 不同策略的实验配置（都使用10个特征）
    experiments = [
        ("first_10_features", ["--use_limited_features", "--num_features", "10", "--feature_strategy", "first"],
         f"{results_dir}/first_10_features.txt"),
        ("random_10_features",
         ["--use_limited_features", "--num_features", "10", "--feature_strategy", "random", "--feature_seed", "42"],
         f"{results_dir}/random_10_features.txt"),
        ("important_10_features", ["--use_limited_features", "--num_features", "10", "--feature_strategy", "important"],
         f"{results_dir}/important_10_features.txt"),
    ]

    success_count = 0
    for config_name, args, output_file in experiments:
        if run_experiment(config_name, args, output_file):
            success_count += 1
        else:
            print(f"⚠️  跳过后续实验...")
            break

    print(f"\n📊 策略比较完成: {success_count}/{len(experiments)} 个实验成功")
    return success_count == len(experiments)


def run_robustness_test():
    """运行鲁棒性测试实验"""
    print("🛡️  开始鲁棒性测试实验...")

    # 创建结果目录
    results_dir = "robustness_test_results"
    os.makedirs(results_dir, exist_ok=True)

    # 模拟不同程度的特征缺失
    experiments = [
        ("severe_loss_1_feature", ["--use_limited_features", "--num_features", "1", "--feature_strategy", "important"],
         f"{results_dir}/severe_loss_1_feature.txt"),
        ("major_loss_3_features", ["--use_limited_features", "--num_features", "3", "--feature_strategy", "important"],
         f"{results_dir}/major_loss_3_features.txt"),
        ("moderate_loss_8_features",
         ["--use_limited_features", "--num_features", "8", "--feature_strategy", "random", "--feature_seed", "123"],
         f"{results_dir}/moderate_loss_8_features.txt"),
        ("minor_loss_20_features", ["--use_limited_features", "--num_features", "20", "--feature_strategy", "first"],
         f"{results_dir}/minor_loss_20_features.txt"),
    ]

    success_count = 0
    for config_name, args, output_file in experiments:
        if run_experiment(config_name, args, output_file):
            success_count += 1
        else:
            print(f"⚠️  跳过后续实验...")
            break

    print(f"\n📊 鲁棒性测试完成: {success_count}/{len(experiments)} 个实验成功")
    return success_count == len(experiments)


def quick_test():
    """快速测试特征控制功能是否正常"""
    print("⚡ 运行快速功能测试...")

    # 简单测试 - 使用5个重要特征
    args = ["--use_limited_features", "--num_features", "5", "--feature_strategy", "important", "--batch_size", "1"]

    return run_experiment("quick_test", args)


def main():
    """主函数"""
    print("🚀 特征控制实验批量运行器")
    print("=" * 60)

    if len(sys.argv) > 1:
        experiment_type = sys.argv[1].lower()
    else:
        print("请选择实验类型:")
        print("1. quick - 快速功能测试")
        print("2. ablation - 消融研究")
        print("3. strategy - 策略比较")
        print("4. robustness - 鲁棒性测试")
        print("5. all - 运行所有实验")

        choice = input("\n请输入选择 (1-5): ").strip()
        experiment_map = {
            "1": "quick",
            "2": "ablation",
            "3": "strategy",
            "4": "robustness",
            "5": "all"
        }
        experiment_type = experiment_map.get(choice, "quick")

    print(f"\n选择的实验类型: {experiment_type}")

    start_time = time.time()
    overall_success = True

    try:
        if experiment_type == "quick":
            overall_success = quick_test()

        elif experiment_type == "ablation":
            overall_success = run_ablation_study()

        elif experiment_type == "strategy":
            overall_success = run_strategy_comparison()

        elif experiment_type == "robustness":
            overall_success = run_robustness_test()

        elif experiment_type == "all":
            print("🎯 运行完整实验套件...")
            overall_success = (
                    quick_test() and
                    run_ablation_study() and
                    run_strategy_comparison() and
                    run_robustness_test()
            )

        else:
            print(f"❌ 未知的实验类型: {experiment_type}")
            overall_success = False

    except KeyboardInterrupt:
        print(f"\n🛑 实验被用户中断")
        overall_success = False

    total_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"🏁 实验套件完成")
    print(f"⏱️  总耗时: {total_time:.1f} 秒")
    print(f"📊 结果: {'✅ 成功' if overall_success else '❌ 失败'}")
    print(f"{'=' * 60}")

    return 0 if overall_success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)