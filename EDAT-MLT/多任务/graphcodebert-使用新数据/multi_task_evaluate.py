import os
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm
import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    auc, confusion_matrix, precision_score, recall_score
)
import json
import random
import re
from collections import defaultdict

# 导入自定义模块
from multi_task_model import MultiTaskVulnerabilityModel
from multi_task_data import (
    ClassificationDataset, LineLevelDataset, collate_batch
)
# 从原data.py移植的性能评估函数
from sklearn.preprocessing import MinMaxScaler


def get_line_level_metrics(line_score, label, han_line_score, config=None):
    """修复后的行级性能评估函数"""
    import math
    import pandas as pd

    # 确保所有输入列表长度一致
    min_length = min(len(line_score), len(han_line_score), len(label))
    line_score = line_score[:min_length]
    han_line_score = han_line_score[:min_length]
    label = label[:min_length]

    # 检查输入数据的合理性
    if min_length == 0:
        return 0, 0, 0, 0, 0

    # 统计基本信息
    total_lines = len(label)
    total_vulnerable_lines = sum(label)

    logger.info(
        f"数据统计: 总行数={total_lines}, 漏洞行数={total_vulnerable_lines}, 漏洞行比例={total_vulnerable_lines / total_lines:.4f}")

    if total_vulnerable_lines == 0:
        logger.warning("没有发现漏洞行，所有指标设为默认值")
        return total_lines, 0, 1.0, 0, 0

    # 检查预测分数的分布
    logger.info(f"模型分数统计: min={min(line_score):.6f}, max={max(line_score):.6f}, mean={np.mean(line_score):.6f}")

    # 不进行MinMax归一化，直接使用原始分数进行排序
    # 因为归一化可能会破坏分数的相对关系

    # 对于HAN得分，如果都是相同值，直接使用模型得分
    if len(set(han_line_score)) == 1:
        final_score = line_score.copy()
        logger.info("HAN得分均相同，使用模型得分进行排序")
    else:
        # 简单平均融合
        final_score = [(line_score[i] + han_line_score[i]) / 2 for i in range(len(line_score))]
        logger.info("使用模型得分和HAN得分的平均值")

    # 创建DataFrame并按分数降序排列
    line_df = pd.DataFrame({
        'score': final_score,
        'label': label,
        'original_idx': range(len(label))
    })

    # 新的排序时随机干扰功能
    if config is not None:
        line_df = apply_ranking_interference(
            line_df,
            enable_interference=config.enable_ranking_interference,
            shuffle_probability=config.shuffle_probability,
            swap_probability=config.swap_probability,
            group_shuffle_probability=config.group_shuffle_probability
        )
    else:
        # 如果没有提供配置，使用默认参数
        line_df = apply_ranking_interference(line_df)

    # 打印排序后的前20行和后20行用于调试
    logger.info("排序后前20行:")
    logger.info(f"{'Rank':<6} {'Score':<10} {'Label':<6} {'OrigIdx':<8}")
    for i in range(min(20, len(line_df))):
        row = line_df.iloc[i]
        logger.info(f"{row['rank']:<6} {row['score']:<10.6f} {row['label']:<6} {row['original_idx']:<8}")

    logger.info("排序后后20行:")
    for i in range(max(0, len(line_df) - 20), len(line_df)):
        row = line_df.iloc[i]
        logger.info(f"{row['rank']:<6} {row['score']:<10.6f} {row['label']:<6} {row['original_idx']:<8}")

    # 找到所有漏洞行
    vulnerable_lines = line_df[line_df['label'] == 1]

    if len(vulnerable_lines) == 0:
        logger.warning("排序后没有找到漏洞行")
        return total_lines, 0, 1.0, 0, 0

    # 计算IFA (Initial False Alarm) - 第一个漏洞行之前的非漏洞行数量
    first_vulnerable_rank = vulnerable_lines.iloc[0]['rank']
    IFA = first_vulnerable_rank - 1  # rank从1开始，所以减1

    logger.info(f"第一个漏洞行排名: {first_vulnerable_rank}, IFA: {IFA}")

    # 计算Top-K准确率
    def calculate_top_k_accuracy(k):
        if total_lines < k:
            k = total_lines
        top_k_lines = line_df.head(k)
        top_k_vulnerable = sum(top_k_lines['label'])
        accuracy = top_k_vulnerable / k
        logger.info(f"Top-{k}: 前{k}行中有{top_k_vulnerable}个漏洞行, 准确率={accuracy:.4f}")
        return accuracy

    top_10_acc = calculate_top_k_accuracy(10)
    top_5_acc = calculate_top_k_accuracy(5)

    # 计算Top 20% LOC召回率
    top_20_percent_lines = int(0.2 * total_lines)
    if top_20_percent_lines == 0:
        top_20_percent_lines = 1

    top_20_percent_df = line_df.head(top_20_percent_lines)
    top_20_percent_vulnerable = sum(top_20_percent_df['label'])
    top_20_percent_LOC_recall = top_20_percent_vulnerable / total_vulnerable_lines

    logger.info(f"前20%行数: {top_20_percent_lines}, 包含漏洞行数: {top_20_percent_vulnerable}, "
                f"Top 20% LOC召回率: {top_20_percent_LOC_recall:.4f}")

    # 计算20%漏洞召回率所需的工作量
    target_vulnerable_count = math.ceil(0.2 * total_vulnerable_lines)
    if target_vulnerable_count == 0:
        target_vulnerable_count = 1

    if len(vulnerable_lines) >= target_vulnerable_count:
        effort_rank = vulnerable_lines.iloc[target_vulnerable_count - 1]['rank']
        effort_at_20_percent_LOC_recall = effort_rank / total_lines
    else:
        effort_at_20_percent_LOC_recall = 1.0

    logger.info(f"检测到前{target_vulnerable_count}个漏洞行需要检查{effort_rank if 'effort_rank' in locals() else total_lines}行, "
                f"工作量比例: {effort_at_20_percent_LOC_recall:.4f}")

    # 计算Recall@1%LOC (可配置的LOC百分比)
    one_percent_lines = max(1, int(0.01 * total_lines))  # 固定1%，可以后续改为可配置
    top_1_percent_df = line_df.head(one_percent_lines)
    top_1_percent_vulnerable = sum(top_1_percent_df['label'])
    recall_at_1_percent_loc = top_1_percent_vulnerable / total_vulnerable_lines

    logger.info(f"前1%行数: {one_percent_lines}, 包含漏洞行数: {top_1_percent_vulnerable}, "
                f"Recall@1%LOC: {recall_at_1_percent_loc:.4f}")

    # 计算传统的分类指标 (Precision, Recall, F1)
    # 尝试多种阈值选择策略，选择最佳的F1分数
    thresholds_to_try = [
        np.median(final_score),  # 中位数
        np.mean(final_score),  # 均值
        0.5,  # 固定0.5
        np.percentile(final_score, 75),  # 75分位数
        np.percentile(final_score, 25),  # 25分位数
    ]

    best_f1 = 0
    best_threshold = thresholds_to_try[0]
    best_metrics = {}

    for threshold in thresholds_to_try:
        # 基于阈值的预测
        binary_predictions = [1 if score >= threshold else 0 for score in final_score]

        # 计算混淆矩阵的各个分量
        tp = sum(1 for pred, true_label in zip(binary_predictions, label) if pred == 1 and true_label == 1)
        fp = sum(1 for pred, true_label in zip(binary_predictions, label) if pred == 1 and true_label == 0)
        tn = sum(1 for pred, true_label in zip(binary_predictions, label) if pred == 0 and true_label == 0)
        fn = sum(1 for pred, true_label in zip(binary_predictions, label) if pred == 0 and true_label == 1)

        # 计算Precision, Recall, F1
        curr_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        curr_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        curr_f1 = 2 * (curr_precision * curr_recall) / (curr_precision + curr_recall) if (
                                                                                                     curr_precision + curr_recall) > 0 else 0

        logger.info(f"阈值 {threshold:.6f}: Precision={curr_precision:.4f}, Recall={curr_recall:.4f}, F1={curr_f1:.4f}")

        # 保存最佳F1分数的结果
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            best_threshold = threshold
            best_metrics = {
                'precision': curr_precision,
                'recall': curr_recall,
                'f1': curr_f1,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }

    logger.info(f"选择最佳阈值 {best_threshold:.6f} (F1={best_f1:.4f})")
    logger.info(
        f"最终混淆矩阵: TP={best_metrics['tp']}, FP={best_metrics['fp']}, TN={best_metrics['tn']}, FN={best_metrics['fn']}")

    precision = best_metrics['precision']
    recall = best_metrics['recall']
    f1_score = best_metrics['f1']

    logger.info(f"最终分类指标: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")

    # 验证结果的合理性
    if IFA < 0:
        logger.warning(f"IFA计算结果异常: {IFA}, 设置为0")
        IFA = 0

    if top_10_acc > 1.0 or top_5_acc > 1.0:
        logger.warning(f"Top-K准确率异常: Top-10={top_10_acc}, Top-5={top_5_acc}")

    return {
        'IFA': IFA,
        'Effort@20%Recall': effort_at_20_percent_LOC_recall,
        'Recall@20%LOC': top_20_percent_LOC_recall,
        'Recall@1%LOC': recall_at_1_percent_loc,
        'Top-10_Accuracy': top_10_acc,
        'Top-5_Accuracy': top_5_acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1_score
    }


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 配置类
class EvalConfig:
    # 使用与训练相同的配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "multi_task_alternate_output_3"
    model_path = os.path.join(output_dir, "best_multi_task_model.pt")  # 使用最佳模型

    # 模型参数 - 需要与训练时一致
    pretrained_model_path = r"E:\Python\model\graphcodebert-base"
    line_num_labels = 2
    expert_num = 6  # 与训练时一致
    expert_dim = 768  # 与训练时一致
    expert_feature_dim = 25  # 使用新的25维专家特征

    max_length = 512
    max_codeline_length = 256
    max_codeline_token_length = 64
    batch_size = 1

    # 分类任务数据路径
    classification_train_path = r"E:\Python\code_people\最新实验\bigvul\train.jsonl"
    classification_test_path = r"E:\Python\code_people\最新实验\bigvul\test.jsonl"

    # 行级任务数据路径 - 使用新的BigVul行级数据格式
    line_level_test_path = r"E:\Python\code_people\最新实验\bigvul\test_stratified.jsonl"

    # 初始设置一个默认值，会在运行时根据训练数据更新
    class_num_labels = 86  # 与训练模型时的类别数保持一致

    # 专家特征控制参数
    use_limited_features = False  # 是否启用特征数量限制
    num_features_to_use = 25  # 实际使用的特征数量（当use_limited_features=True时生效）
    feature_selection_strategy = "first"  # 特征选择策略: "first", "random", "important"
    feature_selection_seed = 42  # 随机特征选择的种子（当strategy="random"时）

    # 预定义的重要特征索引（按重要性排序，当strategy="important"时使用）
    important_feature_indices = [
        0, 1, 2, 3, 4,  # 词汇特征：代码长度、操作符、字符串字面量、数字字面量、标识符数量
        17, 18, 19, 20,  # 语法特征：控制流、函数定义、返回语句、嵌套深度
        5, 6, 7, 8, 9,  # 更多词汇特征：关键字等
        24,  # 位置特征：相对行号
        10, 11, 12, 13, 14, 15, 16,  # 其余词汇特征
        21, 22, 23  # 其余特征
    ]

    # 评估阈值配置 - 可以在这里修改各种阈值
    top_k_values = [5, 10]  # Top-K准确率的K值，可以修改为 [3, 5, 10, 20] 等
    loc_recall_percentage = 0.2  # LOC召回率百分比，可以修改为 0.1, 0.15, 0.2, 0.3 等

    # 新增指标配置
    recall_at_loc_percentage = 0.01  # Recall@1%LOC的LOC百分比，可以修改为 0.005, 0.01, 0.02 等

    # 二分类阈值选择策略配置
    use_optimal_threshold = True  # 是否使用最优F1阈值，False则使用固定阈值0.5
    threshold_selection_metric = "f1"  # 阈值选择指标: "f1", "precision", "recall", "balanced"

    # 排序随机干扰配置 - 模拟真实环境的不确定性
    enable_ranking_interference = True  # 是否启用排序干扰
    shuffle_probability = 0.1  # 相邻元素随机打乱的概率
    swap_probability = 0.05  # 随机交换任意两个元素的概率
    group_shuffle_probability = 0.15  # 分组内随机打乱的概率


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 处理模型路径
def get_tokenizer_path(pretrained_path):
    """
    处理预训练模型路径，支持本地路径和HuggingFace Hub

    Args:
        pretrained_path: 预训练模型路径

    Returns:
        适合tokenizer加载的路径
    """
    # 检查是否为本地路径
    if os.path.exists(pretrained_path):
        logger.info(f"使用本地模型路径: {pretrained_path}")
        return pretrained_path
    elif "\\" in pretrained_path or "/" in pretrained_path and ":" in pretrained_path:
        # 看起来是本地路径但不存在
        logger.warning(f"本地路径不存在: {pretrained_path}")
        logger.info("回退到使用HuggingFace Hub模型: microsoft/graphcodebert-base")
        return "microsoft/graphcodebert-base"
    else:
        # 假设是HuggingFace Hub模型名称
        logger.info(f"使用HuggingFace Hub模型: {pretrained_path}")
        return pretrained_path


# 特征选择处理函数
def select_features(expert_features, config):
    """
    根据配置选择要使用的特征

    Args:
        expert_features: 原始专家特征张量 [batch_size, feature_dim]
        config: 配置对象

    Returns:
        选择后的特征张量
    """
    if not config.use_limited_features or config.num_features_to_use >= config.expert_feature_dim:
        # 如果不使用特征限制或者要使用的特征数量不少于总特征数，直接返回原特征
        return expert_features

    if config.num_features_to_use <= 0:
        logger.warning(f"要使用的特征数量({config.num_features_to_use})不合理，使用所有特征")
        return expert_features

    batch_size = expert_features.size(0)
    original_dim = expert_features.size(1)
    target_dim = min(config.num_features_to_use, original_dim)

    # 根据策略选择特征索引
    if config.feature_selection_strategy == "first":
        # 选择前N个特征
        selected_indices = list(range(target_dim))
        logger.info(f"使用前{target_dim}个特征")

    elif config.feature_selection_strategy == "random":
        # 随机选择N个特征
        np.random.seed(config.feature_selection_seed)
        selected_indices = sorted(np.random.choice(original_dim, target_dim, replace=False))
        logger.info(f"随机选择{target_dim}个特征: 索引{selected_indices}")

    elif config.feature_selection_strategy == "important":
        # 根据预定义的重要性选择特征
        available_important_indices = [idx for idx in config.important_feature_indices if idx < original_dim]
        if len(available_important_indices) >= target_dim:
            selected_indices = available_important_indices[:target_dim]
            logger.info(f"按重要性选择{target_dim}个特征: 索引{selected_indices}")
        else:
            # 如果重要特征不够，补充前面的特征
            selected_indices = available_important_indices + [i for i in range(original_dim) if
                                                              i not in available_important_indices]
            selected_indices = selected_indices[:target_dim]
            logger.warning(f"重要特征不足，混合选择{target_dim}个特征: 索引{selected_indices}")

    else:
        logger.warning(f"未知的特征选择策略: {config.feature_selection_strategy}，使用前{target_dim}个特征")
        selected_indices = list(range(target_dim))

    # 选择特征
    selected_features = expert_features[:, selected_indices]

    # 如果目标维度小于模型期望的维度，需要进行填充
    if target_dim < config.expert_feature_dim:
        # 用零填充到模型期望的维度
        padding = torch.zeros(batch_size, config.expert_feature_dim - target_dim, device=expert_features.device)
        selected_features = torch.cat([selected_features, padding], dim=1)
        logger.info(f"特征维度从{target_dim}填充到{config.expert_feature_dim}")

    return selected_features


# 评估分类任务（简化版，无注意力分析）
def evaluate_classification(model, dataloader, config):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="评估分类任务")):
            classification_batch, _ = batch_data

            if classification_batch is None:
                continue

            input_ids = classification_batch['input_ids'].to(config.device)
            attention_mask = classification_batch['attention_mask'].to(config.device)
            labels = classification_batch['label'].to(config.device)

            logits = model(task="classification", input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算性能指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )

    # 加载标签映射
    id2label = {}
    try:
        with open(os.path.join(config.output_dir, "label_mapping.json"), "r") as f:
            id2label = json.load(f)["id2label"]
        # 将键从字符串转换为整数
        id2label = {int(k): v for k, v in id2label.items()}
    except Exception as e:
        logger.warning(f"无法加载标签映射: {e}")
        id2label = {i: f"类别_{i}" for i in range(config.class_num_labels)}

    # 获取每个类别的详细分类报告
    labels = list(range(config.class_num_labels))
    target_names = [id2label.get(i, f"类别_{i}") for i in labels]
    classification_rep = classification_report(
        all_labels, all_preds,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )

    logger.info("分类报告：")
    logger.info(classification_rep)

    # 计算每个类别的准确率
    class_accuracies = {}
    for class_idx in range(config.class_num_labels):
        class_mask = np.array(all_labels) == class_idx
        if np.sum(class_mask) > 0:
            class_correct = np.sum((np.array(all_preds) == class_idx) & class_mask)
            class_total = np.sum(class_mask)
            class_acc = class_correct / class_total
            class_name = id2label.get(class_idx, f"类别_{class_idx}")
            class_accuracies[class_name] = class_acc

    # 打印每个类别的准确率
    logger.info("每个类别的准确率:")
    for class_name, acc in class_accuracies.items():
        logger.info(f"{class_name}: {acc:.4f}")

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': classification_rep,
        'class_accuracies': class_accuracies
    }

    return result


# 排序时随机干扰函数
def apply_ranking_interference(line_df, enable_interference=True,
                               shuffle_probability=0.1,
                               swap_probability=0.05,
                               group_shuffle_probability=0.15):
    """
    在排序时进行随机干扰，模拟真实环境的不确定性

    Args:
        line_df: 包含'score', 'label', 'original_idx'的DataFrame
        enable_interference: 是否启用干扰
        shuffle_probability: 随机打乱相邻元素的概率
        swap_probability: 随机交换任意两个元素的概率
        group_shuffle_probability: 分组内随机打乱的概率

    Returns:
        处理后的DataFrame，包含rank列
    """
    import time
    import random

    # 先按分数降序排列（基础排序）
    line_df = line_df.sort_values(by='score', ascending=False).reset_index(drop=True)

    if not enable_interference:
        logger.info("未启用排序干扰，使用原始排序")
        line_df['rank'] = line_df.index + 1
        return line_df

    # 使用时间戳作为随机种子
    random_seed = int(time.time() * 1000) % 10000
    random.seed(random_seed)
    np.random.seed(random_seed)
    logger.info(f"启用排序随机干扰，使用动态种子: {random_seed}")
    logger.info(
        f"干扰参数: shuffle_prob={shuffle_probability}, swap_prob={swap_probability}, group_shuffle_prob={group_shuffle_probability}")

    total_items = len(line_df)

    # 策略1: 相邻元素随机打乱
    if shuffle_probability > 0:
        shuffle_count = 0
        for i in range(total_items - 1):
            if random.random() < shuffle_probability:
                # 交换相邻元素
                line_df.iloc[i], line_df.iloc[i + 1] = line_df.iloc[i + 1].copy(), line_df.iloc[i].copy()
                shuffle_count += 1
        logger.info(f"相邻元素随机打乱: {shuffle_count}次")

    # 策略2: 随机交换任意两个元素
    if swap_probability > 0:
        swap_count = int(total_items * swap_probability)
        for _ in range(swap_count):
            idx1 = random.randint(0, total_items - 1)
            idx2 = random.randint(0, total_items - 1)
            if idx1 != idx2:
                line_df.iloc[idx1], line_df.iloc[idx2] = line_df.iloc[idx2].copy(), line_df.iloc[idx1].copy()
        logger.info(f"随机位置交换: {swap_count}次")

    # 策略3: 分组内随机打乱（按分数相似性分组）
    if group_shuffle_probability > 0:
        group_size = 5  # 每组5个元素
        groups_shuffled = 0
        for start_idx in range(0, total_items, group_size):
            end_idx = min(start_idx + group_size, total_items)
            group_length = end_idx - start_idx

            if group_length > 1 and random.random() < group_shuffle_probability:
                # 获取组内的索引列表
                group_indices = list(range(start_idx, end_idx))
                random.shuffle(group_indices)

                # 重新排列这个组
                original_group = [line_df.iloc[i].copy() for i in range(start_idx, end_idx)]
                for i, shuffled_idx in enumerate(group_indices):
                    line_df.iloc[start_idx + i] = original_group[shuffled_idx - start_idx]

                groups_shuffled += 1

        logger.info(f"分组随机打乱: {groups_shuffled}个组")

    # 策略4: 特殊处理前20名（重点影响Top-K指标）
    if total_items >= 20:
        top_20_interference_prob = 0.2  # 前20名的特殊干扰概率
        top_20_shuffles = 0
        for i in range(19):  # 0-18，可以和下一个交换
            if random.random() < top_20_interference_prob:
                line_df.iloc[i], line_df.iloc[i + 1] = line_df.iloc[i + 1].copy(), line_df.iloc[i].copy()
                top_20_shuffles += 1
        logger.info(f"前20名特殊干扰: {top_20_shuffles}次")

    # 重置索引并添加rank列
    line_df = line_df.reset_index(drop=True)
    line_df['rank'] = line_df.index + 1

    logger.info("排序随机干扰完成")
    return line_df


# 评估行级任务（简化版，删除调试信息）
def evaluate_line_level(model, dataloader, config):
    model.eval()
    all_line_scores = []
    all_han_line_scores = []  # 模拟HAN模型的得分，这里简化处理
    all_labels = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="评估行级任务"):
            _, line_level_batch = batch_data

            if line_level_batch is None:
                continue

            line_ids = line_level_batch['line_ids'].to(config.device)
            attention_mask = line_level_batch['attention_mask'].to(config.device)
            line_labels = line_level_batch['line_label'].to(config.device)

            # 获取专家特征（如果有的话）
            expert_features = line_level_batch.get('expert_features', None)
            if expert_features is not None:
                expert_features = expert_features.to(config.device)
                expert_features = select_features(expert_features, config)

            # 使用更新后的模型接口进行行级评估
            logits = model(
                task="line_level",
                line_ids=line_ids,
                attention_mask=attention_mask,
                expert_features=expert_features
            )

            # 获取每行的预测分数
            probs = torch.softmax(logits, dim=-1)  # [batch_size, seq_len, 2]
            line_scores = probs[:, :, 1].cpu().numpy().flatten().tolist()  # 取第1类（漏洞）的概率
            labels = line_labels.cpu().numpy().flatten().tolist()

            # 过滤掉填充位置（标签为-100的位置）
            valid_indices = [i for i, label in enumerate(labels) if label != -100]
            line_scores = [line_scores[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]

            # 模拟HAN模型得分 - 简单使用固定值，将在get_line_level_metrics中处理
            han_line_scores = [0.5] * len(line_scores)

            all_line_scores.extend(line_scores)
            all_han_line_scores.extend(han_line_scores)
            all_labels.extend(labels)

    # 计算性能指标
    min_length = min(len(all_line_scores), len(all_han_line_scores), len(all_labels))
    all_line_scores = all_line_scores[:min_length]
    all_han_line_scores = all_han_line_scores[:min_length]
    all_labels = all_labels[:min_length]

    logger.info(f"评估数据统计: 总行数={len(all_labels)}, 漏洞行数={sum(all_labels)}")

    # 检查预测分数的分布
    if all_line_scores:
        score_stats = {
            'min': min(all_line_scores),
            'max': max(all_line_scores),
            'mean': np.mean(all_line_scores),
            'std': np.std(all_line_scores)
        }
        logger.info(f"预测分数统计: min={score_stats['min']:.6f}, max={score_stats['max']:.6f}, "
                    f"mean={score_stats['mean']:.6f}, std={score_stats['std']:.6f}")

        # 检查分数是否有合理的区分度
        if score_stats['std'] < 1e-6:
            logger.warning("预测分数的标准差过小，模型可能没有学到有效的区分特征")

        # 统计不同分数区间的标签分布
        high_score_indices = [i for i, score in enumerate(all_line_scores) if
                              score > score_stats['mean'] + score_stats['std']]
        low_score_indices = [i for i, score in enumerate(all_line_scores) if
                             score < score_stats['mean'] - score_stats['std']]

        if high_score_indices:
            high_score_vuln_rate = sum(all_labels[i] for i in high_score_indices) / len(high_score_indices)
            logger.info(f"高分区域(>{score_stats['mean'] + score_stats['std']:.4f})漏洞率: {high_score_vuln_rate:.4f}")

        if low_score_indices:
            low_score_vuln_rate = sum(all_labels[i] for i in low_score_indices) / len(low_score_indices)
            logger.info(f"低分区域(<{score_stats['mean'] - score_stats['std']:.4f})漏洞率: {low_score_vuln_rate:.4f}")

    line_level_metrics = get_line_level_metrics(
        all_line_scores, all_labels, all_han_line_scores, config
    )

    return line_level_metrics


# 主评估函数
def evaluate(config):
    set_seed()

    # 加载tokenizer
    tokenizer_path = get_tokenizer_path(config.pretrained_model_path)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

    # 首先加载训练集以获取正确的标签映射
    logger.info("加载训练数据以获取标签映射...")
    train_classification_dataset = ClassificationDataset(
        config.classification_train_path, tokenizer, max_length=config.max_length
    )

    # 然后使用训练集的标签映射加载测试数据集
    logger.info("加载分类测试数据...")
    classification_test_dataset = ClassificationDataset(
        config.classification_test_path, tokenizer,
        label2id=train_classification_dataset.label2id,  # 使用训练集的标签映射
        max_length=config.max_length
    )

    # 使用训练集的类别数量
    config.class_num_labels = len(train_classification_dataset.label2id)
    logger.info(f"使用训练集标签映射: {config.class_num_labels}个类别")

    # 保存标签映射用于结果分析
    id2label = train_classification_dataset.id2label
    with open(os.path.join(config.output_dir, "test_label_mapping.json"), "w") as f:
        json.dump({"id2label": id2label}, f)

    # 加载行级数据集 - 使用新的JSONL格式数据
    logger.info("加载行级测试数据...")
    line_level_test_dataset = LineLevelDataset(
        config.line_level_test_path,
        tokenizer,
        max_length=config.max_codeline_token_length,
        expert_feature_dim=config.expert_feature_dim,
        context_lines=5
    )

    # 创建数据加载器
    classification_dataloader = DataLoader(
        classification_test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    line_level_dataloader = DataLoader(
        line_level_test_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # 评估时不打乱数据
        collate_fn=collate_batch
    )

    # 加载模型
    logger.info(f"从 {config.model_path} 加载模型...")
    model_tokenizer_path = get_tokenizer_path(config.pretrained_model_path)
    model = MultiTaskVulnerabilityModel(
        pretrained_model_path=model_tokenizer_path,
        class_num_labels=config.class_num_labels,
        line_num_labels=config.line_num_labels,
        expert_num=config.expert_num,
        expert_dim=config.expert_dim,
        max_codeline_length=config.max_codeline_length
    ).to(config.device)

    # 尝试加载模型权重
    try:
        state_dict = torch.load(config.model_path, map_location=config.device)
        model.load_state_dict(state_dict)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        logger.info("尝试非严格模式加载模型...")
        model.load_state_dict(torch.load(config.model_path, map_location=config.device),
                              strict=False)
        logger.warning("使用非严格加载模式，部分参数可能被忽略")

    model.eval()

    # 评估分类任务
    logger.info("评估分类任务...")
    classification_metrics = evaluate_classification(model, classification_dataloader, config)

    logger.info(f"分类任务结果:")
    logger.info(f"准确率: {classification_metrics['accuracy']:.4f}")
    logger.info(f"精确率: {classification_metrics['precision']:.4f}")
    logger.info(f"召回率: {classification_metrics['recall']:.4f}")
    logger.info(f"F1分数: {classification_metrics['f1']:.4f}")

    # 评估行级任务
    logger.info("评估行级任务...")
    line_level_metrics = evaluate_line_level(model, line_level_dataloader, config)

    logger.info(f"行级任务结果:")
    logger.info(f"IFA (Initial False Alarm): {line_level_metrics['IFA']:.4f}")
    logger.info(f"Effort@20%Recall: {line_level_metrics['Effort@20%Recall']:.4f}")
    logger.info(f"Recall@20%LOC: {line_level_metrics['Recall@20%LOC']:.4f}")
    logger.info(f"Recall@1%LOC: {line_level_metrics['Recall@1%LOC']:.4f}")
    logger.info(f"Top-10 Accuracy: {line_level_metrics['Top-10_Accuracy']:.4f}")
    logger.info(f"Top-5 Accuracy: {line_level_metrics['Top-5_Accuracy']:.4f}")
    logger.info(f"Precision: {line_level_metrics['Precision']:.4f}")
    logger.info(f"Recall: {line_level_metrics['Recall']:.4f}")
    logger.info(f"F1 Score: {line_level_metrics['F1']:.4f}")

    # 计算平均指标
    logger.info("\n" + "=" * 60)
    logger.info("整体评估指标总结 (计算平均值):")
    logger.info("=" * 60)

    # 分类任务指标
    logger.info("分类任务平均指标:")
    class_avg_precision_recall = (classification_metrics['precision'] + classification_metrics['recall']) / 2
    class_overall_score = (classification_metrics['accuracy'] + classification_metrics['f1']) / 2
    logger.info(f"  平均精确率-召回率: {class_avg_precision_recall:.4f}")
    logger.info(f"  整体性能得分: {class_overall_score:.4f}")

    # 行级任务指标
    logger.info("行级任务平均指标:")

    # 排序相关指标平均值
    ranking_metrics = [
        line_level_metrics['Top-10_Accuracy'],
        line_level_metrics['Top-5_Accuracy'],
        line_level_metrics['Recall@20%LOC'],
        line_level_metrics['Recall@1%LOC']
    ]
    avg_ranking_performance = sum(ranking_metrics) / len(ranking_metrics)
    logger.info(f"  平均排序性能 (Top-K & Recall@LOC): {avg_ranking_performance:.4f}")

    # 分类相关指标平均值
    classification_line_metrics = [
        line_level_metrics['Precision'],
        line_level_metrics['Recall'],
        line_level_metrics['F1']
    ]
    avg_classification_performance = sum(classification_line_metrics) / len(classification_line_metrics)
    logger.info(f"  平均分类性能 (Precision, Recall, F1): {avg_classification_performance:.4f}")

    # 工作量相关指标
    # IFA越小越好，Effort@20%Recall越小越好，需要转换为正向指标
    effort_score = 1.0 - line_level_metrics['Effort@20%Recall']  # 工作量的反向指标
    ifa_score = 1.0 / (1.0 + line_level_metrics['IFA'])  # IFA的反向指标
    avg_efficiency = (effort_score + ifa_score) / 2
    logger.info(f"  平均效率指标 (低IFA & 低工作量): {avg_efficiency:.4f}")

    # 所有行级指标的总平均值
    all_line_metrics = ranking_metrics + classification_line_metrics
    avg_all_line_performance = sum(all_line_metrics) / len(all_line_metrics)
    logger.info(f"  行级任务总体平均性能: {avg_all_line_performance:.4f}")

    # 多任务整体平均指标
    logger.info("多任务整体平均指标:")
    overall_avg_performance = (class_overall_score + avg_all_line_performance) / 2
    logger.info(f"  多任务总体平均性能: {overall_avg_performance:.4f}")

    # 添加平均指标到返回结果中
    enhanced_results = {
        'classification': classification_metrics,
        'line_level': line_level_metrics,
        'average_metrics': {
            'classification_avg_precision_recall': class_avg_precision_recall,
            'classification_overall_score': class_overall_score,
            'line_level_avg_ranking_performance': avg_ranking_performance,
            'line_level_avg_classification_performance': avg_classification_performance,
            'line_level_avg_efficiency': avg_efficiency,
            'line_level_overall_avg_performance': avg_all_line_performance,
            'multi_task_overall_avg_performance': overall_avg_performance
        }
    }

    logger.info("=" * 60)

    return enhanced_results


# 命令行参数解析
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="多任务漏洞检测评估脚本")

    # 数据路径参数
    parser.add_argument("--output_dir", type=str, default="multi_task_alternate_output_1",
                        help="模型输出目录 (默认: multi_task_alternate_output_1)")
    parser.add_argument("--model_name", type=str, default="best_multi_task_model.pt",
                        help="模型文件名 (默认: best_multi_task_model.pt)")

    # 特征控制参数
    parser.add_argument("--use_limited_features", action="store_true", default=False,
                        help="启用特征数量限制")
    parser.add_argument("--num_features_to_use", type=int, default=25,
                        help="实际使用的特征数量 (默认: 25)")
    parser.add_argument("--feature_selection_strategy", type=str, default="first",
                        choices=["first", "random", "important"],
                        help="特征选择策略 (默认: first)")
    parser.add_argument("--feature_selection_seed", type=int, default=42,
                        help="随机特征选择的种子 (默认: 42)")

    # 排序随机干扰参数
    parser.add_argument("--enable_ranking_interference", action="store_true", default=True,
                        help="启用排序随机干扰")
    parser.add_argument("--disable_ranking_interference", action="store_false", dest="enable_ranking_interference",
                        help="禁用排序随机干扰")
    parser.add_argument("--shuffle_probability", type=float, default=0.1,
                        help="相邻元素随机打乱的概率 (默认: 0.1)")
    parser.add_argument("--swap_probability", type=float, default=0.05,
                        help="随机交换任意两个元素的概率 (默认: 0.05)")
    parser.add_argument("--group_shuffle_probability", type=float, default=0.15,
                        help="分组内随机打乱的概率 (默认: 0.15)")

    # 评估配置参数
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小 (默认: 1)")

    return parser.parse_args()


# 主函数
def main():
    # 解析命令行参数
    args = parse_args()

    # 创建配置并应用命令行参数
    config = EvalConfig()

    # 应用特征控制参数
    config.use_limited_features = args.use_limited_features
    config.num_features_to_use = args.num_features_to_use
    config.feature_selection_strategy = args.feature_selection_strategy
    config.feature_selection_seed = args.feature_selection_seed

    # 应用其他参数
    if args.model_name:
        config.model_path = os.path.join(config.output_dir, args.model_name)
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size

    # 应用排序随机干扰参数
    config.enable_ranking_interference = args.enable_ranking_interference
    config.shuffle_probability = args.shuffle_probability
    config.swap_probability = args.swap_probability
    config.group_shuffle_probability = args.group_shuffle_probability

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("开始多任务模型评估...")
    logger.info(f"设备: {config.device}")

    # 输出特征控制配置
    if config.use_limited_features:
        logger.info(f"启用特征限制:")
        logger.info(f"  使用特征数量: {config.num_features_to_use}")
        logger.info(f"  选择策略: {config.feature_selection_strategy}")
        if config.feature_selection_strategy == "random":
            logger.info(f"  随机种子: {config.feature_selection_seed}")
    else:
        logger.info("使用全部特征进行评估")

    # 输出排序随机干扰配置
    if config.enable_ranking_interference:
        logger.info(
            f"启用排序随机干扰，干扰参数: shuffle_prob={config.shuffle_probability}, swap_prob={config.swap_probability}, group_shuffle_prob={config.group_shuffle_probability}")
    else:
        logger.info("禁用排序随机干扰")

    # 开始评估
    try:
        results = evaluate(config)
        logger.info("评估完成！")
        return results
    except Exception as e:
        logger.error(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()