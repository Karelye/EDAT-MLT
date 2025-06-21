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
from collections import defaultdict

# 导入自定义模块
from multi_task_model import MultiTaskVulnerabilityModel
from multi_task_data import (
    ClassificationDataset, LineLevelDataset, collate_batch
)
from data import get_line_level_metrics  # 从原始代码导入性能评估函数

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
    output_dir = "multi_task_alternate_output_classification"
    model_path = os.path.join(output_dir, "multi_task_model_epoch_3.pt")  # 使用最佳模型

    # 模型参数 - 需要与训练时一致
    pretrained_model_path = r"E:\Python\model\graphcodebert-base"  # 使用HuggingFace Hub模型名称
    line_num_labels = 2
    expert_num = 6  # 与训练时一致
    expert_dim = 768  # 与训练时一致
    expert_feature_dim = 0  # 已移除提交特征和专家特征

    max_length = 512
    max_codeline_length = 256
    max_codeline_token_length = 64
    batch_size = 16

    # 分类任务数据路径
    classification_train_path = r"E:\Python\code_people\最新实验\bigvul\train.jsonl"
    classification_test_path = r"E:\Python\code_people\最新实验\bigvul\test.jsonl"

    # 行级任务数据路径
    line_level_test_path = r"E:\Python\code_people\最新实验\bigvul\test_line_level.jsonl"

    # 初始设置一个默认值，会在运行时根据训练数据更新
    class_num_labels = 86  # 与训练模型时的类别数保持一致


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


# 评估分类任务
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
        all_labels, all_preds, average='weighted', zero_division=0
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
        if np.sum(class_mask) > 0:  # 确保有该类别的样本
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


# 建立数据关联映射
def build_data_mapping(classification_dataset, line_level_dataset):
    """
    建立分类数据集和行级数据集之间的映射关系
    通过commit_id进行关联
    """
    classification_count = len(classification_dataset.examples) if hasattr(classification_dataset, 'examples') else len(
        classification_dataset)
    line_level_count = len(line_level_dataset.examples) if hasattr(line_level_dataset, 'examples') else len(
        line_level_dataset)

    logger.info(f"分类数据集包含 {classification_count} 个样本")
    logger.info(f"行级数据集包含 {line_level_count} 个样本")

    logger.info("数据映射构建完成，将通过batch处理时进行实际关联")

    return {}, {}, set()


# 融合注意力得分和模型预测
def calculate_comprehensive_line_metrics(y_true, y_scores, y_pred_binary=None):
    """
    计算全面的行级定位性能指标

    Args:
        y_true: 真实标签 (0/1)
        y_scores: 预测得分 (概率值)
        y_pred_binary: 二进制预测结果 (可选)

    Returns:
        comprehensive_metrics: 包含各种性能指标的字典
    """
    metrics = {}

    # 确保输入为numpy数组
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # 如果没有提供二进制预测，使用0.5作为阈值
    if y_pred_binary is None:
        y_pred_binary = (y_scores >= 0.5).astype(int)
    else:
        y_pred_binary = np.array(y_pred_binary)

    # 基础分类指标
    try:
        # ROC AUC
        if len(np.unique(y_true)) > 1:  # 确保有正负样本
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            # PR AUC (Average Precision)
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0

        # 基础分类性能
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred_binary, average='binary', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1

        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        metrics['true_positive'] = int(tp)
        metrics['false_positive'] = int(fp)
        metrics['true_negative'] = int(tn)
        metrics['false_negative'] = int(fn)

        # 特异性 (Specificity)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        # 敏感性 (Sensitivity) 就是召回率
        metrics['sensitivity'] = recall

    except Exception as e:
        logger.warning(f"计算基础分类指标时出错: {e}")
        metrics.update({
            'roc_auc': 0.0, 'pr_auc': 0.0, 'accuracy': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'specificity': 0.0, 'sensitivity': 0.0
        })

    # Top-K准确率
    try:
        # 按得分排序获取排名
        sorted_indices = np.argsort(y_scores)[::-1]  # 降序排列
        sorted_labels = y_true[sorted_indices]

        total_positives = np.sum(y_true)
        if total_positives > 0:
            # Top-1, Top-3, Top-5, Top-10准确率
            for k in [1, 3, 5, 10, 20]:
                if k <= len(sorted_labels):
                    top_k_positives = np.sum(sorted_labels[:k])
                    metrics[f'top_{k}_precision'] = top_k_positives / k
                    metrics[f'top_{k}_recall'] = top_k_positives / total_positives
                else:
                    metrics[f'top_{k}_precision'] = 0.0
                    metrics[f'top_{k}_recall'] = 0.0
        else:
            for k in [1, 3, 5, 10, 20]:
                metrics[f'top_{k}_precision'] = 0.0
                metrics[f'top_{k}_recall'] = 0.0

    except Exception as e:
        logger.warning(f"计算Top-K指标时出错: {e}")
        for k in [1, 3, 5, 10, 20]:
            metrics[f'top_{k}_precision'] = 0.0
            metrics[f'top_{k}_recall'] = 0.0

    # 排序性能指标
    try:
        # MRR (Mean Reciprocal Rank)
        positive_indices = np.where(y_true == 1)[0]
        if len(positive_indices) > 0:
            reciprocal_ranks = []
            for pos_idx in positive_indices:
                rank = np.sum(y_scores >= y_scores[pos_idx])  # 该样本的排名
                reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
            metrics['mrr'] = np.mean(reciprocal_ranks)
        else:
            metrics['mrr'] = 0.0

        # NDCG (Normalized Discounted Cumulative Gain)
        def dcg_at_k(scores, k):
            """计算DCG@k"""
            scores = scores[:k]
            return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

        if total_positives > 0:
            # 理想排序的DCG
            ideal_scores = np.sort(y_true)[::-1]
            for k in [5, 10, 20]:
                if k <= len(sorted_labels):
                    dcg_k = dcg_at_k(sorted_labels, k)
                    idcg_k = dcg_at_k(ideal_scores, k)
                    metrics[f'ndcg_at_{k}'] = dcg_k / idcg_k if idcg_k > 0 else 0.0
                else:
                    metrics[f'ndcg_at_{k}'] = 0.0
        else:
            for k in [5, 10, 20]:
                metrics[f'ndcg_at_{k}'] = 0.0

    except Exception as e:
        logger.warning(f"计算排序指标时出错: {e}")
        metrics['mrr'] = 0.0
        for k in [5, 10, 20]:
            metrics[f'ndcg_at_{k}'] = 0.0

    # 不同阈值下的性能
    try:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_metrics = {}
        for thresh in thresholds:
            y_pred_thresh = (y_scores >= thresh).astype(int)
            if len(np.unique(y_pred_thresh)) > 1 or np.sum(y_pred_thresh) > 0:
                prec = precision_score(y_true, y_pred_thresh, zero_division=0)
                rec = recall_score(y_true, y_pred_thresh, zero_division=0)
                threshold_metrics[f'precision_at_{thresh}'] = prec
                threshold_metrics[f'recall_at_{thresh}'] = rec
            else:
                threshold_metrics[f'precision_at_{thresh}'] = 0.0
                threshold_metrics[f'recall_at_{thresh}'] = 0.0

        metrics['threshold_metrics'] = threshold_metrics

    except Exception as e:
        logger.warning(f"计算阈值指标时出错: {e}")
        metrics['threshold_metrics'] = {}

    return metrics


# 评估行级任务
def evaluate_line_level(model, dataloader, config):
    model.eval()
    all_line_scores = []
    all_han_line_scores = []  # 模拟HAN模型的得分，这里简化处理
    all_labels = []
    file_count = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="评估行级任务")):
            _, line_level_batch = batch_data

            if line_level_batch is None:
                continue

            file_count += 1

            line_ids = line_level_batch['line_ids'].to(config.device)
            attention_mask = line_level_batch['attention_mask'].to(config.device)
            line_labels = line_level_batch['line_label'].to(config.device)

            # 使用更新后的模型接口进行行级评估
            logits = model(
                task="line_level",
                line_ids=line_ids,
                attention_mask=attention_mask
            )

            # 获取每行的预测分数
            line_scores = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy().flatten().tolist()
            labels = line_labels.cpu().numpy().flatten().tolist()

            # 模拟HAN模型得分（用于兼容原始get_line_level_metrics函数）
            han_line_scores = [0.5] * len(line_scores)

            all_line_scores.extend(line_scores)
            all_han_line_scores.extend(han_line_scores)
            all_labels.extend(labels)

    logger.info(f"评估了 {file_count} 个批次的行级数据，共 {len(all_labels)} 行代码")

    # 确保数据长度一致
    min_length = min(len(all_line_scores), len(all_han_line_scores), len(all_labels))
    all_line_scores = all_line_scores[:min_length]
    all_han_line_scores = all_han_line_scores[:min_length]
    all_labels = all_labels[:min_length]

    # 使用原始的get_line_level_metrics函数计算指标
    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = get_line_level_metrics(
        all_line_scores, all_labels, all_han_line_scores
    )

    # 计算综合的行级定位性能指标
    comprehensive_metrics = calculate_comprehensive_line_metrics(all_labels, all_line_scores)

    original_metrics = {
        'IFA': IFA,
        'top_20_percent_LOC_recall': top_20_percent_LOC_recall,
        'effort_at_20_percent_LOC_recall': effort_at_20_percent_LOC_recall,
        'top_10_acc': top_10_acc,
        'top_5_acc': top_5_acc
    }

    # 合并原始指标和综合指标
    original_metrics.update(comprehensive_metrics)

    return original_metrics


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
    # 准备分类数据路径，用于提取上下文特征
    classification_paths = [
        config.classification_train_path,
        config.classification_test_path
    ]

    line_level_test_dataset = LineLevelDataset(
        config.line_level_test_path,
        tokenizer,
        max_length=config.max_codeline_token_length,
        expert_feature_dim=0,  # 设置为0，因为我们不使用专家特征
        context_lines=3,  # 提取前后3行作为上下文
        classification_data_paths=classification_paths
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
        shuffle=True,  # 打乱测试数据，以便更客观地评估模型在行级预测任务上的性能
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

    # 建立数据映射关系
    logger.info("建立数据集间的映射关系...")
    build_data_mapping(classification_test_dataset, line_level_test_dataset)

    # 评估分类任务
    logger.info("开始评估分类任务...")
    classification_results = evaluate_classification(model, classification_dataloader, config)

    # 评估行级任务
    logger.info("开始评估行级任务...")
    line_level_results = evaluate_line_level(model, line_level_dataloader, config)

    # 打印最终结果
    logger.info("\n" + "=" * 50)
    logger.info("最终评估结果")
    logger.info("=" * 50)

    logger.info("\n--- 分类任务性能 ---")
    logger.info(f"准确率: {classification_results['accuracy']:.4f}")
    logger.info(f"加权精确率: {classification_results['precision']:.4f}")
    logger.info(f"加权召回率: {classification_results['recall']:.4f}")
    logger.info(f"加权F1分数: {classification_results['f1']:.4f}")
    logger.info("\n详细分类报告:")
    logger.info(classification_results['classification_report'])

    logger.info("\n--- 行级定位性能 ---")
    for key, value in line_level_results.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value:.4f}" if isinstance(sub_value, float) else f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    # 保存结果到文件
    results_file = os.path.join(config.output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "classification_results": classification_results,
            "line_level_results": line_level_results
        }, f, indent=4)
    logger.info(f"\n评估结果已保存到: {results_file}")


if __name__ == "__main__":
    config = EvalConfig()
    evaluate(config)