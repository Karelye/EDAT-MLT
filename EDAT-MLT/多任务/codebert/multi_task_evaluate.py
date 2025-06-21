import os
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import random

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
    output_dir = "multi_task_alternate_output"
    model_path = os.path.join(output_dir, "best_multi_task_model.pt")  # 使用最佳模型

    # 模型参数 - 需要与训练时一致
    pretrained_model_path = r"E:\models\graphcodebert-base"
    line_num_labels = 2
    expert_num = 6  # 与训练时一致
    expert_dim = 768  # 与训练时一致
    expert_feature_dim = 0  # 已移除提交特征和专家特征

    max_length = 512
    max_codeline_length = 256
    max_codeline_token_length = 64
    batch_size = 16

    # 分类任务数据路径
    classification_train_path = r"C:\Users\Admin\Desktop\csy\最新实验\bigvul\91分\train.jsonl"
    classification_test_path = r"C:\Users\Admin\Desktop\csy\最新实验\bigvul\91分\test.jsonl"

    # 行级任务数据路径
    line_level_test_path = r"C:\Users\Admin\Desktop\csy\最新实验\bigvul\91分\test_line_level.jsonl"

    # 初始设置一个默认值，会在运行时根据训练数据更新
    class_num_labels = 86  # 与训练模型时的类别数保持一致


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 评估分类任务
def evaluate_classification(model, dataloader, config):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="评估分类任务"):
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

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': classification_rep,
        'class_accuracies': class_accuracies
    }


# 新增自定义指标计算函数
def calculate_line_level_metrics(pred_scores, true_labels):
    """计算行级任务的评估指标
    
    参数:
        pred_scores: 预测分数列表
        true_labels: 真实标签列表(0或1)
        
    返回:
        dict: 包含各项指标的字典
    """
    # 确保输入长度一致
    assert len(pred_scores) == len(true_labels), "预测分数和真实标签的长度不匹配"
    
    # 将预测分数和标签转换为numpy数组
    pred_scores = np.array(pred_scores)
    true_labels = np.array(true_labels)
    
    # 统计真实漏洞的总行数
    total_lines = len(true_labels)
    total_buggy_lines = np.sum(true_labels)
    buggy_line_percentage = total_buggy_lines / total_lines if total_lines > 0 else 0
    
    # 定义漏洞预测阈值(通常为0.5)
    threshold = 0.5
    pred_labels = (pred_scores >= threshold).astype(int)
    predicted_buggy_lines = np.sum(pred_labels)
    
    # 按预测分数从高到低排序
    sorted_indices = np.argsort(pred_scores)[::-1]
    sorted_labels = true_labels[sorted_indices]
    sorted_scores = pred_scores[sorted_indices]
    
    # 计算前N行中的漏洞行数
    top_ns = [5, 10, 20, 50, 100]
    top_n_stats = {}
    for n in top_ns:
        if n <= total_lines:
            top_n_labels = sorted_labels[:n]
            top_n_buggy = np.sum(top_n_labels)
            top_n_stats[f"top_{n}"] = {
                "buggy_count": int(top_n_buggy),
                "accuracy": float(top_n_buggy / n),
                "scores_range": (float(sorted_scores[min(n-1, len(sorted_scores)-1)]), float(sorted_scores[0])),  # (min, max)
            }
    
    # 初始False Alarm (IFA): 在第一个缺陷被发现之前需要检查的非缺陷行数
    if total_buggy_lines > 0 and 1 in sorted_labels:
        first_bug_idx = np.where(sorted_labels == 1)[0][0]
        ifa = first_bug_idx
    else:
        ifa = total_lines  # 如果没有缺陷行，IFA设为总行数
    
    # Top-20% LOC Recall: 检查20%代码行能发现的缺陷百分比
    loc_20_percent = int(total_lines * 0.2)
    if loc_20_percent > 0 and total_buggy_lines > 0:
        top_20_recall = np.sum(sorted_labels[:loc_20_percent]) / total_buggy_lines
    else:
        top_20_recall = 0.0
    
    # Effort@20%Recall: 找到20%缺陷需要检查的代码行百分比
    recall_20_percent = int(total_buggy_lines * 0.2)
    if recall_20_percent > 0:
        # 计算累积缺陷数
        cum_bugs = np.cumsum(sorted_labels)
        # 找到达到20%缺陷召回率的位置
        if cum_bugs[-1] >= recall_20_percent:
            idx_20_percent_recall = np.where(cum_bugs >= recall_20_percent)[0][0]
            effort_at_20_percent_recall = (idx_20_percent_recall + 1) / total_lines
        else:
            effort_at_20_percent_recall = 1.0  # 如果检查所有行也无法达到20%召回率
    else:
        effort_at_20_percent_recall = 0.0
    
    # Top-10 Accuracy
    top_10 = min(10, total_lines)
    top_10_acc = np.sum(sorted_labels[:top_10]) / top_10 if top_10 > 0 else 0.0
    
    # Top-5 Accuracy
    top_5 = min(5, total_lines)
    top_5_acc = np.sum(sorted_labels[:top_5]) / top_5 if top_5 > 0 else 0.0
    
    # 返回所有指标
    return {
        'ifa': ifa,
        'top_20_percent_loc_recall': top_20_recall,
        'effort_at_20_percent_recall': effort_at_20_percent_recall,
        'top_10_acc': top_10_acc,
        'top_5_acc': top_5_acc,
        'total_lines': total_lines,
        'total_buggy_lines': int(total_buggy_lines),
        'buggy_line_percentage': buggy_line_percentage,
        'predicted_buggy_lines': int(predicted_buggy_lines), 
        'top_n_stats': top_n_stats
    }


# 评估行级任务
def evaluate_line_level(model, dataloader, config):
    model.eval()
    all_line_scores = []
    all_labels = []
    file_count = 0

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="评估行级任务"):
            _, line_level_batch = batch_data

            if line_level_batch is None:
                continue
                
            file_count += 1

            line_ids = line_level_batch['line_ids'].to(config.device)
            attention_mask = line_level_batch['attention_mask'].to(config.device)
            line_labels = line_level_batch['line_label'].to(config.device)

            # 移除对提交特征和专家特征的使用
            logits = model(
                task="line_level",
                line_ids=line_ids,
                attention_mask=attention_mask
            )

            # 获取每行的预测分数
            line_scores = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy().flatten().tolist()
            labels = line_labels.cpu().numpy().flatten().tolist()

            all_line_scores.extend(line_scores)
            all_labels.extend(labels)

    logger.info(f"评估了 {file_count} 个批次的行级数据，共 {len(all_labels)} 行代码")
    
    # 确保标签为0或1
    all_labels = [1 if label > 0.5 else 0 for label in all_labels]
    
    # 使用自定义函数计算指标
    metrics = calculate_line_level_metrics(all_line_scores, all_labels)

    return metrics


# 主评估函数
def evaluate(config):
    set_seed()

    # 加载tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config.pretrained_model_path)

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
        expert_feature_dim=0  # 设置为0，因为我们不使用专家特征
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
        shuffle=False,
        collate_fn=collate_batch
    )

    # 加载模型
    logger.info(f"从 {config.model_path} 加载模型...")
    model = MultiTaskVulnerabilityModel(
        pretrained_model_path=config.pretrained_model_path,
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
    logger.info(f"总代码行数: {line_level_metrics['total_lines']}")
    logger.info(f"真实漏洞行数: {line_level_metrics['total_buggy_lines']} ({line_level_metrics['buggy_line_percentage']:.2%})")
    logger.info(f"预测为漏洞的行数 (阈值=0.5): {line_level_metrics['predicted_buggy_lines']}")
    logger.info(f"IFA (初始误报): {line_level_metrics['ifa']}")
    logger.info(f"Top 20% LOC召回率: {line_level_metrics['top_20_percent_loc_recall']:.4f}")
    logger.info(f"20%召回率的工作量: {line_level_metrics['effort_at_20_percent_recall']:.4f}")
    
    # 打印Top-N统计
    top_n_stats = line_level_metrics['top_n_stats']
    for n, stats in sorted(top_n_stats.items()):
        logger.info(f"{n.replace('_', '-')}准确率: {stats['accuracy']:.4f} ({stats['buggy_count']}个漏洞行/{n.split('_')[1]}行)")
        logger.info(f"  分数范围: {stats['scores_range'][1]:.4f} - {stats['scores_range'][0]:.4f}")

    return {
        'classification': classification_metrics,
        'line_level': line_level_metrics
    }


if __name__ == "__main__":
    config = EvalConfig()
    logger.info(f"设备: {config.device}")
    evaluate(config)