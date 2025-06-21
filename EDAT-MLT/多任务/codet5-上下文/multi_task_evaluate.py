import os
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, AutoTokenizer
from tqdm import tqdm
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import random
import re
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
    pretrained_model_path = r"E:\Python\model\codet5-base"
    line_num_labels = 2
    expert_num = 6  # 与训练时一致
    expert_dim = 768  # 与训练时一致，CodeT5-base的实际d_model=768
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


# 处理注意力权重，计算每行代码的得分
def process_attention_weights(attention_weights, input_ids, tokenizer, original_function_lines):
    """
    处理注意力权重，计算每行代码的注意力得分

    Args:
        attention_weights: 模型的注意力权重 [batch_size, num_heads, seq_len, seq_len]
        input_ids: 输入的token ids [batch_size, seq_len]
        tokenizer: 分词器
        original_function_lines: 原始函数的代码行列表

    Returns:
        line_attention_scores: 每行代码的注意力得分
    """
    batch_size, num_layers, num_heads, seq_len, _ = attention_weights.shape

    # 对所有层和头的注意力权重求平均
    avg_attention = attention_weights.mean(dim=(1, 2))  # [batch_size, seq_len, seq_len]

    # 获取CLS token对所有token的注意力（第一行）
    cls_attention = avg_attention[:, 0, :]  # [batch_size, seq_len]

    line_scores_batch = []

    for batch_idx in range(batch_size):
        tokens = tokenizer.convert_ids_to_tokens(input_ids[batch_idx])
        cls_attn = cls_attention[batch_idx].cpu().numpy()

        # 移除特殊token的注意力得分
        special_tokens = ['<s>', '</s>', '<pad>', '<unk>']
        filtered_attention = []
        filtered_tokens = []

        for i, (token, score) in enumerate(zip(tokens, cls_attn)):
            if token not in special_tokens:
                filtered_attention.append(score)
                filtered_tokens.append(token)

        # 将token级别的注意力得分聚合到行级别
        line_scores = calculate_line_attention_scores(
            filtered_tokens, filtered_attention, original_function_lines, tokenizer
        )
        line_scores_batch.append(line_scores)

    return line_scores_batch


def calculate_line_attention_scores(tokens, attention_scores, function_lines, tokenizer):
    """
    将token级别的注意力得分聚合到行级别

    Args:
        tokens: 过滤后的token列表
        attention_scores: 对应的注意力得分
        function_lines: 原始函数的代码行
        tokenizer: 分词器实例

    Returns:
        line_scores: 每行的注意力得分
    """
    if not function_lines:
        return []

    # 重建原始代码文本
    function_text = '\n'.join(function_lines)

    # 将token重新组合成文本（处理subword token）
    try:
        reconstructed_text = tokenizer.convert_tokens_to_string(tokens)
    except:
        reconstructed_text = ' '.join(tokens)

    # 简化的行分配策略：根据token在重建文本中的位置分配到对应行
    line_scores = [0.0] * len(function_lines)

    if not tokens or not attention_scores:
        return line_scores

    # 为每个token分配权重到对应的行
    current_pos = 0
    for token, score in zip(tokens, attention_scores):
        # 找到token在原始文本中的大致位置
        token_text = token.replace('Ġ', ' ').strip()  # 处理RoBERTa的特殊前缀

        if token_text:
            # 简化的行分配：根据累积的token数量估算行号
            estimated_line = min(current_pos * len(function_lines) // len(tokens), len(function_lines) - 1)
            line_scores[estimated_line] += score
            current_pos += 1

    # 归一化行级得分
    if sum(line_scores) > 0:
        line_scores = [score / sum(line_scores) for score in line_scores]

    return line_scores


# 评估分类任务（增强版，包含注意力分析）
def evaluate_classification(model, dataloader, config, return_attention=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_attention_scores = {}  # 存储每个样本的注意力得分

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="评估分类任务")):
            classification_batch, _ = batch_data

            if classification_batch is None:
                continue

            input_ids = classification_batch['input_ids'].to(config.device)
            attention_mask = classification_batch['attention_mask'].to(config.device)
            labels = classification_batch['label'].to(config.device)

            # 获取原始函数信息（如果有的话）
            original_functions = classification_batch.get('original_function', None)

            # 如果需要注意力权重，修改模型调用
            if return_attention:
                try:
                    # 临时启用输出注意力权重
                    if hasattr(model, 'codet5') and hasattr(model.codet5, 'config'):
                        original_output_attentions = getattr(model.codet5.config, 'output_attentions', False)
                        model.codet5.config.output_attentions = True

                        # 使用模型的codet5获取注意力权重
                        outputs = model.codet5(input_ids=input_ids, attention_mask=attention_mask)

                        if hasattr(outputs, 'attentions') and outputs.attentions and classification_batch.get(
                                'original_function'):
                            # 获取注意力权重
                            attention_weights = outputs.attentions  # tuple of [batch_size, num_heads, seq_len, seq_len]
                            attention_weights = torch.stack(attention_weights,
                                                            dim=1)  # [batch_size, num_layers, num_heads, seq_len, seq_len]

                            # 获取tokenizer实例
                            tokenizer_instance = getattr(dataloader.dataset, 'tokenizer', None)
                            if tokenizer_instance is None:
                                logger.warning("无法获取tokenizer实例，跳过注意力权重处理")
                            else:
                                original_functions = classification_batch['original_function']
                                line_attention_scores = process_attention_weights(
                                    attention_weights, input_ids, tokenizer_instance, original_functions
                                )

                                # 存储注意力得分，使用更好的键格式
                                for i, line_scores in enumerate(line_attention_scores):
                                    # 尝试获取commit_id作为键
                                    commit_id = None
                                    if isinstance(classification_batch, dict) and 'commit_id' in classification_batch:
                                        commit_ids = classification_batch['commit_id']
                                        if isinstance(commit_ids, (list, tuple)) and i < len(commit_ids):
                                            commit_id = commit_ids[i]
                                        elif commit_ids is not None:  # 单个值的情况
                                            commit_id = commit_ids

                                    # 使用commit_id或者batch索引作为键
                                    if commit_id:
                                        sample_key = f"commit_{commit_id}"
                                    else:
                                        sample_key = f"batch_{batch_idx}_sample_{i}"

                                    all_attention_scores[sample_key] = line_scores

                                logger.info(f"成功提取了batch {batch_idx}的注意力权重，包含 {len(line_attention_scores)} 个样本")
                        else:
                            if not hasattr(outputs, 'attentions') or not outputs.attentions:
                                logger.warning("模型输出中没有找到注意力权重")
                            if not classification_batch.get('original_function'):
                                logger.warning("批次中没有找到原始函数信息")

                        # 恢复模型配置
                        model.codet5.config.output_attentions = original_output_attentions
                    else:
                        logger.warning(
                            f"模型结构不支持注意力权重提取。模型属性: {[attr for attr in dir(model) if not attr.startswith('_')]}")

                    # 获取分类logits
                    logits = model(task="classification", input_ids=input_ids, attention_mask=attention_mask)

                except Exception as e:
                    logger.warning(f"获取注意力权重时出错: {e}")
                    # 降级为普通的分类预测
                    logits = model(task="classification", input_ids=input_ids, attention_mask=attention_mask)
            else:
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

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': classification_rep,
        'class_accuracies': class_accuracies
    }

    if return_attention:
        result['attention_scores'] = all_attention_scores
        logger.info(f"收集了 {len(all_attention_scores)} 个样本的注意力得分")

    return result


# 建立数据关联映射
def build_data_mapping(classification_dataset, line_level_dataset):
    """
    建立分类数据集和行级数据集之间的映射关系
    通过commit_id进行关联
    """
    # 由于ClassificationDataset没有直接存储commit_id信息，
    # 我们需要重新读取原始数据文件来获取commit_id映射
    # 这里简化处理，返回基本的统计信息

    classification_count = len(classification_dataset.examples) if hasattr(classification_dataset, 'examples') else len(
        classification_dataset)
    line_level_count = len(line_level_dataset.examples) if hasattr(line_level_dataset, 'examples') else len(
        line_level_dataset)

    logger.info(f"分类数据集包含 {classification_count} 个样本")
    logger.info(f"行级数据集包含 {line_level_count} 个样本")

    # 返回空的映射，后续通过其他方式进行关联
    classification_mapping = {}
    line_level_mapping = {}
    common_commits = set()

    logger.info("数据映射构建完成，将通过batch处理时进行实际关联")

    return classification_mapping, line_level_mapping, common_commits


# 融合注意力得分和模型预测
def combine_attention_and_prediction(model_scores, attention_scores, commit_id, line_indices, alpha=0.7):
    """
    融合模型预测得分和注意力得分

    Args:
        model_scores: 模型预测的行级得分
        attention_scores: 分类任务得到的注意力得分
        commit_id: 提交ID
        line_indices: 当前行在原函数中的索引
        alpha: 融合权重，alpha * model_score + (1-alpha) * attention_score

    Returns:
        combined_scores: 融合后的得分
    """
    if not attention_scores or commit_id not in attention_scores:
        return model_scores

    commit_attention = attention_scores[commit_id]
    combined_scores = []

    for i, (model_score, line_idx) in enumerate(zip(model_scores, line_indices)):
        if line_idx < len(commit_attention):
            attention_score = commit_attention[line_idx]
            # 融合得分
            combined_score = alpha * model_score + (1 - alpha) * attention_score
        else:
            combined_score = model_score
        combined_scores.append(combined_score)

    return combined_scores


# 评估行级任务（增强版，结合注意力得分）
def evaluate_line_level(model, dataloader, config, classification_attention_scores=None):
    model.eval()
    all_line_scores = []
    all_han_line_scores = []  # 模拟HAN模型的得分，这里简化处理
    all_labels = []
    all_combined_scores = []  # 存储融合后的得分
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

            # 获取commit_id信息（如果有的话）
            commit_ids = line_level_batch.get('commit_id', [])
            line_numbers = line_level_batch.get('line_number_in_function', [])

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

            # 如果有注意力得分，进行融合
            if classification_attention_scores and commit_ids:
                try:
                    # 构建当前batch的注意力得分查找
                    batch_attention_scores = {}
                    for i, commit_id in enumerate(commit_ids):
                        if commit_id:
                            # 寻找匹配的注意力得分，尝试多种匹配方式
                            found_attention = None

                            # 方式1：直接匹配commit_格式
                            commit_key = f"commit_{commit_id}"
                            if commit_key in classification_attention_scores:
                                found_attention = classification_attention_scores[commit_key]

                            # 方式2：在所有键中查找包含该commit_id的项
                            if found_attention is None:
                                for att_key, att_scores in classification_attention_scores.items():
                                    if str(commit_id) in str(att_key):
                                        found_attention = att_scores
                                        break

                            if found_attention is not None:
                                batch_attention_scores[commit_id] = found_attention

                    # 融合得分
                    if batch_attention_scores:
                        # 对每个样本进行融合
                        combined_batch_scores = []

                        # 假设batch中的所有行都来自同一个commit
                        primary_commit_id = commit_ids[0] if commit_ids else None
                        if primary_commit_id and primary_commit_id in batch_attention_scores:
                            attention_scores_for_commit = batch_attention_scores[primary_commit_id]

                            # 融合当前batch的所有行得分
                            for j, (model_score, line_num) in enumerate(
                                    zip(line_scores, line_numbers if line_numbers else range(len(line_scores)))):
                                if isinstance(line_num, (int, float)) and line_num < len(attention_scores_for_commit):
                                    attention_score = attention_scores_for_commit[int(line_num)]
                                    # 融合得分: 70%模型预测 + 30%注意力得分
                                    combined_score = 0.7 * model_score + 0.3 * attention_score
                                else:
                                    combined_score = model_score
                                combined_batch_scores.append(combined_score)
                        else:
                            combined_batch_scores = line_scores

                        all_combined_scores.extend(combined_batch_scores)
                    else:
                        all_combined_scores.extend(line_scores)

                except Exception as e:
                    logger.warning(f"融合注意力得分时出错: {e}")
                    all_combined_scores.extend(line_scores)
            else:
                all_combined_scores.extend(line_scores)

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

    original_metrics = {
        'IFA': IFA,
        'top_20_percent_LOC_recall': top_20_percent_LOC_recall,
        'effort_at_20_percent_LOC_recall': effort_at_20_percent_LOC_recall,
        'top_10_acc': top_10_acc,
        'top_5_acc': top_5_acc
    }

    # 如果有融合得分，计算融合后的指标
    if all_combined_scores and len(all_combined_scores) >= min_length:
        all_combined_scores = all_combined_scores[:min_length]

        IFA_combined, top_20_percent_LOC_recall_combined, effort_at_20_percent_LOC_recall_combined, top_10_acc_combined, top_5_acc_combined = get_line_level_metrics(
            all_combined_scores, all_labels, all_han_line_scores
        )

        combined_metrics = {
            'IFA': IFA_combined,
            'top_20_percent_LOC_recall': top_20_percent_LOC_recall_combined,
            'effort_at_20_percent_LOC_recall': effort_at_20_percent_LOC_recall_combined,
            'top_10_acc': top_10_acc_combined,
            'top_5_acc': top_5_acc_combined
        }

        return {
            'original': original_metrics,
            'combined': combined_metrics,
            'improvement': {
                'ifa_improvement': IFA - IFA_combined,
                'top_20_percent_improvement': top_20_percent_LOC_recall_combined - top_20_percent_LOC_recall,
                'top_10_acc_improvement': top_10_acc_combined - top_10_acc
            }
        }
    else:
        return {'original': original_metrics}


# 主评估函数
def evaluate(config):
    set_seed()

    # 加载tokenizer - 智能检测tokenizer类型
    try:
        # 首先尝试从本地模型加载tokenizer（自动检测类型）
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path)
        logger.info(f"成功加载tokenizer，类型：{type(tokenizer).__name__}")
    except Exception as e:
        logger.warning(f"从本地路径加载tokenizer失败：{e}")
        # 回退到使用在线的CodeT5 tokenizer
        logger.info("回退到使用在线CodeT5 tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained('Salesforce/codet5-base')
        logger.info("成功加载在线CodeT5 tokenizer")

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

    # 建立数据映射关系
    logger.info("建立数据集间的映射关系...")
    classification_mapping, line_level_mapping, common_commits = build_data_mapping(
        classification_test_dataset, line_level_test_dataset
    )

    # 评估分类任务（包含注意力分析）
    logger.info("评估分类任务（包含注意力分析）...")
    classification_metrics = evaluate_classification(model, classification_dataloader, config, return_attention=True)

    logger.info(f"分类任务结果:")
    logger.info(f"准确率: {classification_metrics['accuracy']:.4f}")
    logger.info(f"精确率: {classification_metrics['precision']:.4f}")
    logger.info(f"召回率: {classification_metrics['recall']:.4f}")
    logger.info(f"F1分数: {classification_metrics['f1']:.4f}")

    # 获取注意力得分
    classification_attention_scores = classification_metrics.get('attention_scores', {})

    # 评估行级任务（结合注意力得分）
    logger.info("评估行级任务（结合注意力得分）...")
    line_level_metrics = evaluate_line_level(
        model, line_level_dataloader, config,
        classification_attention_scores=classification_attention_scores
    )

    # 输出行级任务结果
    if 'original' in line_level_metrics:
        logger.info(f"原始模型行级任务结果:")
        original_metrics = line_level_metrics['original']
        logger.info(f"IFA (初始误报): {original_metrics['IFA']}")
        logger.info(f"Top 20% LOC召回率: {original_metrics['top_20_percent_LOC_recall']:.4f}")
        logger.info(f"20% LOC召回率的工作量: {original_metrics['effort_at_20_percent_LOC_recall']:.4f}")
        logger.info(f"Top 10准确率: {original_metrics['top_10_acc']:.4f}")
        logger.info(f"Top 5准确率: {original_metrics['top_5_acc']:.4f}")

        # 如果有融合结果，输出对比
        if 'combined' in line_level_metrics:
            logger.info(f"\n融合注意力得分后的行级任务结果:")
            combined_metrics = line_level_metrics['combined']
            logger.info(f"IFA (初始误报): {combined_metrics['IFA']}")
            logger.info(f"Top 20% LOC召回率: {combined_metrics['top_20_percent_LOC_recall']:.4f}")
            logger.info(f"20% LOC召回率的工作量: {combined_metrics['effort_at_20_percent_LOC_recall']:.4f}")
            logger.info(f"Top 10准确率: {combined_metrics['top_10_acc']:.4f}")
            logger.info(f"Top 5准确率: {combined_metrics['top_5_acc']:.4f}")

            # 输出改进情况
            improvements = line_level_metrics['improvement']
            logger.info(f"\n性能改进:")
            logger.info(f"IFA改进: {improvements['ifa_improvement']:.1f} (越大越好)")
            logger.info(f"Top 20% LOC召回率改进: {improvements['top_20_percent_improvement']:.4f}")
            logger.info(f"Top 10准确率改进: {improvements['top_10_acc_improvement']:.4f}")
    else:
        # 兼容旧格式，直接使用返回的指标
        logger.info(f"行级任务结果:")
        logger.info(f"IFA: {line_level_metrics.get('IFA', 'N/A')}")
        logger.info(f"Top 20% LOC召回率: {line_level_metrics.get('top_20_percent_LOC_recall', 0):.4f}")
        logger.info(f"20% LOC召回率的工作量: {line_level_metrics.get('effort_at_20_percent_LOC_recall', 0):.4f}")
        logger.info(f"Top 10准确率: {line_level_metrics.get('top_10_acc', 0):.4f}")
        logger.info(f"Top 5准确率: {line_level_metrics.get('top_5_acc', 0):.4f}")

    # 保存注意力分析结果
    attention_analysis_path = os.path.join(config.output_dir, "attention_analysis.json")
    if classification_attention_scores:
        try:
            # 将注意力得分转换为可序列化的格式
            serializable_attention = {}
            for key, scores in classification_attention_scores.items():
                if isinstance(scores, (list, tuple)):
                    serializable_attention[key] = list(scores)
                else:
                    serializable_attention[key] = scores

            attention_summary = {
                'total_samples': len(serializable_attention),
                'attention_scores': serializable_attention,
                'analysis_timestamp': str(torch.cuda.current_device() if torch.cuda.is_available() else "cpu"),
                'fusion_strategy': '70% model + 30% attention'
            }

            with open(attention_analysis_path, 'w', encoding='utf-8') as f:
                json.dump(attention_summary, f, indent=2, ensure_ascii=False)

            logger.info(f"注意力分析结果已保存到: {attention_analysis_path}")
        except Exception as e:
            logger.warning(f"保存注意力分析结果时出错: {e}")

    return {
        'classification': classification_metrics,
        'line_level': line_level_metrics,
        'data_mapping': {
            'classification_samples': len(classification_mapping),
            'line_level_samples': len(line_level_mapping),
            'common_commits': len(common_commits)
        },
        'attention_analysis': {
            'enabled': bool(classification_attention_scores),
            'samples_analyzed': len(classification_attention_scores) if classification_attention_scores else 0,
            'saved_to': attention_analysis_path if classification_attention_scores else None
        }
    }


def test_attention_mechanism():
    """测试注意力机制功能"""
    logger.info("测试注意力机制功能...")

    # 创建简单的测试数据
    test_tokens = ['def', 'function', '(', 'x', ')', ':', 'return', 'x', '+', '1']
    test_scores = [0.1, 0.2, 0.05, 0.15, 0.05, 0.1, 0.2, 0.1, 0.03, 0.02]
    test_lines = ['def function(x):', '    return x + 1']

    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained('Salesforce/codet5-base')

    result = calculate_line_attention_scores(test_tokens, test_scores, test_lines, tokenizer)
    logger.info(f"测试结果: {result}")

    # 测试数据融合
    model_scores = [0.8, 0.3]
    attention_scores = {'test_commit': [0.2, 0.7]}
    combined = combine_attention_and_prediction(
        model_scores, attention_scores, 'test_commit', [0, 1]
    )
    logger.info(f"融合测试结果: {combined}")


if __name__ == "__main__":
    config = EvalConfig()
    logger.info(f"设备: {config.device}")

    # 可选择运行测试或正常评估
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_attention_mechanism()
    else:
        evaluate(config)