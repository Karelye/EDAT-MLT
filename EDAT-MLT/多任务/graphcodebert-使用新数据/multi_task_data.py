import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from transformers import RobertaTokenizer
import pandas as pd
import re
from tqdm import tqdm
import random
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 代码漏洞分类数据集
class ClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id=None, max_length=512):
        self.examples = []
        self.labels = []
        self.commit_ids = []  # 存储commit_id信息
        self.original_functions = []  # 存储原始函数，用于注意力分析
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 收集所有标签
        all_labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_content in f:
                data = json.loads(line_content.strip())
                # 修改: 使用"CWE ID"作为标签字段名
                cwe_id_val = data.get("CWE ID")  # 使用 .get 以避免 KeyError
                if cwe_id_val is None or not isinstance(cwe_id_val, str):
                    logger.warning(f"期望 CWE ID 为字符串，但得到: {type(cwe_id_val)} for data {data}")
                    continue
                all_labels.append(cwe_id_val.strip())

        # 创建标签映射
        if label2id is None:
            self.label2id, self.id2label = self._create_label_mapping(all_labels)
        else:
            self.label2id = label2id
            self.id2label = {v: k for k, v in label2id.items()}

        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_content in f:
                data = json.loads(line_content.strip())
                # 修改: 读取 func_before
                code = data.get("func_before")
                # 修改: 使用"CWE ID"作为标签字段名
                cwe_id_val = data.get("CWE ID")
                commit_id = data.get("commit_id")

                if code is None or cwe_id_val is None or not isinstance(cwe_id_val, str):
                    logger.warning(f"跳过不完整或格式不正确的数据行: {data}")
                    continue

                cwe_id_val = cwe_id_val.strip()
                if cwe_id_val in self.label2id:
                    self.examples.append(code)
                    self.labels.append(self.label2id[cwe_id_val])
                    self.commit_ids.append(commit_id)
                    self.original_functions.append(code.split('\n') if code else [])
                # else:
                #     logger.warning(f"标签 '{cwe_id_val}' 未在 label2id 映射中找到。数据: {data}")

        logger.info(f"成功加载了 {len(self.examples)} 个分类样本从 {file_path}")

    def _create_label_mapping(self, labels):
        """创建标签到索引的映射"""
        label_set = sorted(list(set(labels)))
        label2id = {label: idx for idx, label in enumerate(label_set)}
        id2label = {idx: label for label, idx in label2id.items()}
        return label2id, id2label

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        code = self.examples[idx]
        label = self.labels[idx]
        commit_id = self.commit_ids[idx]
        original_function = self.original_functions[idx]

        # 使用tokenizer处理代码
        encoding = self.tokenizer(
            code,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        return {
            'task': 'classification',
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long),
            'commit_id': commit_id,
            'original_function': original_function
        }


# 行级预测数据集 - 适配新的BigVul行级数据格式
class LineLevelDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, expert_feature_dim=25, context_lines=5):
        self.examples = []  # 存储所有行的数据
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.expert_feature_dim = expert_feature_dim
        self.context_lines = context_lines

        # 加载实际的数据格式 - 每行JSONL是一行代码的完整信息
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_content in f:
                data = json.loads(line_content.strip())

                # 验证必要字段 - 适应实际格式，检查行级字段而非lines_data
                if not self._validate_line_data(data):
                    logger.warning(f"跳过不完整的行数据: {data.get('commit_id', 'unknown')}")
                    continue

                # 提取函数级元数据
                function_metadata = {
                    'cve_id': data.get('cve_id', ''),
                    'commit_id': data.get('commit_id', ''),
                    'file_path': data.get('file_path', ''),
                    'function_name': data.get('function_name', ''),
                    'cwe_id': data.get('cwe_id', ''),
                    'project': data.get('project', ''),
                    'lang': data.get('lang', ''),
                    'function_total_lines': data.get('function_total_lines', 100)  # 默认值，因为实际数据可能没有这个字段
                }

                # 直接使用当前数据作为行数据（因为每行JSONL就是一行代码）
                line_sample = {
                    'function_metadata': function_metadata,
                    'line_data': data  # 整个data就是line_data
                }
                self.examples.append(line_sample)

        logger.info(f"成功加载了 {len(self.examples)} 个行级样本从 {file_path}")

    def _validate_line_data(self, line_data):
        """验证行数据的完整性 - 适应实际数据格式，不验证标签泄露字段"""
        required_fields = [
            'line_text', 'relative_line_number', 'is_vulnerable',
            'line_length', 'indentation_level', 'num_tokens'
        ]

        # 检查基本字段
        for field in required_fields:
            if field not in line_data:
                logger.warning(f"行数据缺少必需字段: {field}")
                return False

        # 检查上下文文本字段是否存在（只验证文本，不验证标签）
        context_text_fields = [
            'previous_5_line_texts', 'next_5_line_texts'
        ]

        for field in context_text_fields:
            if field not in line_data:
                logger.warning(f"行数据缺少上下文字段: {field}")
                return False

        # 注意：故意不验证 previous_N_is_vulnerable_labels, next_N_is_vulnerable_labels
        # 和 vul_func_with_fix 字段，因为这些字段包含标签泄露信息，不应用于训练

        return True

    def _extract_expert_features(self, line_data):
        """从行数据中提取专家特征向量"""
        # 提取所有数值特征
        lexical_features = [
            line_data.get('line_length', 0),
            line_data.get('indentation_level', 0),
            line_data.get('num_tokens', 0),
            int(line_data.get('contains_pointer_op', False)),
            int(line_data.get('contains_array_op', False)),
            int(line_data.get('contains_assignment_op', False)),
            int(line_data.get('contains_comparison_op', False)),
            int(line_data.get('contains_arithmetic_op', False)),
            int(line_data.get('contains_logical_op', False)),
            line_data.get('num_function_calls', 0),
            line_data.get('num_keywords', 0),
            int(line_data.get('contains_memory_keywords', False)),
            int(line_data.get('contains_literal_string', False)),
            int(line_data.get('contains_literal_number', False)),
            int(line_data.get('is_comment_line', False)),
            int(line_data.get('is_preprocessor_directive', False))
        ]

        # 语法特征
        syntactic_features = [
            int(line_data.get('starts_with_control_flow', False)),
            int(line_data.get('is_function_definition_line', False)),
            int(line_data.get('is_return_statement', False)),
            line_data.get('nesting_depth_approx', 0)
        ]

        # 位置特征
        position_features = [
            line_data.get('relative_line_number', 0) / 100.0,  # 归一化行号
        ]

        # 合并所有特征
        all_features = lexical_features + syntactic_features + position_features

        # 确保特征维度正确
        if len(all_features) < self.expert_feature_dim:
            # 如果特征不够，用0填充
            all_features.extend([0.0] * (self.expert_feature_dim - len(all_features)))
        else:
            # 如果特征过多，截断
            all_features = all_features[:self.expert_feature_dim]

        return torch.tensor(all_features, dtype=torch.float32)

    def _format_context_from_new_data(self, line_data, function_metadata=None):
        """从实际数据格式中格式化上下文信息 - 包含vul_func_with_fix修复信息"""
        context_parts = []

        # 获取上下文行数据 - 只使用文本内容，不使用标签信息
        previous_lines = line_data.get('previous_5_line_texts', [])
        next_lines = line_data.get('next_5_line_texts', [])

        # 添加前面的行 - 移除标签信息，避免数据泄露
        for i, line in enumerate(previous_lines):
            if line and line.strip():  # 检查line是否为非空字符串
                context_parts.append(f"[BEFORE-{5 - i}] {line.strip()}")

        # 添加位置信息 - 从function metadata中获取总行数
        total_lines = function_metadata.get('function_total_lines', 100) if function_metadata else 100
        rel_pos = line_data.get('relative_line_number', 0) / max(total_lines, 1)
        if rel_pos < 0.3:
            pos_desc = "函数开始部分"
        elif rel_pos > 0.7:
            pos_desc = "函数结束部分"
        else:
            pos_desc = "函数中间部分"
        context_parts.append(f"[POS] {pos_desc}")

        # 添加后面的行 - 移除标签信息，避免数据泄露
        for i, line in enumerate(next_lines):
            if line and line.strip():  # 检查line是否为非空字符串
                context_parts.append(f"[AFTER+{i + 1}] {line.strip()}")

        # 添加修复信息 - 来自function metadata中的vul_func_with_fix字段
        if function_metadata and 'vul_func_with_fix' in function_metadata:
            fix_info = function_metadata['vul_func_with_fix']
            if fix_info and fix_info.strip():
                # 将修复信息截断到合理长度，避免序列过长
                fix_info_clean = fix_info.strip().replace('\n', ' ').replace('\t', ' ')
                # 限制修复信息长度
                max_fix_length = 200
                if len(fix_info_clean) > max_fix_length:
                    fix_info_clean = fix_info_clean[:max_fix_length] + "..."
                context_parts.append(f"[FIX] {fix_info_clean}")

        return " ".join(context_parts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        line_sample = self.examples[idx]
        function_metadata = line_sample['function_metadata']
        line_data = line_sample['line_data']

        # 获取基本信息
        line_text = line_data['line_text']
        label = line_data['is_vulnerable']

        # 格式化上下文
        context_text = self._format_context_from_new_data(line_data, function_metadata)

        # 组合当前行和上下文
        if context_text:
            combined_text = f"{context_text} [CURRENT] {line_text}"
        else:
            combined_text = f"[CURRENT] {line_text}"

        # 编码组合文本
        encoding = self.tokenizer(
            combined_text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        line_ids = encoding['input_ids'].squeeze(0)  # [max_length]
        line_attention_mask = encoding['attention_mask'].squeeze(0)  # [max_length]

        # 提取专家特征（新的多维特征）
        expert_features = self._extract_expert_features(line_data)

        # 注意：不再使用commit_features，所有信息都在expert_features中

        return {
            'task': 'line_level',
            'line_ids': line_ids,
            'attention_mask': line_attention_mask,
            'expert_features': expert_features,
            'line_label': torch.tensor(label, dtype=torch.long),
            'commit_id': function_metadata['commit_id'],
            'line_number_in_function': line_data['relative_line_number'],
            'function_metadata': function_metadata  # 添加函数元数据
        }


# 多任务数据批处理整理函数
def collate_batch(batch):
    """整理不同任务的批次数据"""
    classification_batch = []
    line_level_batch = []

    for item in batch:
        if item['task'] == 'classification':
            classification_batch.append(item)
        elif item['task'] == 'line_level':
            line_level_batch.append(item)

    # 处理分类任务批次
    if classification_batch:
        classification_data = {
            'task': 'classification',
            'input_ids': torch.stack([item['input_ids'] for item in classification_batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in classification_batch]),
            'label': torch.stack([item['label'] for item in classification_batch]),
            'commit_id': [item['commit_id'] for item in classification_batch],
            'original_function': [item['original_function'] for item in classification_batch]
        }
    else:
        classification_data = None

    # 处理行级任务批次
    if line_level_batch:
        # 将所有行级数据堆叠为3维张量 [batch_size, 1, token_len]
        # 每个样本被视为只有一行代码的序列
        line_level_data = {
            'task': 'line_level',
            'line_ids': torch.stack([item['line_ids'] for item in line_level_batch]).unsqueeze(1),
            # [batch_size, 1, token_len]
            'attention_mask': torch.stack([item['attention_mask'] for item in line_level_batch]).unsqueeze(1),
            # [batch_size, 1, token_len]
            'expert_features': torch.stack([item['expert_features'] for item in line_level_batch]),
            'line_label': torch.stack([item['line_label'] for item in line_level_batch]).unsqueeze(1),
            # [batch_size, 1]
            'commit_id': [item['commit_id'] for item in line_level_batch],  # 保持为列表
            'line_number_in_function': [item['line_number_in_function'] for item in line_level_batch],  # 保持为列表
            'function_metadata': [item['function_metadata'] for item in line_level_batch]  # 添加函数元数据
        }
    else:
        line_level_data = None

    return classification_data, line_level_data


# 混合数据集
class MixedDataset(Dataset):
    def __init__(self, classification_dataset, line_level_dataset, ratio=0.5):
        """
        创建混合数据集，包含两个任务的数据

        参数:
            classification_dataset: 分类任务数据集
            line_level_dataset: 行级任务数据集
            ratio: 分类任务占总批次的比例
        """
        self.classification_dataset = classification_dataset
        self.line_level_dataset = line_level_dataset
        self.ratio = ratio

        self.classification_len = len(classification_dataset) if classification_dataset else 0
        self.line_level_len = len(line_level_dataset) if line_level_dataset else 0

        # 检查两个数据集是否都为空
        if self.classification_len == 0 and self.line_level_len == 0:
            raise ValueError("两个数据集都为空，无法创建混合数据集")

        # 如果其中一个为空，将比例调整为只使用非空的数据集
        if self.classification_len == 0:
            logger.warning("分类数据集为空，仅使用行级数据集")
            self.ratio = 0.0
        elif self.line_level_len == 0:
            logger.warning("行级数据集为空，仅使用分类数据集")
            self.ratio = 1.0

        self.length = max(self.classification_len, self.line_level_len)
        logger.info(f"混合数据集创建完成: 分类数据 {self.classification_len} 样本, 行级数据 {self.line_level_len} 样本")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 根据比例随机选择任务，如果其中一个为空则始终选择非空的那个
        if self.line_level_len == 0 or (self.classification_len > 0 and random.random() < self.ratio):
            # 选择分类任务
            cl_idx = idx % self.classification_len
            return self.classification_dataset[cl_idx]
        else:
            # 选择行级任务
            ll_idx = idx % self.line_level_len
            return self.line_level_dataset[ll_idx]