import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from transformers import T5Tokenizer, AutoTokenizer
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
                cwe_id_val = data.get("CWE ID") # 使用 .get 以避免 KeyError
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


# 行级预测数据集
class LineLevelDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, expert_feature_dim=14, context_lines=3, classification_data_paths=None):
        self.examples = []
        self.labels = []
        self.contexts = []  # 存储上下文信息
        self.line_positions = []  # 存储行在函数中的位置
        self.commit_ids = []  # 存储commit_id信息
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.expert_feature_dim = expert_feature_dim
        self.context_lines = context_lines  # 上下文行数
        
        # 加载分类任务的数据用于提取完整函数
        self.function_data = {}
        if classification_data_paths:
            self._load_classification_data(classification_data_paths)

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_content in f:
                data = json.loads(line_content.strip())
                line_text = data.get("line_text")
                is_vulnerable = data.get("is_vulnerable_line")
                line_number = data.get("line_number_in_function")
                commit_id = data.get("commit_id")
                file_name = data.get("file_name")
                func_identifier = data.get("original_function_identifier")
                original_vul = data.get("original_vul")

                if line_text is None or is_vulnerable is None:
                    logger.warning(f"跳过不完整的数据行: {data} in file {file_path}")
                    continue
                
                # 提取上下文
                context = self._extract_context(
                    line_text, line_number, commit_id, file_name, 
                    func_identifier, original_vul
                )
                
                self.examples.append(line_text)
                self.contexts.append(context)
                self.line_positions.append(line_number if line_number is not None else 0)
                self.commit_ids.append(commit_id)  # 存储commit_id
                
                try:
                    self.labels.append(int(is_vulnerable))
                except ValueError:
                    logger.warning(f"无法将 is_vulnerable_line 转换为整数: {is_vulnerable} for data {data} in file {file_path}")
                    continue

        logger.info(f"成功加载了 {len(self.examples)} 个行级样本从 {file_path}")

    def _load_classification_data(self, classification_data_paths):
        """加载分类任务数据，用于提取完整函数信息"""
        for path in classification_data_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line_content in f:
                        data = json.loads(line_content.strip())
                        func_before = data.get("func_before")
                        commit_id = data.get("commit_id")
                        
                        if func_before and commit_id:
                            # 使用commit_id作为key存储函数信息
                            if commit_id not in self.function_data:
                                self.function_data[commit_id] = []
                            self.function_data[commit_id].append(func_before)
                            
                logger.info(f"从 {path} 加载了函数数据")
            except Exception as e:
                logger.warning(f"无法加载分类数据文件 {path}: {e}")

    def _extract_context(self, line_text, line_number, commit_id, file_name, func_identifier, original_vul):
        """提取代码行的上下文信息"""
        context_info = {
            'before_lines': [],
            'after_lines': [],
            'function_signature': '',
            'relative_position': 0.0  # 行在函数中的相对位置
        }
        
        # 尝试从原始函数中提取上下文
        if original_vul and line_number is not None:
            function_lines = original_vul.split('\n')
            context_info['relative_position'] = (line_number - 1) / max(len(function_lines), 1)
            
            # 提取前后几行作为上下文
            line_idx = line_number - 1  # 转换为0索引
            
            # 提取前面的行
            start_idx = max(0, line_idx - self.context_lines)
            context_info['before_lines'] = function_lines[start_idx:line_idx]
            
            # 提取后面的行
            end_idx = min(len(function_lines), line_idx + self.context_lines + 1)
            context_info['after_lines'] = function_lines[line_idx + 1:end_idx]
            
            # 提取函数签名（通常是第一行）
            if function_lines:
                context_info['function_signature'] = function_lines[0].strip()
        
        # 如果没有原始函数信息，尝试从分类数据中获取
        elif commit_id in self.function_data:
            for func in self.function_data[commit_id]:
                if line_text.strip() in func:
                    function_lines = func.split('\n')
                    # 找到当前行在函数中的位置
                    for i, func_line in enumerate(function_lines):
                        if line_text.strip() in func_line.strip():
                            line_idx = i
                            context_info['relative_position'] = i / max(len(function_lines), 1)
                            
                            # 提取上下文
                            start_idx = max(0, line_idx - self.context_lines)
                            context_info['before_lines'] = function_lines[start_idx:line_idx]
                            
                            end_idx = min(len(function_lines), line_idx + self.context_lines + 1)
                            context_info['after_lines'] = function_lines[line_idx + 1:end_idx]
                            
                            if function_lines:
                                context_info['function_signature'] = function_lines[0].strip()
                            break
                    break
        
        return context_info

    def _format_context(self, context_info):
        """将上下文信息格式化为文本"""
        context_parts = []
        
        # 添加函数签名
        if context_info['function_signature']:
            context_parts.append(f"[FUNC] {context_info['function_signature']}")
        
        # 添加前面的行
        for i, line in enumerate(context_info['before_lines']):
            if line.strip():
                context_parts.append(f"[BEFORE-{len(context_info['before_lines'])-i}] {line.strip()}")
        
        # 添加位置信息
        rel_pos = context_info['relative_position']
        if rel_pos < 0.3:
            pos_desc = "函数开始部分"
        elif rel_pos > 0.7:
            pos_desc = "函数结束部分"
        else:
            pos_desc = "函数中间部分"
        context_parts.append(f"[POS] {pos_desc}")
        
        # 添加后面的行
        for i, line in enumerate(context_info['after_lines']):
            if line.strip():
                context_parts.append(f"[AFTER+{i+1}] {line.strip()}")
        
        return " ".join(context_parts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        line_text = self.examples[idx]
        label = self.labels[idx]
        context_info = self.contexts[idx]
        line_position = self.line_positions[idx]
        commit_id = self.commit_ids[idx]

        # 格式化上下文
        context_text = self._format_context(context_info)
        
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
        
        # 保持原有的2维格式，在collate_batch中处理维度
        line_ids = encoding['input_ids'].squeeze(0)  # [max_length]
        line_attention_mask = encoding['attention_mask'].squeeze(0)  # [max_length]
        
        # 创建位置特征（归一化的行位置）
        position_feature = torch.tensor([
            context_info['relative_position'],  # 相对位置
            len(context_info['before_lines']) / 10.0,  # 前面行数（归一化）
            len(context_info['after_lines']) / 10.0,   # 后面行数（归一化）
            line_position / 100.0 if line_position else 0.0  # 绝对行号（归一化）
        ], dtype=torch.float32)
        
        # 创建commit_features和expert_features（保持与原有接口兼容）
        commit_features = torch.zeros(128)
        expert_features = position_feature.repeat(self.expert_feature_dim // 4 + 1)[:self.expert_feature_dim]
        
        return {
            'task': 'line_level',
            'line_ids': line_ids,
            'attention_mask': line_attention_mask,
            'commit_features': commit_features,
            'expert_features': expert_features,
            'line_label': torch.tensor(label, dtype=torch.long),
            'commit_id': commit_id,  # 添加commit_id信息
            'line_number_in_function': line_position  # 添加行号信息
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
            'line_ids': torch.stack([item['line_ids'] for item in line_level_batch]).unsqueeze(1),  # [batch_size, 1, token_len]
            'attention_mask': torch.stack([item['attention_mask'] for item in line_level_batch]).unsqueeze(1),  # [batch_size, 1, token_len]
            'commit_features': torch.stack([item['commit_features'] for item in line_level_batch]),
            'expert_features': torch.stack([item['expert_features'] for item in line_level_batch]),
            'line_label': torch.stack([item['line_label'] for item in line_level_batch]).unsqueeze(1),  # [batch_size, 1]
            'commit_id': [item['commit_id'] for item in line_level_batch],  # 保持为列表
            'line_number_in_function': [item['line_number_in_function'] for item in line_level_batch]  # 保持为列表
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