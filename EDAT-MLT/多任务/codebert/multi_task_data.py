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

                if code is None or cwe_id_val is None or not isinstance(cwe_id_val, str):
                    logger.warning(f"跳过不完整或格式不正确的数据行: {data}")
                    continue
                
                cwe_id_val = cwe_id_val.strip()
                if cwe_id_val in self.label2id:
                    self.examples.append(code)
                    self.labels.append(self.label2id[cwe_id_val])
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
            'label': torch.tensor(label, dtype=torch.long)
        }


# 行级预测数据集
class LineLevelDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, expert_feature_dim=14):
        self.examples = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.expert_feature_dim = expert_feature_dim

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_content in f:
                data = json.loads(line_content.strip())
                line_text = data.get("line_text")
                is_vulnerable = data.get("is_vulnerable_line")

                if line_text is None or is_vulnerable is None:
                    logger.warning(f"跳过不完整的数据行: {data} in file {file_path}")
                    continue
                
                self.examples.append(line_text)
                try:
                    self.labels.append(int(is_vulnerable)) #确保标签是整数
                except ValueError:
                    logger.warning(f"无法将 is_vulnerable_line 转换为整数: {is_vulnerable} for data {data} in file {file_path}")
                    continue

        logger.info(f"成功加载了 {len(self.examples)} 个行级样本从 {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        line_text = self.examples[idx]
        label = self.labels[idx]

        # 注意：如果 line_text 需要与 preprocess_code_line 类似的预处理，
        # 你可能需要在这里调用它或一个类似的函数。
        # line_text = preprocess_code_line(line_text) # 如果需要

        encoding = self.tokenizer(
            line_text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        # 为了与模型期望的格式匹配，我们需要改变返回数据的结构
        # 创建一个包含单行代码的3D张量 [batch_size=1, seq_len=1, token_len]
        line_ids = encoding['input_ids'].unsqueeze(0)  # 添加一个维度作为seq_len
        line_attention_mask = encoding['attention_mask'].unsqueeze(0)
        
        # 创建伪造的commit_features和expert_features（如果没有真实数据）
        # 通常这些应该从数据中获取，但这里为了演示我们创建占位符
        commit_features = torch.zeros(128)  # 假设commit_features的维度是128
        expert_features = torch.zeros(self.expert_feature_dim)  # 使用配置的专家特征维度
        
        return {
            'task': 'line_level',
            'line_ids': line_ids,  # [1, token_len] -> 单行代码
            'attention_mask': line_attention_mask,
            'commit_features': commit_features,
            'expert_features': expert_features,
            'line_label': torch.tensor(label, dtype=torch.long).unsqueeze(0)  # 添加seq_len维度
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
            'label': torch.stack([item['label'] for item in classification_batch])
        }
    else:
        classification_data = None

    # 处理行级任务批次
    if line_level_batch:
        # 由于我们已经在__getitem__中为每行添加了seq_len维度
        # 这里需要将所有行堆叠起来形成一个批次
        line_level_data = {
            'task': 'line_level',
            'line_ids': torch.cat([item['line_ids'] for item in line_level_batch], dim=0),
            'attention_mask': torch.cat([item['attention_mask'] for item in line_level_batch], dim=0),
            'commit_features': torch.stack([item['commit_features'] for item in line_level_batch]),
            'expert_features': torch.stack([item['expert_features'] for item in line_level_batch]),
            'line_label': torch.cat([item['line_label'] for item in line_level_batch], dim=0)
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