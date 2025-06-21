import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import os

logger = logging.getLogger(__name__)

def convert_dtype_dataframe(df, feature_name):
    df['fix'] = df['fix'].apply(lambda x: float(bool(x)))
    df = df.astype({i: 'float32' for i in feature_name})
    return df

manual_features_columns = ['la', 'ld', 'nf', 'ns', 'nd', 'entropy', 'ndev',
                           'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp', 'fix']



class InputFeatures(object):
    def __init__(self, commit_id, line_ids, attention_mask, commit_features, expert_features, line_label):
        self.commit_id = commit_id
        self.line_ids = line_ids
        self.attention_mask = attention_mask
        self.commit_features = commit_features
        self.expert_features = expert_features
        self.line_label = line_label

def convert_examples_to_features(item, cls_token='[CLS]', sep_token='[SEP]', sequence_a_segment_id=0,
                                 sequence_b_segment_id=1, cls_token_segment_id=1, pad_token_segment_id=0,
                                 pad_token=0, mask_padding_with_zero=True, no_abstraction=True,
                                 buggy_commit_lines_df=None, args=None):
    commit_id, files, msg, label, tokenizer, args, manual_features = item

    old_add_code_lines = list(files['added_code'])
    old_delete_code_lines = list(files['removed_code'])

    add_code_lines_labels = []
    delete_code_lines_labels = []

    if commit_id in buggy_commit_lines_df['commit hash'].to_list():
        commit_info_df = buggy_commit_lines_df[buggy_commit_lines_df['commit hash'] == commit_id].reset_index(drop=True)
        commit_info_df['code line'] = commit_info_df['code line'].apply(lambda x: re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]", '', x))

        add_code_lines_dict = {re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]", '', line): line for line in old_add_code_lines}
        delete_code_lines_dict = {re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]", '', line): line for line in old_delete_code_lines}

        commit_info_df_added = commit_info_df[commit_info_df['change type'] == 'added'].reset_index(drop=True)
        add_code_lines = [add_code_lines_dict[line] for line in commit_info_df_added['code line'] if line in add_code_lines_dict]
        add_code_lines_labels = commit_info_df_added['label'].tolist()

        commit_info_df_deleted = commit_info_df[commit_info_df['change type'] == 'deleted'].reset_index(drop=True)
        delete_code_lines = [delete_code_lines_dict[line] for line in commit_info_df_deleted['code line'] if line in delete_code_lines_dict]
        delete_code_lines_labels = commit_info_df_deleted['label'].tolist()
    else:
        add_code_lines = old_add_code_lines
        delete_code_lines = old_delete_code_lines
        add_code_lines_labels = [0] * len(add_code_lines)
        delete_code_lines_labels = [0] * len(delete_code_lines)

    assert len(add_code_lines) == len(add_code_lines_labels)
    assert len(delete_code_lines) == len(delete_code_lines_labels)

    add_code_lines = [line for line in add_code_lines]
    delete_code_lines = [line for line in delete_code_lines]

    def process_line_add_delete_emptyline(code_lines, code_lines_label, type_code=None):
        temp_lines = []
        temp_labels = []
        for idx, line in enumerate(code_lines):
            if len(line):
                if type_code == 'added':
                    temp_lines.append('[ADD] ' + line)
                elif type_code == 'deleted':
                    temp_lines.append('[DEL] ' + line)
                temp_labels.append(code_lines_label[idx])
        return temp_lines, temp_labels

    add_code_lines, add_code_lines_labels = process_line_add_delete_emptyline(add_code_lines, add_code_lines_labels, 'added')
    delete_code_lines, delete_code_lines_labels = process_line_add_delete_emptyline(delete_code_lines, delete_code_lines_labels, 'deleted')

    cm_codelines = add_code_lines + delete_code_lines
    cm_codeline_labels = add_code_lines_labels + delete_code_lines_labels

    if len(cm_codelines) >= args.max_codeline_length:
        cm_codelines = cm_codelines[:args.max_codeline_length]
        cm_codeline_labels = cm_codeline_labels[:args.max_codeline_length]
    else:
        cm_codelines.extend([''] * (args.max_codeline_length - len(cm_codelines)))
        cm_codeline_labels.extend([0] * (args.max_codeline_length - len(cm_codeline_labels)))

    encodings = [tokenizer(line, max_length=args.max_codeline_token_length, padding='max_length', truncation=True, return_tensors='pt') for line in cm_codelines]
    line_ids = torch.stack([enc['input_ids'].squeeze(0) for enc in encodings])
    attention_mask = torch.stack([enc['attention_mask'].squeeze(0) for enc in encodings])

    # 生成 commit_features（示例：使用 tokenizer 编码 msg）
    msg_encoding = tokenizer(msg, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    commit_features = msg_encoding['input_ids'].squeeze(0)  # 假设使用 input_ids 作为特征
    expert_features = torch.tensor(manual_features, dtype=torch.float32)

    return InputFeatures(
        commit_id=commit_id,
        line_ids=line_ids,
        attention_mask=attention_mask,
        commit_features=commit_features,
        expert_features=expert_features,
        line_label=cm_codeline_labels
    )

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, changes_file_path=None, features_file_path=None, buggy_lines_file_path=None, mode='train'):
        self.examples = []
        self.args = args

        # 加载三个数据集
        buggy_commit_lines_df = pd.read_pickle(buggy_lines_file_path)
        ddata = pd.read_pickle(changes_file_path)
        features_data = pd.read_pickle(features_file_path)
        features_data = convert_dtype_dataframe(features_data, manual_features_columns)
        features_data = features_data[['commit_hash'] + manual_features_columns]

        data = []
        commit_ids, labels, msgs, codes = ddata
        for commit_id, label, msg, files in zip(commit_ids, labels, msgs, codes):
            manual_features = features_data[features_data['commit_hash'] == commit_id][manual_features_columns].to_numpy().squeeze()
            data.append((commit_id, files, msg, label, tokenizer, args, manual_features))

        random.seed(args.seed)
        if mode == 'train':
            random.shuffle(data)

        self.examples = [convert_examples_to_features(x, buggy_commit_lines_df=buggy_commit_lines_df, args=args) for x in tqdm(data, total=len(data))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (
            self.examples[item].line_ids,  # 直接返回张量
            self.examples[item].attention_mask,  # 直接返回张量
            self.examples[item].commit_features,  # 直接返回张量
            self.examples[item].expert_features,  # 直接返回张量
            torch.tensor(self.examples[item].line_label, dtype=torch.long)  # line_label 是列表，需要转换
        )


def get_line_level_metrics(line_score, label, han_line_score):
    # 确保所有输入列表长度一致
    min_length = min(len(line_score), len(han_line_score), len(label))
    line_score = line_score[:min_length]
    han_line_score = han_line_score[:min_length]
    label = label[:min_length]

    scaler = MinMaxScaler()
    line_score = scaler.fit_transform(np.array(line_score).reshape(-1, 1))
    line_score = [float(val) for val in list(line_score)]

    han_scaler = MinMaxScaler()
    han_line_score = han_scaler.fit_transform(np.array(han_line_score).reshape(-1, 1))
    han_line_score = [float(val) for val in list(han_line_score)]

    # 确保两个列表长度一致后再计算temp_score
    temp_score = [(line_score[i] + han_line_score[i]) / 2 for i in range(len(line_score))]
    line_score = han_line_score  # 这里使用han_line_score作为最终的line_score

    pred = np.round(line_score)
    line_df = pd.DataFrame({'scr': line_score, 'label': label})
    line_df = line_df.sort_values(by='scr', ascending=False)
    line_df['row'] = np.arange(1, len(line_df) + 1)

    real_buggy_lines = line_df[line_df['label'] == 1]
    if len(real_buggy_lines) < 1:
        IFA = len(line_df)
        top_20_percent_LOC_recall = 0
        effort_at_20_percent_LOC_recall = math.ceil(0.2 * len(line_df))
        top_10_acc = 0  # 添加默认值
        top_5_acc = 0   # 添加默认值
    else:
        IFA = line_df[line_df['label'] == 1].iloc[0]['row'] - 1
        label_list = list(line_df['label'])
        all_rows = len(label_list)

        top_10_acc = np.sum(label_list[:min(10, all_rows)]) / min(10, all_rows)
        top_5_acc = np.sum(label_list[:min(5, all_rows)]) / min(5, all_rows)

        LOC_20_percent = line_df.head(int(0.2 * len(line_df)))
        buggy_line_num = LOC_20_percent[LOC_20_percent['label'] == 1]
        top_20_percent_LOC_recall = float(len(buggy_line_num)) / float(len(real_buggy_lines))

        buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
        if len(buggy_20_percent) > 0:  # 添加检查以避免空DataFrame
            buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
            effort_at_20_percent_LOC_recall = int(buggy_20_percent_row_num) / float(len(line_df))
        else:
            effort_at_20_percent_LOC_recall = 0

    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc
