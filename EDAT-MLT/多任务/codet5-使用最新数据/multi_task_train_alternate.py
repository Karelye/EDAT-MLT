import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import time
import random
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse

# 导入自定义模块
from multi_task_model import MultiTaskVulnerabilityModel
from multi_task_data import (
    ClassificationDataset, LineLevelDataset, collate_batch
)
from multi_task_pgd import PGDAttack, extract_identifiers_from_code, get_pgd_loss
from losses import FocalLoss

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 配置类
class Config:
    # 基本参数
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "multi_task_alternate_output_1"

    # 模型参数
    pretrained_model_path = r"E:\Python\model\codet5-base"  # 使用本地CodeT5模型（RobertaTokenizer）
    class_num_labels = 10  # 会在训练时根据数据集自动更新
    line_num_labels = 2
    expert_num = 6
    expert_dim = 768
    expert_feature_dim = 25

    # 数据参数
    max_length = 512
    max_codeline_length = 256
    max_codeline_token_length = 64

    # 训练参数
    batch_size = 2
    learning_rate = 1e-5
    encoder_lr = 5e-6
    num_epochs = 20
    warmup_ratio = 0.1
    weight_decay = 0.01
    gradient_accumulation_steps = 1  # 简化为1，以便更好地控制交替
    max_grad_norm = 1.0

    # 分类任务数据路径
    classification_train_path = r"E:\Python\code_people\最新实验\bigvul\train.jsonl"
    classification_test_path = r"E:\Python\code_people\最新实验\bigvul\test.jsonl"
    classification_valid_path = r"E:\Python\code_people\最新实验\bigvul\test.jsonl"

    # 行级任务数据路径 - 使用新的BigVul行级数据格式
    line_level_train_path = r"E:\Python\code_people\最新实验\bigvul\train_stratified.jsonl"  # 新格式的训练数据
    line_level_valid_path = r"E:\Python\code_people\最新实验\bigvul\test_stratified.jsonl"  # 新格式的验证数据
    line_level_test_path = r"E:\Python\code_people\最新实验\bigvul\test_stratified.jsonl"    # 新格式的测试数据
    # 是否使用交替训练
    use_alternate_training = False  # 新增参数，决定是否使用交替训练

    # 交替训练策略
    alternate_strategy = "epoch"  # 可选: "batch", "epoch", "progressive"

    # 对于"batch"策略的特定参数
    batch_alternate_frequency = 1  # 每训练多少个批次交替一次任务

    # 对于"epoch"策略的特定参数
    epoch_alternate_order = ["classification", "classification"]  # 任务交替顺序

    # 对于"progressive"策略的特定参数
    progressive_epochs = {
        "classification_only": 2,  # 先训练分类任务2个epoch
        "line_level_only": 2,  # 再训练行级任务2个epoch
        "classification_focus": 2,  # 然后主要训练分类(分类:行级 = 3:1)
        "line_level_focus": 2,  # 然后主要训练行级(行级:分类 = 3:1)
        "balanced": 2  # 最后平衡训练(分类:行级 = 1:1)
    }

    # 任务特定学习率 - 允许针对不同任务使用不同的学习率
    task_learning_rates = {
        "classification": 1e-5,
        "line_level": 2e-6
    }

    # 早停配置
    patience = 3
    min_delta = 0.001

    # 添加PGD对抗训练参数
    use_pgd = True  # 是否启用PGD对抗训练
    pgd_epsilon = 0.03  # PGD扰动大小
    pgd_alpha = 1e-2  # PGD步长
    pgd_n_steps = 3  # PGD迭代步数
    
    # 任务权重配置 - 控制两个任务在损失函数中的占比
    classification_weight = 2.0  # 分类任务权重，默认1.0
    line_level_weight = 1.0      # 行级任务权重，默认1.0
    # 权重归一化选项
    normalize_task_weights = True  # 是否将权重归一化使其和为1
    # 动态权重调整
    use_dynamic_weights = False    # 是否使用动态权重调整
    dynamic_weight_strategy = "uncertainty"  # 动态权重策略: "uncertainty", "gradient_norm", "loss_ratio"


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 计算任务权重的函数
def get_task_weights(config, classification_loss=None, line_level_loss=None, epoch=None):
    """
    根据配置和当前状态计算任务权重
    
    Args:
        config: 配置对象
        classification_loss: 分类任务损失 (用于动态权重计算)
        line_level_loss: 行级任务损失 (用于动态权重计算)
        epoch: 当前epoch (用于动态权重计算)
    
    Returns:
        tuple: (classification_weight, line_level_weight)
    """
    if config.use_dynamic_weights and classification_loss is not None and line_level_loss is not None:
        # 动态权重调整策略
        if config.dynamic_weight_strategy == "loss_ratio":
            # 基于损失比率的动态权重：损失越高，权重越低
            total_loss = classification_loss + line_level_loss
            if total_loss > 0:
                # 反比权重：损失高的任务权重降低
                classification_weight = line_level_loss / total_loss
                line_level_weight = classification_loss / total_loss
            else:
                classification_weight = config.classification_weight
                line_level_weight = config.line_level_weight
                
        elif config.dynamic_weight_strategy == "uncertainty":
            # 基于不确定性的动态权重（简化版本）
            # 损失越高说明模型越不确定，给予更高权重
            total_loss = classification_loss + line_level_loss
            if total_loss > 0:
                classification_weight = classification_loss / total_loss
                line_level_weight = line_level_loss / total_loss
            else:
                classification_weight = config.classification_weight
                line_level_weight = config.line_level_weight
                
        elif config.dynamic_weight_strategy == "gradient_norm":
            # 可以在后续版本中实现基于梯度范数的权重调整
            classification_weight = config.classification_weight
            line_level_weight = config.line_level_weight
            
        else:
            # 默认使用静态权重
            classification_weight = config.classification_weight
            line_level_weight = config.line_level_weight
    else:
        # 使用静态权重
        classification_weight = config.classification_weight
        line_level_weight = config.line_level_weight
    
    # 权重归一化
    if config.normalize_task_weights:
        total_weight = classification_weight + line_level_weight
        if total_weight > 0:
            classification_weight = classification_weight / total_weight
            line_level_weight = line_level_weight / total_weight
    
    return classification_weight, line_level_weight


# 验证函数 - 修改为仅验证分类任务
def validate(model, classification_dataloader, classification_loss_fn, config):
    model.eval()
    val_classification_loss = 0.0
    val_classification_batches = 0
    classification_correct = 0
    classification_total = 0

    with torch.no_grad():
        # 验证分类任务
        if classification_dataloader is not None:
            for batch in tqdm(classification_dataloader, desc="验证分类任务"):
                classification_batch, _ = batch

                if classification_batch is None:
                    continue

                input_ids = classification_batch['input_ids'].to(config.device)
                attention_mask = classification_batch['attention_mask'].to(config.device)
                labels = classification_batch['label'].to(config.device)

                logits = model(task="classification", input_ids=input_ids, attention_mask=attention_mask)
                classification_loss = classification_loss_fn(logits, labels)

                # 计算准确率
                preds = torch.argmax(logits, dim=1)
                classification_correct += (preds == labels).sum().item()
                classification_total += labels.size(0)

                val_classification_loss += classification_loss.item()
                val_classification_batches += 1

    # 计算平均损失和准确率
    avg_val_classification_loss = val_classification_loss / max(val_classification_batches, 1)
    classification_accuracy = classification_correct / max(classification_total, 1)

    return {
        'classification_loss': avg_val_classification_loss,
        'classification_accuracy': classification_accuracy,
        'total_loss': avg_val_classification_loss  # 仅使用分类损失作为总损失
    }


# 保存模型和配置
def save_model_with_config(model, config, epoch, val_metrics, best=False):
    os.makedirs(config.output_dir, exist_ok=True)

    if best:
        save_path = os.path.join(config.output_dir, "best_multi_task_model.pt")
        config_path = os.path.join(config.output_dir, "best_model_config.json")
    else:
        save_path = os.path.join(config.output_dir, f"multi_task_model_epoch_{epoch + 1}.pt")
        config_path = os.path.join(config.output_dir, f"model_config_epoch_{epoch + 1}.json")

    torch.save(model.state_dict(), save_path)

    config_dict = {
        'class_num_labels': config.class_num_labels,
        'line_num_labels': config.line_num_labels,
        'expert_num': config.expert_num,
        'expert_dim': config.expert_dim,
        'expert_feature_dim': config.expert_feature_dim,
        'max_length': config.max_length,
        'max_codeline_length': config.max_codeline_length,
        'max_codeline_token_length': config.max_codeline_token_length,
        'epoch': epoch + 1,
        'validation_metrics': val_metrics,
        'alternate_strategy': config.alternate_strategy
    }

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

    if best:
        logger.info(f"保存最佳模型到 {save_path}")
    else:
        logger.info(f"保存模型到 {save_path}")


# 进行单一任务训练的函数
def train_single_task(model, dataloader, optimizer, scheduler, loss_fn, config, task, tokenizer=None):
    """训练单个任务一个epoch，添加对抗训练"""
    model.train()
    total_loss = 0
    steps = 0

    # 初始化PGD攻击实例（如果启用）
    pgd = PGDAttack(model) if config.use_pgd and tokenizer is not None else None

    progress_bar = tqdm(dataloader, desc=f"训练 {task}")
    for batch_idx, batch_data in enumerate(progress_bar):
        # 切分数据为分类任务和行级任务
        classification_batch, line_level_batch = batch_data

        # 根据当前训练的任务选择相应的批次
        if task == "classification" and classification_batch is not None:
            batch = classification_batch

            # 对抗训练流程（分类任务）
            if config.use_pgd and pgd is not None:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['label'].to(config.device)

                # 前向传播
                logits = model(task="classification", input_ids=input_ids, attention_mask=attention_mask)

                # 计算损失
                loss = loss_fn(logits, labels)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)

                # 优化步骤
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 记录损失
                total_loss += loss.item()
                steps += 1

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        else:  # line_level
            # 从batch元组中获取行级数据
            if line_level_batch is None:
                continue

            # 从行级批次获取数据 (更新的键名)
            line_ids = line_level_batch['line_ids'].to(config.device)
            attention_mask = line_level_batch['attention_mask'].to(config.device)
            line_labels = line_level_batch['line_label'].to(config.device)
            
            # 正常前向传播 (已简化，去除annotation特征)
            outputs = model(
                task=task,
                line_ids=line_ids,
                attention_mask=attention_mask
            )
            # 使用损失函数计算损失值
            loss = loss_fn(outputs.view(-1, config.line_num_labels), line_labels.view(-1))

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)

            # 优化步骤
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 记录损失
            total_loss += loss.item()
            steps += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

    # 计算平均损失
    avg_loss = total_loss / max(steps, 1)
    return avg_loss


# 交替批次训练函数
def train_batch_alternate(model, dataloaders, optimizers, schedulers, loss_fns, config):
    """
    在批次级别交替训练不同任务
    
    注意：该函数不支持PGD对抗训练，因为它不接收tokenizer参数。
    如果config.use_pgd=True，将使用普通训练代替。
    """
    model.train()

    # 准备数据加载器迭代器
    classification_iter = iter(dataloaders["classification"])
    line_level_iter = iter(dataloaders["line_level"])

    # 确定批次总数(取较大的一个)
    total_classification_batches = len(dataloaders["classification"])
    total_line_level_batches = len(dataloaders["line_level"])
    total_batches = max(total_classification_batches, total_line_level_batches)

    # 损失记录
    classification_losses = []
    line_level_losses = []

    # 当前任务
    current_task = "classification"  # 从分类任务开始
    task_batch_count = 0  # 当前任务已处理的批次数

    progress_bar = tqdm(range(total_batches * 2), desc="批次交替训练")  # *2因为两个任务

    for _ in range(total_batches * 2):  # 每个原始批次处理两个任务
        if current_task == "classification" and task_batch_count < total_classification_batches:
            # 获取分类任务批次
            try:
                classification_batch = next(classification_iter)
            except StopIteration:
                classification_iter = iter(dataloaders["classification"])
                classification_batch = next(classification_iter)

            # 检查是否有有效的分类批次
            classification_loss = 0
            if classification_batch is not None and isinstance(classification_batch, tuple):
                batch = classification_batch[0]  # 从元组中获取分类数据
                if batch is not None:
                    # 处理分类任务
                    input_ids = batch['input_ids'].to(config.device)
                    attention_mask = batch['attention_mask'].to(config.device)
                    labels = batch['label'].to(config.device)

                    # 对抗训练流程（分类任务）
                    if config.use_pgd:
                        # 对抗训练需要tokenizer，但train_batch_alternate不接收tokenizer参数
                        # 所以这里只进行普通训练
                        optimizer = optimizers["classification"]
                        scheduler = schedulers["classification"]
                        
                        optimizer.zero_grad()

                        # 前向传播
                        logits = model(task="classification", input_ids=input_ids, attention_mask=attention_mask)

                        # 计算损失
                        loss = loss_fns["classification"](logits, labels)

                        # 反向传播
                        loss.backward()

                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)

                        # 优化步骤
                        optimizer.step()
                        scheduler.step()

                        # 记录损失
                        classification_loss = loss.item()
                    else:
                        # 正常前向传播（无对抗训练）
                        optimizer = optimizers["classification"]
                        scheduler = schedulers["classification"]
                        
                        optimizer.zero_grad()
                        
                        logits = model(task="classification", input_ids=input_ids, attention_mask=attention_mask)
                        loss = loss_fns["classification"](logits, labels)
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        
                        classification_loss = loss.item()

                    # 记录分类损失
                    classification_losses.append(classification_loss)

        # 修改train_batch_alternate中处理行级任务的部分
        elif current_task == "line_level" and task_batch_count < total_line_level_batches:
            # 获取行级任务批次
            try:
                line_level_batch = next(line_level_iter)
            except StopIteration:
                line_level_iter = iter(dataloaders["line_level"])
                line_level_batch = next(line_level_iter)

            # 检查是否有有效的行级批次
            line_level_loss = 0
            if line_level_batch is not None and isinstance(line_level_batch, tuple):
                batch = line_level_batch[1]  # 从元组中获取行级数据
                if batch is not None:
                    # 处理行级任务，确保必要的字段存在
                    line_ids = batch['line_ids'].to(config.device)
                    attention_mask = batch['attention_mask'].to(config.device)
                    line_labels = batch['line_label'].to(config.device)
                    
                    # 确保专家特征存在
                    if 'expert_features' in batch:
                        expert_features = batch['expert_features'].to(config.device)
                    else:
                        batch_size = line_ids.size(0)
                        expert_features = torch.zeros(batch_size, config.expert_feature_dim).to(config.device)

                    optimizer = optimizers["line_level"]
                    scheduler = schedulers["line_level"]
                    
                    optimizer.zero_grad()
                    
                    # 使用简化的模型接口，传递专家特征
                    logits = model(
                        task="line_level", 
                        line_ids=line_ids, 
                        attention_mask=attention_mask,
                        expert_features=expert_features
                    )
                    
                    loss = loss_fns["line_level"](logits.view(-1, config.line_num_labels), line_labels.view(-1))
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    
                    line_level_loss = loss.item()

                    # 记录行级损失
                    line_level_losses.append(line_level_loss)

        task_batch_count += 1

        # 检查是否需要切换任务
        if task_batch_count >= config.batch_alternate_frequency:
            current_task = "classification" if current_task == "line_level" else "line_level"
            task_batch_count = 0

        progress_bar.update(1)

    # 计算平均损失
    avg_classification_loss = sum(classification_losses) / max(len(classification_losses), 1)
    avg_line_level_loss = sum(line_level_losses) / max(len(line_level_losses), 1)
    
    # 计算加权总损失
    cls_weight, line_weight = get_task_weights(config, avg_classification_loss, avg_line_level_loss)
    weighted_total_loss = cls_weight * avg_classification_loss + line_weight * avg_line_level_loss

    return {
        "classification_loss": avg_classification_loss,
        "line_level_loss": avg_line_level_loss,
        "total_loss": weighted_total_loss,
        "classification_weight": cls_weight,
        "line_level_weight": line_weight
    }


# 添加联合训练函数（非交替训练）
def train_joint(model, dataloaders, optimizers, schedulers, loss_fns, config, tokenizer=None):
    """同时训练所有任务，不使用交替训练策略"""
    model.train()

    # 准备数据加载器迭代器
    classification_iter = iter(dataloaders["classification"])
    line_level_iter = iter(dataloaders["line_level"])

    # 确定批次总数(取较小的一个，以确保每个epoch两个任务都能够完整训练一遍)
    total_classification_batches = len(dataloaders["classification"])
    total_line_level_batches = len(dataloaders["line_level"])
    total_batches = min(total_classification_batches, total_line_level_batches)

    # 损失记录
    classification_losses = []
    line_level_losses = []

    # 初始化PGD攻击实例（如果启用）
    pgd = None
    if config.use_pgd and tokenizer is not None:
        pgd = PGDAttack(model)

    # 创建进度条
    progress_bar = tqdm(total=total_batches, desc="联合训练")

    # 主训练循环
    for _ in range(total_batches):
        # 获取分类任务批次
        try:
            classification_batch = next(classification_iter)
        except StopIteration:
            classification_iter = iter(dataloaders["classification"])
            classification_batch = next(classification_iter)

        # 检查是否有有效的分类批次
        classification_loss = 0
        if classification_batch is not None:
            # 从batch元组中获取分类数据
            if isinstance(classification_batch, tuple) and len(classification_batch) >= 1:
                batch = classification_batch[0]  # 从元组中获取分类数据
            else:
                batch = classification_batch

            if batch is not None and 'input_ids' in batch:
                # 处理分类任务
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['label'].to(config.device)

                # 对抗训练流程（分类任务）
                if config.use_pgd and pgd is not None and tokenizer is not None:
                    # 先计算当前损失
                    logits = model(task="classification", input_ids=input_ids, attention_mask=attention_mask)
                    current_loss = loss_fns["classification"](logits, labels)
                    model.current_loss = current_loss
                    
                    # 使用get_pgd_loss函数计算包含对抗训练的损失
                    classification_loss = get_pgd_loss(
                        model=model,
                        batch=batch,
                        task="classification",
                        loss_fn=loss_fns["classification"],
                        pgd_attack=pgd,  # pgd已经经过上面的检查确保不为None
                        tokenizer=tokenizer,  # tokenizer已经经过上面的检查确保不为None
                        config=config
                    )
                else:
                    # 正常前向传播（无对抗训练）
                    logits = model(task="classification", input_ids=input_ids, attention_mask=attention_mask)
                    classification_loss = loss_fns["classification"](logits, labels)

                # 记录分类损失
                classification_losses.append(classification_loss.item())

        # 获取行级任务批次
        line_level_loss = 0
        try:
            line_level_batch = next(line_level_iter)
        except StopIteration:
            line_level_iter = iter(dataloaders["line_level"])
            line_level_batch = next(line_level_iter)

        # 检查是否有有效的行级批次
        if line_level_batch is not None:
            # 从batch元组中获取行级数据
            if isinstance(line_level_batch, tuple) and len(line_level_batch) >= 2:
                batch = line_level_batch[1]  # 假设元组的第二个元素是行级数据
            else:
                batch = line_level_batch

            if batch is not None and 'line_ids' in batch:
                # 确保必要的字段存在
                if 'expert_features' not in batch:
                    batch_size = batch['line_ids'].size(0)
                    batch['expert_features'] = torch.zeros(batch_size, config.expert_feature_dim).to(config.device)
                    
                # 确保标签字段存在
                if 'line_label' not in batch:
                    logger.warning("行级批次中缺少'line_label'字段，跳过该批次")
                    continue

                # 复制必要的数据到设备
                for key in ['line_ids', 'attention_mask', 'expert_features', 'line_label']:
                    if key in batch:
                        batch[key] = batch[key].to(config.device)

                # 对抗训练流程（行级任务）
                if config.use_pgd and pgd is not None and tokenizer is not None:
                    # 先计算当前损失
                    outputs = model(
                        task="line_level",
                        line_ids=batch['line_ids'],
                        attention_mask=batch['attention_mask'],
                        expert_features=batch['expert_features']
                    )
                    current_loss = loss_fns["line_level"](
                        outputs.view(-1, config.line_num_labels),
                        batch['line_label'].view(-1)
                    )
                    model.current_loss = current_loss
                    
                    # 使用get_pgd_loss函数计算包含对抗训练的损失
                    line_level_loss = get_pgd_loss(
                        model=model,
                        batch=batch,
                        task="line_level",
                        loss_fn=loss_fns["line_level"],
                        pgd_attack=pgd,  # pgd已经经过上面的检查确保不为None
                        tokenizer=tokenizer,  # tokenizer已经经过上面的检查确保不为None
                        config=config
                    )
                else:
                    # 正常前向传播（无对抗训练）
                    outputs = model(
                        task="line_level",
                        line_ids=batch['line_ids'],
                        attention_mask=batch['attention_mask'],
                        expert_features=batch['expert_features']
                    )
                    line_level_loss = loss_fns["line_level"](
                        outputs.view(-1, config.line_num_labels),
                        batch['line_label'].view(-1)
                    )

                # 记录行级损失
                line_level_losses.append(line_level_loss.item())
            elif 'input_ids' in batch:
                # 这种情况下，数据格式不符合行级任务的需求，应该跳过
                logger.warning("检测到行级任务的批次使用了'input_ids'而不是'line_ids'，跳过该批次")
                continue

        # 如果两个任务都有有效损失，则进行联合优化
        if classification_loss != 0 or line_level_loss != 0:
            # 计算任务权重
            cls_weight, line_weight = get_task_weights(
                config, 
                classification_loss.item() if isinstance(classification_loss, torch.Tensor) else classification_loss,
                line_level_loss.item() if isinstance(line_level_loss, torch.Tensor) else line_level_loss
            )
            
            # 计算加权总损失
            total_loss = cls_weight * classification_loss + line_weight * line_level_loss

            # 梯度清零
            for optimizer in optimizers.values():
                optimizer.zero_grad()

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)

            # 优化步骤
            for optimizer in optimizers.values():
                optimizer.step()

            for scheduler in schedulers.values():
                scheduler.step()

        # 更新进度条
        avg_class_loss = sum(classification_losses) / max(len(classification_losses), 1)
        avg_line_loss = sum(line_level_losses) / max(len(line_level_losses), 1)
        progress_bar.set_postfix({
            'class_loss': f'{avg_class_loss:.4f}',
            'line_loss': f'{avg_line_loss:.4f}',
            'lr': f'{schedulers["classification"].get_last_lr()[0]:.2e}'
        })
        progress_bar.update(1)

    # 计算平均损失
    avg_classification_loss = sum(classification_losses) / max(len(classification_losses), 1)
    avg_line_level_loss = sum(line_level_losses) / max(len(line_level_losses), 1)
    
    # 计算加权总损失
    cls_weight, line_weight = get_task_weights(config, avg_classification_loss, avg_line_level_loss)
    weighted_total_loss = cls_weight * avg_classification_loss + line_weight * avg_line_level_loss

    return {
        "classification_loss": avg_classification_loss,
        "line_level_loss": avg_line_level_loss,
        "total_loss": weighted_total_loss,
        "classification_weight": cls_weight,
        "line_level_weight": line_weight
    }


# 训练函数
def train(config):
    # 设置随机种子
    set_seed(config.seed)

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 初始化tokenizer - 使用RobertaTokenizer加载CodeT5模型
    tokenizer = RobertaTokenizer.from_pretrained(config.pretrained_model_path)

    # 加载分类数据集
    logger.info("加载分类数据集...")
    classification_train_dataset = ClassificationDataset(
        config.classification_train_path, tokenizer, max_length=config.max_length
    )

    # 加载分类验证数据集，用于早停
    logger.info("加载分类验证数据集...")
    classification_valid_dataset = ClassificationDataset(
        config.classification_valid_path, tokenizer,
        label2id=classification_train_dataset.label2id,
        max_length=config.max_length
    )

    # 更新配置中的类别数量
    config.class_num_labels = len(classification_train_dataset.label2id)
    logger.info(f"分类任务有 {config.class_num_labels} 个类别")

    # 保存标签映射，用于之后的评估
    with open(os.path.join(config.output_dir, "label_mapping.json"), "w") as f:
        json.dump({
            "label2id": classification_train_dataset.label2id,
            "id2label": classification_train_dataset.id2label
        }, f)

    # 加载行级数据集
    logger.info("加载行级数据集...")
    
    # 准备分类数据路径列表，用于行级数据集提取上下文
    # 加载行级数据集 - 使用实际的JSONL格式数据（每行是一行代码）
    line_level_train_dataset = LineLevelDataset(
        file_path=config.line_level_train_path,
        tokenizer=tokenizer,
        max_length=config.max_codeline_token_length,
        expert_feature_dim=config.expert_feature_dim,
        context_lines=5  # 与实际数据格式中的previous_5/next_5保持一致
    )
    line_level_valid_dataset = LineLevelDataset(
        file_path=config.line_level_valid_path,
        tokenizer=tokenizer,
        max_length=config.max_codeline_token_length,
        expert_feature_dim=config.expert_feature_dim,
        context_lines=5
    )
    line_level_test_dataset = LineLevelDataset(
        file_path=config.line_level_test_path,
        tokenizer=tokenizer,
        max_length=config.max_codeline_token_length,
        expert_feature_dim=config.expert_feature_dim,
        context_lines=5
    )

    # 创建数据加载器
    classification_train_dataloader = DataLoader(
        classification_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    classification_valid_dataloader = DataLoader(
        classification_valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    line_level_train_dataloader = DataLoader(
        line_level_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    # 初始化模型
    logger.info("初始化模型...")
    model = MultiTaskVulnerabilityModel(
        pretrained_model_path=config.pretrained_model_path,
        class_num_labels=config.class_num_labels,
        line_num_labels=config.line_num_labels,
        expert_feature_dim=config.expert_feature_dim,
        expert_num=config.expert_num,
        expert_dim=config.expert_dim,
        max_codeline_length=config.max_codeline_length
    ).to(config.device)

    # 为不同任务设置不同的优化器
    classification_optimizer = optim.AdamW(
        model.parameters(),
        lr=config.task_learning_rates.get("classification", config.learning_rate),
        weight_decay=config.weight_decay
    )

    line_level_optimizer = optim.AdamW(
        model.parameters(),
        lr=config.task_learning_rates.get("line_level", config.learning_rate),
        weight_decay=config.weight_decay
    )

    # 计算训练步数
    num_classification_steps = len(classification_train_dataloader) * config.num_epochs
    num_line_level_steps = len(line_level_train_dataloader) * config.num_epochs

    # 创建学习率调度器
    classification_scheduler = get_linear_schedule_with_warmup(
        classification_optimizer,
        num_warmup_steps=int(num_classification_steps * config.warmup_ratio),
        num_training_steps=num_classification_steps
    )

    line_level_scheduler = get_linear_schedule_with_warmup(
        line_level_optimizer,
        num_warmup_steps=int(num_line_level_steps * config.warmup_ratio),
        num_training_steps=num_line_level_steps
    )

    # 设置损失函数
    classification_loss_fn = nn.CrossEntropyLoss()  # 保持分类任务的损失函数不变

    # 为行级任务使用焦点损失
    line_level_loss_fn = FocalLoss(
        alpha=0.25,  # 可以调整这个参数来平衡正负样本
        gamma=2.0,  # 可以调整这个参数来调节难易样本的权重
        reduction='mean'
    )

    # 合并优化器和调度器
    optimizers = {
        "classification": classification_optimizer,
        "line_level": line_level_optimizer
    }

    schedulers = {
        "classification": classification_scheduler,
        "line_level": line_level_scheduler
    }

    loss_fns = {
        "classification": classification_loss_fn,
        "line_level": line_level_loss_fn
    }

    # 合并数据加载器
    dataloaders = {
        "classification": classification_train_dataloader,
        "line_level": line_level_train_dataloader
    }

    # 初始化早停变量
    early_stopping = {
        "best_score": float('inf'),
        "counter": 0,
        "patience": config.patience,
        "best_epoch": 0
    }

    # 跟踪训练历史
    history = {
        "train_losses": [],
        "val_losses": [],
        "val_accuracies": [],
        "task_losses": {
            "classification": [],
            "line_level": []
        },
        "task_weights": {
            "classification": [],
            "line_level": []
        }
    }

    # 根据是否使用交替训练策略选择训练方法
    if config.use_alternate_training:
        # 使用交替训练策略
        logger.info(f"开始使用 {config.alternate_strategy} 交替训练策略进行训练...")

        # 根据不同的交替策略训练模型
        if config.alternate_strategy == "batch":
            # 按批次交替训练
            for epoch in range(config.num_epochs):
                logger.info(f"Epoch {epoch + 1}/{config.num_epochs} - 批次交替训练")

                # 训练一个epoch
                train_metrics = train_batch_alternate(
                    model, dataloaders, optimizers, schedulers, loss_fns, config
                )

                # 记录训练损失
                history["train_losses"].append(train_metrics["total_loss"])
                history["task_losses"]["classification"].append(train_metrics["classification_loss"])
                history["task_losses"]["line_level"].append(train_metrics["line_level_loss"])

                # 验证
                val_metrics = validate(model, classification_valid_dataloader, classification_loss_fn, config)
                history["val_losses"].append(val_metrics["classification_loss"])
                history["val_accuracies"].append(val_metrics["classification_accuracy"])

                # 获取任务权重
                task_weights = torch.softmax(model.log_task_weights, dim=0).detach().cpu().numpy()
                history["task_weights"]["classification"].append(task_weights[0])
                history["task_weights"]["line_level"].append(task_weights[1])

                # 保存当前模型
                save_model_with_config(model, config, epoch, val_metrics)

                # 打印信息
                logger.info(f"Epoch {epoch + 1} - 训练损失: {train_metrics['total_loss']:.4f}, "
                            f"验证损失: {val_metrics['classification_loss']:.4f}, "
                            f"验证准确率: {val_metrics['classification_accuracy']:.4f}")
                logger.info(f"任务权重: Classification={task_weights[0]:.4f}, Line Level={task_weights[1]:.4f}")

                # 早停检查
                if val_metrics["classification_loss"] < early_stopping["best_score"] - config.min_delta:
                    logger.info(
                        f"验证损失从 {early_stopping['best_score']:.4f} 改善到 {val_metrics['classification_loss']:.4f}")
                    early_stopping["best_score"] = val_metrics["classification_loss"]
                    early_stopping["counter"] = 0
                    early_stopping["best_epoch"] = epoch + 1

                    # 保存最佳模型
                    save_model_with_config(model, config, epoch, val_metrics, best=True)
                else:
                    early_stopping["counter"] += 1
                    logger.info(f"验证损失未改善。早停计数: {early_stopping['counter']}/{early_stopping['patience']}")

                    if early_stopping["counter"] >= early_stopping["patience"]:
                        logger.info(f"早停触发! 在第 {epoch + 1} 轮训练后停止")
                        break

        elif config.alternate_strategy == "epoch":
            # 按周期交替训练 - 一个epoch只训练一个任务
            for epoch in range(config.num_epochs):
                # 确定本轮要训练的任务
                task_idx = epoch % len(config.epoch_alternate_order)
                current_task = config.epoch_alternate_order[task_idx]

                logger.info(f"Epoch {epoch + 1}/{config.num_epochs} - 训练任务: {current_task}")

                # 训练当前任务
                task_loss = train_single_task(
                    model, dataloaders[current_task], optimizers[current_task],
                    schedulers[current_task], loss_fns[current_task], config, current_task, tokenizer
                )

                # 更新历史记录
                history["train_losses"].append(task_loss)
                history["task_losses"][current_task].append(task_loss)
                history["task_losses"]["classification" if current_task == "line_level" else "line_level"].append(None)

                # 验证 - 仅使用分类验证集
                val_metrics = validate(model, classification_valid_dataloader, classification_loss_fn, config)
                history["val_losses"].append(val_metrics["classification_loss"])
                history["val_accuracies"].append(val_metrics["classification_accuracy"])

                # 尝试获取当前任务权重
                task_weights = torch.softmax(model.log_task_weights, dim=0).detach().cpu().numpy()
                history["task_weights"]["classification"].append(task_weights[0])
                history["task_weights"]["line_level"].append(task_weights[1])

                # 保存当前epoch的模型
                save_model_with_config(model, config, epoch, val_metrics)

                logger.info(f"Epoch {epoch + 1} - 任务: {current_task} - 训练损失: {task_loss:.4f}, "
                            f"验证损失: {val_metrics['classification_loss']:.4f}, "
                            f"验证准确率: {val_metrics['classification_accuracy']:.4f}")
                logger.info(f"任务权重: Classification={task_weights[0]:.4f}, Line Level={task_weights[1]:.4f}")

                # 早停检查
                if val_metrics["classification_loss"] < early_stopping["best_score"] - config.min_delta:
                    logger.info(
                        f"验证损失从 {early_stopping['best_score']:.4f} 改善到 {val_metrics['classification_loss']:.4f}")
                    early_stopping["best_score"] = val_metrics["classification_loss"]
                    early_stopping["counter"] = 0
                    early_stopping["best_epoch"] = epoch + 1

                    # 保存最佳模型
                    save_model_with_config(model, config, epoch, val_metrics, best=True)
                else:
                    early_stopping["counter"] += 1
                    logger.info(f"验证损失未改善。早停计数: {early_stopping['counter']}/{early_stopping['patience']}")

                    if early_stopping["counter"] >= early_stopping["patience"]:
                        logger.info(f"早停触发! 在第 {epoch + 1} 轮训练后停止")
                        break

        elif config.alternate_strategy == "progressive":
            # 渐进式训练策略
            current_epoch = 0

            # 阶段1: 仅分类任务
            logger.info(
                f"阶段 1/{len(config.progressive_epochs)}: 仅分类任务 ({config.progressive_epochs['classification_only']} epochs)")
            for _ in range(config.progressive_epochs["classification_only"]):
                current_epoch += 1
                logger.info(f"Epoch {current_epoch}/{config.num_epochs} - 训练任务: classification")

                task_loss = train_single_task(
                    model, dataloaders["classification"], optimizers["classification"],
                    schedulers["classification"], loss_fns["classification"], config, "classification", tokenizer
                )

                # 更新历史记录
                history["train_losses"].append(task_loss)
                history["task_losses"]["classification"].append(task_loss)
                history["task_losses"]["line_level"].append(None)

                # 验证
                val_metrics = validate(model, classification_valid_dataloader, classification_loss_fn, config)
                history["val_losses"].append(val_metrics["classification_loss"])
                history["val_accuracies"].append(val_metrics["classification_accuracy"])

                # 任务权重
                task_weights = torch.softmax(model.log_task_weights, dim=0).detach().cpu().numpy()
                history["task_weights"]["classification"].append(task_weights[0])
                history["task_weights"]["line_level"].append(task_weights[1])

                # 保存模型
                save_model_with_config(model, config, current_epoch - 1, val_metrics)

                logger.info(f"Epoch {current_epoch} - 训练损失: {task_loss:.4f}, "
                            f"验证损失: {val_metrics['classification_loss']:.4f}, "
                            f"验证准确率: {val_metrics['classification_accuracy']:.4f}")
                logger.info(f"任务权重: Classification={task_weights[0]:.4f}, Line Level={task_weights[1]:.4f}")

                # 早停检查
                if val_metrics["classification_loss"] < early_stopping["best_score"] - config.min_delta:
                    logger.info(
                        f"验证损失从 {early_stopping['best_score']:.4f} 改善到 {val_metrics['classification_loss']:.4f}")
                    early_stopping["best_score"] = val_metrics["classification_loss"]
                    early_stopping["counter"] = 0
                    early_stopping["best_epoch"] = current_epoch

                    # 保存最佳模型
                    save_model_with_config(model, config, current_epoch - 1, val_metrics, best=True)
                else:
                    early_stopping["counter"] += 1
                    logger.info(f"验证损失未改善。早停计数: {early_stopping['counter']}/{early_stopping['patience']}")

                    if early_stopping["counter"] >= early_stopping["patience"]:
                        logger.info(f"早停触发! 在第 {current_epoch} 轮训练后停止")
                        break

            # 检查是否已经早停
            if early_stopping["counter"] < early_stopping["patience"]:
                # 后续训练阶段...
                # 这里可以添加更多渐进式训练阶段的代码，每个阶段都需要添加类似的早停逻辑
                # 为简化，此处省略
                pass
    else:
        # 使用联合训练策略（同时训练所有任务）
        logger.info("开始使用联合训练策略（同时训练所有任务）...")

        for epoch in range(config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs} - 联合训练")

            # 训练一个epoch
            train_metrics = train_joint(
                model, dataloaders, optimizers, schedulers, loss_fns, config, tokenizer
            )

            # 记录训练损失
            history["train_losses"].append(train_metrics["total_loss"])
            history["task_losses"]["classification"].append(train_metrics["classification_loss"])
            history["task_losses"]["line_level"].append(train_metrics["line_level_loss"])

            # 验证
            val_metrics = validate(model, classification_valid_dataloader, classification_loss_fn, config)
            history["val_losses"].append(val_metrics["classification_loss"])
            history["val_accuracies"].append(val_metrics["classification_accuracy"])

            # 获取任务权重
            task_weights = torch.softmax(model.log_task_weights, dim=0).detach().cpu().numpy()
            history["task_weights"]["classification"].append(task_weights[0])
            history["task_weights"]["line_level"].append(task_weights[1])

            # 保存当前模型
            save_model_with_config(model, config, epoch, val_metrics)

            # 打印信息
            logger.info(f"Epoch {epoch + 1} - 训练损失: {train_metrics['total_loss']:.4f}, "
                        f"验证损失: {val_metrics['classification_loss']:.4f}, "
                        f"验证准确率: {val_metrics['classification_accuracy']:.4f}")
            logger.info(f"任务权重: Classification={task_weights[0]:.4f}, Line Level={task_weights[1]:.4f}")
            
            # 记录实际使用的任务权重（如果有的话）
            if 'classification_weight' in train_metrics:
                logger.info(f"实际损失权重: Classification={train_metrics['classification_weight']:.4f}, "
                           f"Line Level={train_metrics['line_level_weight']:.4f}")

            # 早停检查
            if val_metrics["classification_loss"] < early_stopping["best_score"] - config.min_delta:
                logger.info(
                    f"验证损失从 {early_stopping['best_score']:.4f} 改善到 {val_metrics['classification_loss']:.4f}")
                early_stopping["best_score"] = val_metrics["classification_loss"]
                early_stopping["counter"] = 0
                early_stopping["best_epoch"] = epoch + 1

                # 保存最佳模型
                save_model_with_config(model, config, epoch, val_metrics, best=True)
            else:
                early_stopping["counter"] += 1
                logger.info(f"验证损失未改善。早停计数: {early_stopping['counter']}/{early_stopping['patience']}")

                if early_stopping["counter"] >= early_stopping["patience"]:
                    logger.info(f"早停触发! 在第 {epoch + 1} 轮训练后停止")
                    break

    # 打印最佳模型信息
    logger.info(
        f"训练完成! 最佳模型来自第 {early_stopping['best_epoch']} 轮, 验证损失: {early_stopping['best_score']:.4f}")

    # 绘制训练历史
    plot_training_history(history, config.output_dir)

    logger.info("训练完成!")
    return model, history


# 修改绘制训练历史的函数，添加验证损失和准确率
def plot_training_history(history, output_dir):
    """绘制训练历史曲线"""
    plt.figure(figsize=(15, 15))

    # 绘制训练损失曲线
    plt.subplot(3, 2, 1)
    plt.plot(history['train_losses'], 'b-', label='训练损失')
    if 'val_losses' in history and history['val_losses']:
        plt.plot(history['val_losses'], 'r-', label='验证损失')
    plt.title('损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制验证准确率曲线
    if 'val_accuracies' in history and history['val_accuracies']:
        plt.subplot(3, 2, 2)
        plt.plot(history['val_accuracies'], 'g-', label='验证准确率')
        plt.title('验证准确率')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    # 绘制任务损失曲线
    plt.subplot(3, 2, 3)
    plt.plot(history['task_losses']['classification'], 'g-', label='分类损失')
    plt.plot(history['task_losses']['line_level'], 'c-', label='行级损失')
    plt.title('任务特定损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制任务权重曲线
    plt.subplot(3, 2, 4)
    plt.plot(history['task_weights']['classification'], 'g-', label='分类权重')
    plt.plot(history['task_weights']['line_level'], 'c-', label='行级权重')
    plt.title('任务权重')
    plt.xlabel('Epochs')
    plt.ylabel('Weight')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


# 解析命令行参数的函数
def parse_args():
    parser = argparse.ArgumentParser(description="多任务漏洞检测训练脚本")
    
    # 任务权重相关参数
    parser.add_argument("--classification_weight", type=float, default=2.0,
                       help="分类任务权重 (默认: 2.0)")
    parser.add_argument("--line_level_weight", type=float, default=1.0,
                       help="行级任务权重 (默认: 1.0)")
    parser.add_argument("--normalize_weights", action="store_true", default=True,
                       help="是否归一化任务权重 (默认: True)")
    parser.add_argument("--no_normalize_weights", action="store_false", dest="normalize_weights",
                       help="不归一化任务权重")
    
    # 动态权重参数
    parser.add_argument("--use_dynamic_weights", action="store_true", default=False,
                       help="启用动态权重调整")
    parser.add_argument("--dynamic_weight_strategy", type=str, default="uncertainty",
                       choices=["uncertainty", "loss_ratio", "gradient_norm"],
                       help="动态权重策略 (默认: uncertainty)")
    
    # 训练模式参数
    parser.add_argument("--alternate_training", action="store_true", default=False,
                       help="使用交替训练策略")
    parser.add_argument("--alternate_strategy", type=str, default="epoch",
                       choices=["batch", "epoch", "progressive"],
                       help="交替训练策略 (默认: epoch)")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=2,
                       help="批次大小 (默认: 2)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="学习率 (默认: 1e-5)")
    parser.add_argument("--num_epochs", type=int, default=20,
                       help="训练轮数 (默认: 20)")
    parser.add_argument("--output_dir", type=str, default="multi_task_alternate_output_1",
                       help="输出目录 (默认: multi_task_alternate_output_1)")
    
    # PGD参数
    parser.add_argument("--no_pgd", action="store_true", default=False,
                       help="禁用PGD对抗训练")
    parser.add_argument("--pgd_epsilon", type=float, default=0.03,
                       help="PGD扰动大小 (默认: 0.03)")
    
    return parser.parse_args()


# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置并应用命令行参数
    config = Config()
    
    # 应用权重相关参数
    config.classification_weight = args.classification_weight
    config.line_level_weight = args.line_level_weight
    config.normalize_task_weights = args.normalize_weights
    config.use_dynamic_weights = args.use_dynamic_weights
    config.dynamic_weight_strategy = args.dynamic_weight_strategy
    
    # 应用训练参数
    config.use_alternate_training = args.alternate_training
    config.alternate_strategy = args.alternate_strategy
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.output_dir = args.output_dir
    
    # 应用PGD参数
    config.use_pgd = not args.no_pgd
    config.pgd_epsilon = args.pgd_epsilon
    
    logger.info(f"设备: {config.device}")
    logger.info(f"命令行参数已应用到配置")

    # 输出训练模式
    if config.use_alternate_training:
        logger.info(f"使用交替训练策略: {config.alternate_strategy}")
    else:
        logger.info("使用联合训练策略（同时训练所有任务）")

    # 输出PGD攻击配置
    if config.use_pgd:
        logger.info(
            f"启用PGD对抗训练，epsilon={config.pgd_epsilon}, alpha={config.pgd_alpha}, 步数={config.pgd_n_steps}")
    else:
        logger.info("未启用PGD对抗训练")
    
    # 输出任务权重配置
    logger.info(f"任务权重配置:")
    logger.info(f"  分类任务权重: {config.classification_weight}")
    logger.info(f"  行级任务权重: {config.line_level_weight}")
    logger.info(f"  权重归一化: {config.normalize_task_weights}")
    if config.use_dynamic_weights:
        logger.info(f"  启用动态权重调整，策略: {config.dynamic_weight_strategy}")
    else:
        logger.info(f"  使用静态权重")

    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        # 将配置对象转换为字典（排除非基本类型）
        config_dict = {k: v for k, v in vars(config).items()
                       if isinstance(v, (int, float, str, bool, list, dict))
                       or v is None}
        config_dict['device'] = str(config.device)
        json.dump(config_dict, f, indent=4)

    # 训练模型
    model, history = train(config)
    logger.info("训练完成！")


if __name__ == "__main__":
    main()