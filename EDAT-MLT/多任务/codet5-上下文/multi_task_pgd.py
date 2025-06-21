import torch
import torch.nn as nn
import re
import logging

import tree_sitter_c
from tree_sitter import Language, Parser

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

C_LANGUAGE = Language(tree_sitter_c.language())
parser = Parser()
parser.language = C_LANGUAGE
query = C_LANGUAGE.query("""(identifier) @identifier""")
logger.info("成功加载Tree-sitter C语言解析器")


def extract_identifiers_from_code(code):
    """提取代码中的标识符"""
    if not isinstance(code, bytes):
        try:
            code = code.encode('utf-8')
        except AttributeError:
            logger.warning(f"提取标识符时出错: code必须是字符串或字节，但收到了 {type(code)}")
            return []

    if parser is None or query is None:
        # 备用方案：使用简单的正则表达式
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code.decode('utf-8', errors='ignore'))
        return identifiers

    try:
        tree = parser.parse(code)
        root_node = tree.root_node
        matches = query.matches(root_node)
        identifiers = []
        for match in matches:
            node = match[0]  # 在原始实现中是match[0]
            if hasattr(node, 'start_byte') and hasattr(node, 'end_byte'):
                identifier = code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
                identifiers.append(identifier)
        return identifiers
    except Exception as e:
        logger.warning(f"提取标识符时出错: {e}")
        return []


class PGDAttack:
    """PGD对抗攻击实现，严格遵循train_PGD.py中的流程"""

    def __init__(self, model, emb_name=None):
        self.model = model
        self.emb_name = emb_name

        # 如果未指定嵌入层名称，尝试自动查找
        if not self.emb_name:
            for name, param in self.model.named_parameters():
                # 支持多种嵌入层命名模式：T5的shared权重，embed_tokens，RoBERTa的embeddings等
                if ('embeddings' in name or 'embed_tokens' in name or 'shared' in name) and param.requires_grad:
                    self.emb_name = name
                    logger.info(f"找到嵌入层参数：{name}")
                    break
            else:
                # 如果找不到，列出所有可用参数名称以便调试
                logger.error("无法找到嵌入层参数。以下是所有可用参数：")
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        logger.error(f"  - {name}: {param.shape}")
                raise ValueError("无法找到可训练的嵌入层参数。")

        logger.info(f"将对嵌入层 {self.emb_name} 应用PGD攻击")

    def compute_perturbation(self, epsilon=0.02, alpha=1e-2, n_steps=3):
        """计算对抗扰动，严格遵循原始实现"""
        # 获取嵌入权重 - 使用动态查找的嵌入层
        embeddings = dict(self.model.named_parameters())[self.emb_name]
        perturbation = torch.zeros_like(embeddings)

        for _ in range(n_steps):
            # 使用当前损失计算梯度
            grad = torch.autograd.grad(
                self.model.current_loss,  # 使用模型当前损失，与原始实现保持一致
                embeddings,
                retain_graph=True,
                create_graph=False
            )[0]

            # 基于梯度符号更新扰动
            perturbation += alpha * torch.sign(grad)
            # 限制扰动范围
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)

        return perturbation

    def apply_perturbation(self, token_ids, perturbation, tokenizer, codes):
        """应用扰动到标识符的嵌入"""
        # 获取唯一token ids作为目标
        target_ids = token_ids.unique().flatten()

        # 获取代码中的标识符
        identifiers = []
        for code in codes:
            identifiers.extend(extract_identifiers_from_code(code))
        identifiers = list(set([id for id in identifiers if isinstance(id, str) and len(id) > 0]))

        # 如果没有找到标识符，返回空目标列表和空嵌入张量
        if not identifiers:
            empty_ids = torch.tensor([], device=token_ids.device, dtype=torch.long)
            empty_embeddings = torch.tensor([], device=token_ids.device)
            return empty_ids, empty_embeddings

        # 获取标识符的token ids
        identifier_tokens = tokenizer(
            identifiers,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None
        )

        # 收集所有标识符的token ids
        all_identifier_ids = []
        for ids in identifier_tokens["input_ids"]:
            all_identifier_ids.extend(ids)

        # 转换为tensor并移动到正确的设备
        all_identifier_ids = torch.tensor(
            list(set(all_identifier_ids)),
            device=token_ids.device
        )

        # 与输入token的ids取交集，确保只扰动在当前输入中且是标识符一部分的token
        final_target_ids = torch.tensor(
            [id.item() for id in target_ids if id.item() in all_identifier_ids.tolist()],
            device=token_ids.device
        )

        # 如果没有可扰动的目标，返回空目标列表和空嵌入张量
        if len(final_target_ids) == 0:
            empty_embeddings = torch.tensor([], device=token_ids.device)
            return final_target_ids, empty_embeddings

        # 保存原始嵌入
        with torch.no_grad():
            embeddings = dict(self.model.named_parameters())[self.emb_name]
            original_embeddings = embeddings[final_target_ids].clone()
            # 应用扰动
            embeddings[final_target_ids] += perturbation[final_target_ids]

        return final_target_ids, original_embeddings

    def restore_embeddings(self, target_ids, original_embeddings):
        """恢复原始嵌入"""
        if len(target_ids) > 0:
            with torch.no_grad():
                embeddings = dict(self.model.named_parameters())[self.emb_name]
                embeddings[target_ids] = original_embeddings


def get_pgd_loss(model, batch, task, loss_fn, pgd_attack, tokenizer, config):
    """
    使用PGD对抗训练计算损失

    参数:
        model: 多任务模型
        batch: 当前批次数据
        task: 当前任务 ('classification' 或 'line_level')
        loss_fn: 当前任务的损失函数
        pgd_attack: PGD攻击实例
        tokenizer: tokenizer用于解码
        config: 配置

    返回:
        total_loss: 结合原始损失和对抗损失的总损失
    """
    if task == "classification":
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['label'].to(config.device)

        # 正常前向传播
        logits = model(task=task, input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        # 计算对抗扰动
        perturbation = pgd_attack.compute_perturbation(
            epsilon=config.pgd_epsilon,
            alpha=config.pgd_alpha,
            n_steps=config.pgd_n_steps
        )

        # 解码输入 token IDs 以获取代码文本
        codes = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # 应用扰动
        target_ids, original_embeddings = pgd_attack.apply_perturbation(
            input_ids, perturbation, tokenizer, codes
        )

        if len(target_ids) > 0:
            # 使用扰动的嵌入进行前向传播
            logits_adv = model(task=task, input_ids=input_ids, attention_mask=attention_mask)
            loss_adv = loss_fn(logits_adv, labels)

            # 恢复原始嵌入
            pgd_attack.restore_embeddings(target_ids, original_embeddings)

            # 合并损失
            total_loss = loss + loss_adv
        else:
            total_loss = loss

    elif task == "line_level":
        line_ids = batch['line_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        commit_features = batch['commit_features'].to(config.device)
        expert_features = batch['expert_features'].to(config.device)
        line_label = batch['line_label'].to(config.device)

        # 正常前向传播
        logits = model(
            task=task,
            line_ids=line_ids,
            attention_mask=attention_mask
        )
        loss = loss_fn(logits.view(-1, 2), line_label.view(-1))

        # 计算对抗扰动
        perturbation = pgd_attack.compute_perturbation(
            epsilon=config.pgd_epsilon,
            alpha=config.pgd_alpha,
            n_steps=config.pgd_n_steps
        )

        # 展平line_ids以便解码
        batch_size, seq_len, token_len = line_ids.shape
        flat_line_ids = line_ids.reshape(-1, token_len)

        # 解码输入token IDs以获取代码文本
        try:
            # 使用batch_decode确保正确处理
            codes = []
            for i in range(flat_line_ids.size(0)):
                # 过滤掉填充的token
                valid_ids = flat_line_ids[i][flat_line_ids[i] != 0]
                if len(valid_ids) > 0:
                    decoded = tokenizer.decode(valid_ids, skip_special_tokens=True)
                    codes.append(decoded)
                else:
                    codes.append("")
        except Exception as e:
            logger.warning(f"解码代码时出错: {e}")
            codes = ["" for _ in range(flat_line_ids.size(0))]

        # 应用扰动
        target_ids, original_embeddings = pgd_attack.apply_perturbation(
            flat_line_ids, perturbation, tokenizer, codes
        )

        if len(target_ids) > 0:
            # 使用扰动的嵌入进行前向传播
            logits_adv = model(
                task=task,
                line_ids=line_ids,
                attention_mask=attention_mask
            )
            loss_adv = loss_fn(logits_adv.view(-1, 2), line_label.view(-1))

            # 恢复原始嵌入
            pgd_attack.restore_embeddings(target_ids, original_embeddings)

            # 合并损失
            total_loss = loss + loss_adv
        else:
            total_loss = loss
    else:
        raise ValueError(f"不支持的任务类型: {task}")

    return total_loss 