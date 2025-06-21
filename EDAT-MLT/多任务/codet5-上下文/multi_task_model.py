import torch
import torch.nn as nn
from transformers import T5EncoderModel
import logging


class MMOELayer(nn.Module):
    """改进的多门混合专家层"""

    def __init__(self, input_dim, expert_num, expert_dim, task_num):
        super(MMOELayer, self).__init__()

        # 专家网络，增加中间激活层提升表达能力
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.LayerNorm(expert_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_dim, expert_dim),
                nn.LayerNorm(expert_dim),
                nn.ReLU()
            ) for _ in range(expert_num)
        ])

        # 任务特定的门控网络
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_num),
                nn.Softmax(dim=1)
            ) for _ in range(task_num)
        ])

        # 添加可学习的任务特定缩放因子，用于平衡不同任务
        self.task_scales = nn.Parameter(torch.ones(task_num))

        self.expert_num = expert_num
        self.task_num = task_num
        self.expert_dim = expert_dim

    def forward(self, x):
        # 计算每个专家的输出
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, expert_num, expert_dim]

        # 计算每个任务的门控权重
        gate_outputs = [gate(x).unsqueeze(-1) for gate in self.gates]  # [batch_size, expert_num, 1]

        # 计算每个任务的最终输出并应用任务特定缩放
        task_outputs = []
        for i, gate_output in enumerate(gate_outputs):
            # 应用门控权重
            task_output = torch.sum(expert_outputs * gate_output, dim=1)  # [batch_size, expert_dim]
            # 应用任务特定缩放
            task_output = task_output * torch.sigmoid(self.task_scales[i])
            task_outputs.append(task_output)

        return task_outputs, [g.squeeze(-1) for g in gate_outputs]  # 返回任务输出和门控权重


class TaskSpecificEncoder(nn.Module):
    """增强的任务特化编码层，带残差连接"""

    def __init__(self, hidden_size, nhead=8, dim_feedforward=2048, dropout=0.2, num_layers=2):
        super(TaskSpecificEncoder, self).__init__()

        # 使用多层转换器增强特征提取能力
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # 添加残差连接和层归一化
        self.residual_norm = nn.LayerNorm(hidden_size)

        # 特化层后的非线性变换
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.ReLU()
        )

    def forward(self, x, attention_mask=None):
        """
        输入：
            x: [batch_size, seq_len, hidden_size] 或 [batch_size, hidden_size]
            attention_mask: 可以是 [batch_size, seq_len] 或 [batch_size, seq_len, token_len]
        """
        # 处理不同维度的输入
        original_x = x
        if len(x.shape) == 2:  # [batch_size, hidden_size]
            # 将向量扩展为序列
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
            transformer_mask = None
        else:  # [batch_size, seq_len, hidden_size]
            if attention_mask is not None:
                # 处理注意力掩码的维度，确保是2D [batch_size, seq_len]
                if len(attention_mask.shape) == 3:
                    # 如果是3D掩码，计算每行的总和，为0的位置视为填充
                    # Transformer掩码: True表示填充位置，False表示有效位置
                    transformer_mask = (attention_mask.sum(dim=2) == 0)
                else:
                    # 如果已经是2D掩码，确保格式正确
                    # 对于Transformer，0/False表示有效位置，1/True表示填充位置
                    transformer_mask = (~attention_mask.bool() if attention_mask.dtype == torch.bool
                                        else attention_mask == 0)
            else:
                transformer_mask = None

        # 应用多层transformer编码
        encoded = x
        for layer in self.layers:
            encoded = layer(encoded, src_key_padding_mask=transformer_mask)

        # 应用残差连接和层归一化
        if len(x.shape) == 3 and len(original_x.shape) == 3:
            encoded = self.residual_norm(encoded + original_x)
        elif len(x.shape) == 3 and len(original_x.shape) == 2:
            encoded = self.residual_norm(encoded + original_x.unsqueeze(1))

        # 应用输出投影
        if len(x.shape) == 3 and x.size(1) == 1:
            # 如果是单一向量，去掉序列维度
            encoded = encoded.squeeze(1)
            output = self.output_projection(encoded)
        else:
            # 保留序列维度
            output = self.output_projection(encoded)

        return output


class MultiTaskVulnerabilityModel(nn.Module):
    """改进的多任务漏洞检测模型"""

    def __init__(self, pretrained_model_path, class_num_labels, line_num_labels,
                 expert_feature_dim=0, expert_num=4, expert_dim=512, max_codeline_length=50):
        super(MultiTaskVulnerabilityModel, self).__init__()

        # 添加current_loss属性用于PGD对抗训练
        self.current_loss = None
        
        # 共享的CodeT5编码器
        self.codet5 = T5EncoderModel.from_pretrained(pretrained_model_path)
        hidden_size = self.codet5.config.d_model

        # 任务特化层 - 使用增强的编码器
        self.classification_encoder = TaskSpecificEncoder(
            hidden_size=hidden_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.2,
            num_layers=2
        )

        self.line_level_encoder = TaskSpecificEncoder(
            hidden_size=hidden_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.2,
            num_layers=2
        )
        
        # 分类任务的特定组件
        self.classification_dropout = nn.Dropout(0.3)  # 增强dropout

        # 行级任务的特定组件
        self.line_level_dropout = nn.Dropout(0.3)  # 增强dropout

        # 增加专家数量和深度的MMOE层
        self.mmoe = MMOELayer(
            input_dim=hidden_size,
            expert_num=expert_num,
            expert_dim=expert_dim,
            task_num=2  # 分类和行级
        )

        # 分类头
        self.classification_head = nn.Linear(expert_dim, class_num_labels)

        # 行级预测头
        self.line_level_head = nn.Linear(expert_dim, line_num_labels)

        self.max_codeline_length = max_codeline_length

        # 使用对数参数初始化任务权重，用于动态平衡损失
        self.log_task_weights = nn.Parameter(torch.zeros(2))
        self.task_weight_ema = {"classification": 0.5, "line_level": 0.5}
        self.ema_alpha = 0.9

    def get_balanced_loss(self, classification_loss=None, line_level_loss=None):
        """计算平衡的多任务损失"""
        weights = torch.softmax(self.log_task_weights, dim=0)  # 转换为和为1的正数权重

        losses = []
        loss_values = []

        if classification_loss is not None:
            losses.append(weights[0] * classification_loss)
            loss_values.append(classification_loss.item())

        if line_level_loss is not None:
            losses.append(weights[1] * line_level_loss)
            loss_values.append(line_level_loss.item())

        # 更新任务损失的EMA值
        if classification_loss is not None:
            self.task_weight_ema["classification"] = self.ema_alpha * self.task_weight_ema["classification"] + \
                                                     (1 - self.ema_alpha) * classification_loss.item()
        if line_level_loss is not None:
            self.task_weight_ema["line_level"] = self.ema_alpha * self.task_weight_ema["line_level"] + \
                                                 (1 - self.ema_alpha) * line_level_loss.item()

        # 计算总损失
        if losses:
            return sum(losses), weights.detach().cpu().numpy(), loss_values
        else:
            return torch.tensor(0.0, device=self.log_task_weights.device), weights.detach().cpu().numpy(), []

    def get_gate_weights(self, task, **kwargs):
        """获取MMOE门控权重，用于分析"""
        if task == "classification":
            input_ids = kwargs['input_ids']
            attention_mask = kwargs['attention_mask']
            outputs = self.codet5(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state.mean(dim=1)  # T5使用平均池化
            encoded_output = self.classification_encoder(pooled_output)
            encoded_output = self.classification_dropout(encoded_output)
            _, gate_weights = self.mmoe(encoded_output)
            return gate_weights[0]  # 返回分类任务的门控权重
        elif task == "line_level":
            line_ids = kwargs['line_ids']
            attention_mask = kwargs['attention_mask']

            batch_size, seq_len, token_len = line_ids.shape

            # 获取每行的编码
            line_ids_flat = line_ids.view(-1, token_len)
            attention_mask_flat = attention_mask.view(-1, token_len)

            outputs = self.codet5(input_ids=line_ids_flat, attention_mask=attention_mask_flat)
            code_pooled_output = outputs.last_hidden_state.mean(dim=1)  # T5使用平均池化

            # 应用行级编码器
            line_attention_mask = (attention_mask.sum(dim=2) > 0).float()
            code_pooled_output = self.line_level_encoder(code_pooled_output, line_attention_mask)

            # 对每行获取门控权重
            all_gate_weights = []
            for i in range(seq_len):
                features = code_pooled_output[:, i, :]
                _, gate_weights = self.mmoe(features)
                all_gate_weights.append(gate_weights[1])  # 使用行级任务的门控权重

            # 平均所有行的门控权重
            avg_gate_weights = torch.stack(all_gate_weights, dim=0).mean(dim=0)
            return avg_gate_weights

        return None

    def forward_classification(self, input_ids, attention_mask):
        """分类任务的前向传播"""
        outputs = self.codet5(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # T5使用平均池化

        # 通过分类任务特化层
        encoded_output = self.classification_encoder(pooled_output)
        encoded_output = self.classification_dropout(encoded_output)

        # 将编码后的输出传递给MMOE
        task_outputs, _ = self.mmoe(encoded_output)
        classification_output = task_outputs[0]  # 任务0是分类

        # 分类头
        logits = self.classification_head(classification_output)
        return logits

    def forward_line_level(self, line_ids, attention_mask):
        """行级任务的前向传播"""
        logger = logging.getLogger(__name__)
        
        batch_size, seq_len, token_len = line_ids.shape
        # 现在seq_len固定为1，因为每个样本只包含一行代码
        assert seq_len == 1, f"期望seq_len为1，但得到{seq_len}"

        # 重塑以处理所有代码行 - 由于seq_len=1，这相当于去掉seq_len维度
        line_ids_flat = line_ids.view(batch_size, token_len)
        attention_mask_flat = attention_mask.view(batch_size, token_len)

        # 获取每行的CodeT5编码
        outputs = self.codet5(input_ids=line_ids_flat, attention_mask=attention_mask_flat)
        code_pooled_output = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size] T5使用平均池化
        
        # 由于只有一行，我们不需要序列级别的编码器，直接使用pooled output
        # 应用dropout
        code_encoded = self.line_level_dropout(code_pooled_output)

        # 直接应用MMOE，无需循环处理多行
        task_outputs, _ = self.mmoe(code_encoded)
        line_level_output = task_outputs[1]  # 任务1是行级

        # 行级预测头
        logits = self.line_level_head(line_level_output)  # [batch_size, num_labels]
        
        # 由于输出需要与原始格式匹配，添加seq_len维度
        logits = logits.unsqueeze(1)  # [batch_size, 1, num_labels]
        
        return logits

    def forward(self, task, **kwargs):
        """统一的前向传播接口"""
        if task == "classification":
            return self.forward_classification(**kwargs)
        elif task == "line_level":
            return self.forward_line_level(**kwargs)
        else:
            raise ValueError(f"不支持的任务类型: {task}")