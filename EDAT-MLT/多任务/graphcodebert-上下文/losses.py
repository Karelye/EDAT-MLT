import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    焦点损失函数实现，特别适用于不平衡二分类问题，如行级漏洞检测。
    
    焦点损失的公式: FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        初始化焦点损失
        
        参数:
            alpha: 正样本的权重系数，用于解决类别不平衡
            gamma: 调制因子，减少容易样本的权重，增加困难样本的权重
            reduction: 'none'|'mean'|'sum' 指定损失如何汇总
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算焦点损失
        
        参数:
            inputs: 形状为 [N, C] 的模型输出，未经过softmax
            targets: 形状为 [N] 的目标类别索引
        """
        # 获取批次大小和类别数
        N, C = inputs.shape
        
        # 计算交叉熵损失（不做reduction）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 获取模型对正确类别的预测概率
        pt = torch.exp(-ce_loss)
        
        # 计算焦点权重
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用alpha权重
        if self.alpha is not None:
            # 为正样本使用alpha，为负样本使用1-alpha
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        # 计算最终的焦点损失
        loss = focal_weight * ce_loss
        
        # 根据reduction参数汇总损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 