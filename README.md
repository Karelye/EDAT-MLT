# EDAT-MLT
# 多任务代码漏洞检测模型（含PGD对抗训练）

## 项目简介

本项目基于CodeBERT构建了一个多任务学习模型，联合完成以下任务：

- **行级漏洞预测**（Line-level Vulnerability Detection）：识别代码中可能存在漏洞的行（极度不平衡二分类任务）
- **代码分类预测**（Vulnerability Type Prediction）：对整个代码函数进行多类漏洞类型分类

为增强模型鲁棒性，引入了**PGD对抗训练**方法，在训练阶段对输入的标识符嵌入添加扰动，提升模型对语义变异的泛化能力。

---

## 模型结构概览

```
输入代码 → CodeBERT 编码器 → 多专家混合门控 (MMoE) → 
        ↙                    ↘
    分类特化网络         行级特化网络
        ↓                    ↓
  漏洞类型分类输出      漏洞行预测输出
```

- **编码器**：使用预训练模型CodeBERT提取代码向量
- **MMoE**：共享6个专家网络，结合门控机制实现任务特化
- **分类/行级网络**：各自独立的特化网络
- **损失函数**：
  - 分类任务：交叉熵或焦点损失
  - 行级任务：焦点损失 + 正负样本采样平衡策略

---

## 对抗训练（PGD）

### 目标
增强模型对微小语义变异（如变量名修改）的鲁棒性。

### 实现方式
- 在嵌入空间中对标识符位置添加扰动
- 支持仅对行级任务中的正样本（漏洞行）使用对抗训练
- 可调参数包括：扰动强度（`epsilon`）、步长（`alpha`）、迭代次数（`n_steps`）

### 使用配置（示例）：

```python
use_pgd = True
pgd_epsilon = 0.02
pgd_alpha = 0.002
pgd_n_steps = 3
pgd_line_level_positive_only = True
```

---

## 使用方法

### 模型训练

```bash
python multi_task_train_alternate.py
```

高级训练选项：

```bash
python multi_task_train_alternate.py --progressive_unfreezing
```

### 模型评估

```bash
python multi_task_evaluate.py
```

---

## 推荐配置

- `expert_num`: 6
- `alternate_strategy`: "batch" / "epoch" / "progressive"
- `loss_types`: 支持焦点损失
- `use_pgd`: 是否启用对抗训练
- `use_sampling`: 是否启用正负样本采样

