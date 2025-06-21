# 特征控制评估指南

## 概述

该功能允许你在评估时控制模型使用的专家特征数量，而不是使用全部25个特征。这对于特征重要性分析、模型鲁棒性测试和消融研究非常有用。

## 专家特征说明

当前系统使用25维专家特征，包括：

### 词汇特征 (17个，索引0-16)
- 0: 代码行长度
- 1: 操作符数量 (+, -, *, /, ==, != 等)
- 2: 字符串字面量数量
- 3: 数字字面量数量
- 4: 标识符数量
- 5: 关键字数量 (if, for, while, return 等)
- 6: 函数调用数量
- 7: 变量赋值数量
- 8: 比较操作数量
- 9: 逻辑操作数量 (&&, ||, ! 等)
- 10: 算术操作数量
- 11: 位操作数量
- 12: 内存操作数量 (malloc, free 等)
- 13: 输入输出操作数量
- 14: 指针操作数量
- 15: 数组访问数量
- 16: 注释数量

### 语法特征 (4个，索引17-20)
- 17: 是否以控制流开始
- 18: 是否为函数定义行
- 19: 是否为返回语句
- 20: 嵌套深度

### 其他特征 (4个，索引21-24)
- 21-23: 预留的扩展特征
- 24: 相对行号（位置特征）

## 使用方法

### 基本用法

```bash
# 使用全部25个特征（默认）
python multi_task_evaluate.py

# 只使用前10个特征
python multi_task_evaluate.py --use_limited_features --num_features 10

# 只使用前5个最重要的特征
python multi_task_evaluate.py --use_limited_features --num_features 5 --feature_strategy important
```

### 命令行参数详解

#### 特征控制参数
- `--use_limited_features`: 启用特征数量限制
- `--num_features N`: 使用N个特征（默认25）
- `--feature_strategy {first,random,important}`: 特征选择策略
- `--feature_seed SEED`: 随机选择的种子（当strategy=random时）

#### 其他参数
- `--model_path PATH`: 指定模型文件路径
- `--output_dir DIR`: 输出目录
- `--batch_size N`: 批次大小
- `--no_noise`: 禁用评估干扰
- `--noise_intensity FLOAT`: 噪声强度

### 特征选择策略

#### 1. "first" 策略（默认）
选择前N个特征（按索引顺序）
```bash
python multi_task_evaluate.py --use_limited_features --num_features 10 --feature_strategy first
```

#### 2. "random" 策略
随机选择N个特征
```bash
python multi_task_evaluate.py --use_limited_features --num_features 10 --feature_strategy random --feature_seed 42
```

#### 3. "important" 策略
按预定义的重要性顺序选择N个特征
```bash
python multi_task_evaluate.py --use_limited_features --num_features 10 --feature_strategy important
```

重要性顺序（由高到低）：
1. 词汇特征中的核心特征 (0-4): 代码长度、操作符、字面量、标识符
2. 语法特征 (17-20): 控制流、函数定义、返回语句、嵌套深度
3. 词汇特征中的次要特征 (5-9): 关键字、函数调用等
4. 位置特征 (24): 相对行号
5. 其余特征

## 实验示例

### 消融研究
测试不同特征数量对性能的影响：

```bash
# 测试仅使用5个最重要特征的效果
python multi_task_evaluate.py --use_limited_features --num_features 5 --feature_strategy important

# 测试仅使用10个最重要特征的效果
python multi_task_evaluate.py --use_limited_features --num_features 10 --feature_strategy important

# 测试仅使用15个最重要特征的效果
python multi_task_evaluate.py --use_limited_features --num_features 15 --feature_strategy important
```

### 特征重要性分析
比较不同特征选择策略的效果：

```bash
# 前10个特征
python multi_task_evaluate.py --use_limited_features --num_features 10 --feature_strategy first

# 随机10个特征
python multi_task_evaluate.py --use_limited_features --num_features 10 --feature_strategy random

# 重要的10个特征
python multi_task_evaluate.py --use_limited_features --num_features 10 --feature_strategy important
```

### 鲁棒性测试
测试模型在特征缺失情况下的表现：

```bash
# 模拟特征传感器故障，只有少量特征可用
python multi_task_evaluate.py --use_limited_features --num_features 3 --feature_strategy important

# 模拟随机特征丢失
python multi_task_evaluate.py --use_limited_features --num_features 8 --feature_strategy random --feature_seed 123
```

## 输出说明

启用特征控制时，评估脚本会输出：

```
启用特征限制:
  使用特征数量: 10
  选择策略: important
按重要性选择10个特征: 索引[0, 1, 2, 3, 4, 17, 18, 19, 20, 5]
特征维度从10填充到25
```

这表示：
- 使用了10个特征
- 按重要性策略选择
- 具体使用的特征索引
- 为了与模型兼容，特征被填充到25维（多余位置用0填充）

## 注意事项

1. **模型兼容性**: 特征选择后会自动填充到25维以保持与预训练模型的兼容性
2. **性能影响**: 使用较少特征可能会降低模型性能，这是正常的
3. **实验记录**: 建议记录每次实验的特征配置以便后续分析
4. **种子固定**: 使用random策略时建议固定种子以确保实验可重复

## 典型实验流程

```bash
# 1. 基线评估（全特征）
python multi_task_evaluate.py > baseline_results.txt

# 2. 重要特征评估
python multi_task_evaluate.py --use_limited_features --num_features 10 --feature_strategy important > important_10_results.txt

# 3. 随机特征评估
python multi_task_evaluate.py --use_limited_features --num_features 10 --feature_strategy random --feature_seed 42 > random_10_results.txt

# 4. 极少特征评估
python multi_task_evaluate.py --use_limited_features --num_features 3 --feature_strategy important > minimal_results.txt
```

这样可以系统地分析特征数量和选择策略对模型性能的影响。 