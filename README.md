# Improving Vulnerability Type Prediction and Line-Level Detection via Adversarial Training-Based Data Augmentation and Multi-Task Learning

This repository contains the official implementation of the paper "Improving vulnerability type prediction and line-level detection via adversarial training-based data augmentation and multi-task learning".

## Core Technical Features

1. **Embedding-Driven Adversarial Training (EDAT)**: Introduces adversarial perturbations at the code identifier embedding layer to enhance model robustness
2. **Multi-Task Learning Architecture (MTL)**: Simultaneously learns vulnerability type prediction and line-level detection tasks
3. **Unified Framework**: Combines EDAT with MTL to achieve collaborative optimization of both tasks

## Reproducing Three Research Questions (RQ) Experiments

This codebase is specifically designed to validate the key research questions proposed in the paper. Here are the detailed reproduction guides:

### RQ1: Can our proposed method outperform baseline methods on vulnerability type prediction and line-level vulnerability detection?

**Experimental Purpose**: Validate the performance improvement of the complete EDAT-MTL method compared to baseline methods

**Reproduction Steps**:
```bash
# Run complete baseline comparison experiments
python multi_task_train_alternate.py --config_name="full_comparison"

# Evaluate trained models
python multi_task_evaluate.py --model_path="multi_task_alternate_output_3/best_multi_task_model.pt"
```

**Expected Results**:
- VTP task: Comprehensive improvement over baselines in accuracy, precision, recall, and F1-score
- LVD task: Significant improvements in Top-10 accuracy, IFA (Initial False Alarms), and Top-20% LOC recall
- Particularly notable improvement in identifying rare vulnerability types (such as CWE-416, CWE-476, etc.)

**Corresponding Paper Tables**: Table 3 (VTP results), Table 4 (LVD results)

### RQ2: Can embedding-driven data augmentation improve the performance of vulnerability type prediction and line-level vulnerability detection?

**Experimental Purpose**: Validate the effectiveness of the EDAT module through ablation studies

**Reproduction Steps**:
```bash
# Run EDAT ablation experiment (disable PGD adversarial training)
python multi_task_train_alternate.py --use_pgd=False --output_dir="output_no_pgd"

# Run complete EDAT method
python multi_task_train_alternate.py --use_pgd=True --output_dir="output_with_pgd"

# Compare performance of both models
python multi_task_evaluate.py --model_path="output_no_pgd/best_multi_task_model.pt" --output_suffix="no_pgd"
python multi_task_evaluate.py --model_path="output_with_pgd/best_multi_task_model.pt" --output_suffix="with_pgd"
```

**Key Parameter Configuration**:
```python
# EDAT-related parameters (in multi_task_train_alternate.py)
use_pgd = True          # Whether to enable PGD adversarial training
pgd_epsilon = 0.03      # PGD perturbation magnitude
pgd_alpha = 1e-2        # PGD step size
pgd_n_steps = 3         # PGD iteration steps
```

**Expected Results**:
- Performance improvement in both tasks after enabling EDAT
- Enhanced recognition capability for difficult samples and rare categories
- Improved model robustness, less sensitive to input variations

**Corresponding Paper Tables**: Table 5 (VTP ablation study), Table 6 (LVD ablation study)

### RQ3: Can multi-task learning improve the performance of vulnerability type prediction and line-level vulnerability detection?

**Experimental Purpose**: Validate the performance advantages of MTL compared to single-task learning

**Reproduction Steps**:
```bash
# Run single-task learning (classification task only)
python multi_task_train_alternate.py --use_alternate_training=False --alternate_strategy="classification_only"

# Run single-task learning (line-level task only)
python multi_task_train_alternate.py --use_alternate_training=False --alternate_strategy="line_level_only"

# Run multi-task learning
python multi_task_train_alternate.py --use_alternate_training=True --alternate_strategy="joint"
```

**Key Architecture Components**:
- **MMoE Layer**: `MMOELayer` class in `multi_task_model.py`, containing 6 expert networks
- **Task-Specific Encoder**: `TaskSpecificEncoder` class, providing specialized feature transformation for each task
- **Shared Representation Learning**: Through shared CodeBERT encoder and expert feature fusion layer

**Expected Results**:
- MTL shows performance improvement over single-task learning on both tasks
- Knowledge sharing between tasks brings complementary advantages
- Particularly effective for vulnerability types with scarce data

**Corresponding Paper Tables**: Table 7 (MTL vs Single-task comparison)

## Environment Setup and Installation

### System Requirements
- Python 3.8+
- CUDA 11.0+ (GPU training recommended)
- Memory: At least 16GB RAM

### Installation Steps

1. **Install Dependencies**
```bash
pip install torch transformers scikit-learn pandas numpy tqdm matplotlib tree-sitter-c
```

2. **Download Pre-trained Models**
- Download GraphCodeBERT pre-trained model to specified path

3. **Prepare Dataset**
- Use BigVul dataset in JSONL format
- Classification task data: Contains function code and CWE labels
- Line-level task data: Contains line-level code and vulnerability labels

## Quick Start

### Basic Usage

1. **Train Multi-task Model**
```bash
python multi_task_train_alternate.py
```

2. **Evaluate Model Performance**
```bash
python multi_task_evaluate.py
```

## Experimental Results

### Main Performance Metrics

#### Vulnerability Type Prediction (VTP)
| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| CodeBERT | 0.621 | 0.603 | 0.588 | 0.595 |
| GraphCodeBERT | 0.643 | 0.625 | 0.612 | 0.618 |
| **EDAT-MTL (Ours)** | **0.678** | **0.661** | **0.649** | **0.655** |

#### Line-Level Vulnerability Detection (LVD)
| Method | Top-10 Accuracy | IFA | Top-20% LOC Recall |
|--------|----------------|-----|-------------------|
| LineVul | 0.325 | 12.4 | 0.445 |
| VulDeePecker | 0.298 | 15.2 | 0.412 |
| **EDAT-MTL (Ours)** | **0.387** | **9.8** | **0.503** |

### Ablation Study Results

| Components | VTP F1 | LVD Top-10 | Improvement |
|------------|--------|------------|-------------|
| Baseline (no EDAT, no MTL) | 0.618 | 0.325 | - |
| +EDAT | 0.635 | 0.348 | +2.7%, +7.1% |
| +MTL | 0.642 | 0.356 | +3.9%, +9.5% |
| **+EDAT+MTL** | **0.655** | **0.387** | **+6.0%, +19.1%** |

## Project Structure

```
graphcodebert-使用新数据/
├── multi_task_model.py          # Multi-task model architecture
├── multi_task_train_alternate.py # Training script
├── multi_task_evaluate.py       # Evaluation script
├── multi_task_data.py           # Data loader
├── multi_task_pgd.py           # PGD adversarial training implementation
└── losses.py                   # Loss functions
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Issues and Pull Requests are welcome! Before contributing code, please ensure:

1. Code passes all tests
2. Follows project code style
3. Adds necessary documentation and comments 
