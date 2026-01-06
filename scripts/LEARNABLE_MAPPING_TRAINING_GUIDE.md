# UPSR 可学习不确定度映射 - 分阶段训练指南

## 概述

本指南介绍如何使用三阶段训练策略来训练带有可学习不确定度映射的UPSR模型。

## 训练策略

### 阶段0：固定映射预训练 (100,000次迭代)
使用固定的不确定度映射训练基础模型，建立良好的初始状态。

**配置文件：** `002_UPSR_RealSR_x4.yml`

**特点：**
- 使用固定映射：`uncertainty_mapping.use_learnable: false`
- 完整训练net_g和其他组件
- 总迭代次数：200,000（建议使用前100,000次的checkpoint）

**训练命令：**
```bash
cd /root/project/UPSR_copy
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 \
    basicsr/train.py -opt options/002_UPSR_RealSR_x4.yml --launcher pytorch
```

**关键checkpoint：**
- `experiments/002_UPSR_RealSR_x4/models/net_g_100000.pth` - 用于后续阶段

---

### 阶段1：映射层专项训练 (5,000次迭代)
冻结net_g，只训练可学习的不确定度映射层。

**配置文件：** `003_UPSR_Stage1_MappingOnly_x4.yml`

**特点：**
- 加载阶段0的预训练模型：`pretrain_network_g: experiments/002_UPSR_RealSR_x4/models/net_g_100000.pth`
- 使用可学习映射：`uncertainty_mapping.use_learnable: true`
- 冻结net_g：`uncertainty_mapping.freeze_others_initially: true`
- 只优化映射层参数
- 较大学习率：`lr: 2e-4`
- 总迭代次数：5,000

**训练命令：**
```bash
cd /root/project/UPSR_copy
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 \
    basicsr/train.py -opt options/003_UPSR_Stage1_MappingOnly_x4.yml --launcher pytorch
```

**输出：**
- `experiments/003_UPSR_Stage1_MappingOnly_x4/models/net_g_5000.pth` - 用于阶段2
- `experiments/003_UPSR_Stage1_MappingOnly_x4/models/uncertainty_mapper_5000.pth` - 映射层权重

---

### 阶段2：联合精调 (50,000次迭代)
解冻所有参数，使用差异化学习率进行联合训练。

**配置文件：** `004_UPSR_Stage2_JointTrain_x4.yml`

**特点：**
- 加载阶段1的模型：
  - `pretrain_network_g: experiments/003_UPSR_Stage1_MappingOnly_x4/models/net_g_5000.pth`
  - `uncertainty_mapping.ckpt_path: experiments/003_UPSR_Stage1_MappingOnly_x4/models/uncertainty_mapper_5000.pth`
- 联合训练：`uncertainty_mapping.freeze_others_initially: false`
- 差异化学习率：
  - base_lr: 1e-4
  - net_g: 1e-4 × 0.5 = 5e-5 (较保守)
  - mapper: 1e-4 × 2.0 = 2e-4 (较激进)
- 总迭代次数：50,000

**训练命令：**
```bash
cd /root/project/UPSR_copy
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 \
    basicsr/train.py -opt options/004_UPSR_Stage2_JointTrain_x4.yml --launcher pytorch
```

**输出：**
- `experiments/004_UPSR_Stage2_JointTrain_x4/models/net_g_50000.pth` - 最终模型
- `experiments/004_UPSR_Stage2_JointTrain_x4/models/uncertainty_mapper_50000.pth` - 最终映射层

---

## 代码实现要点

### 1. 可学习映射层 (`uncertainty_mapping.py`)

```python
class LearnableUncertaintyMapping(nn.Module):
    """使用MLP实现的可学习不确定度映射"""
    def __init__(self, un_max=1.0, min_noise=0.0, hidden_dim=64, 
                 num_layers=3, use_residual=True):
        # MLP结构，输出维度与输入相同
        # 使用残差学习以固定映射为基础进行调整
```

**关键特性：**
- 保持空间维度 `[B, C, H, W]`
- 残差学习：`output = base_mapping + learned_adjustment`
- 近恒等初始化，确保训练稳定性

### 2. 模型修改 (`upsr_real_model.py`)

**保存方法：**
```python
def save(self, epoch, current_iter):
    super().save(epoch, current_iter)  # 保存net_g
    # 额外保存uncertainty_mapper
    if self.use_learnable_mapping and self.uncertainty_mapper is not None:
        save_filename = f'uncertainty_mapper_{current_iter}.pth'
        # 保存到同一目录
```

**加载方法：**
```python
def load_uncertainty_mapper(self, load_path, strict=True, param_key='params'):
    # 在__init__中自动调用，如果ckpt_path存在
```

**优化器设置：**
```python
def setup_optimizers(self):
    # 支持三种模式：
    # 1. freeze_others_initially=True: 只优化mapper
    # 2. lr_scale != 1.0: 差异化学习率
    # 3. 默认：相同学习率联合训练
```

---

## 可视化输出

在验证阶段，模型会自动保存以下热力图：

1. **不确定度热力图** - `uncertainty_heatmap_<iter>_<idx>.png`
   - 显示网络预测的不确定度分布
   - 统计信息：均值、标准差、最大/最小值

2. **真实残差热力图** - `real_residual_heatmap_<iter>_<idx>.png`
   - GT与双三次上采样之间的实际差异
   - 反映图像真实的复杂度分布

3. **对比热力图** - `comparison_heatmap_<iter>_<idx>.png`
   - 左：预估残差（网络输出 vs 双三次）
   - 右：真实残差（GT vs 双三次）
   - 用于评估不确定度预测的准确性

---

## 常见问题

### Q1: 为什么要分阶段训练？
**A:** 直接联合训练可能导致映射层学习到退化映射（总是输出常数），因为主网络已经收敛。分阶段训练让映射层先适应已训练好的主网络，再进行微调。

### Q2: 学习率如何设置？
**A:** 
- 阶段1：较大LR (2e-4) 快速适应
- 阶段2：主网络小LR (5e-5) 避免破坏，映射层大LR (2e-4) 继续优化

### Q3: 可以跳过某个阶段吗？
**A:** 
- 可以跳过阶段0，直接使用已有的预训练模型
- 不建议跳过阶段1，这是关键的warm-up阶段
- 可以在阶段1后停止，如果主网络已足够好

### Q4: 如何验证训练效果？
**A:** 观察对比热力图，如果预估残差与真实残差分布相似，说明映射层学习有效。同时关注PSNR/SSIM等指标是否提升。

---

## 实验建议

1. **监控热力图**：每个验证周期检查热力图，确保不确定度预测合理
2. **提前停止**：如果指标不再提升，可以提前停止训练
3. **学习率调优**：根据loss曲线调整lr_scale参数
4. **消融实验**：对比固定映射和可学习映射的效果差异

---

## 参考文件

- 模型实现：`basicsr/models/upsr_real_model.py`
- 映射层：`basicsr/models/uncertainty_mapping.py`
- 配置示例：`options/00[234]_UPSR_*.yml`
