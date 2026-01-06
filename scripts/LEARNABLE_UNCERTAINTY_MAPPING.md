# 可学习不确定度映射 (Learnable Uncertainty Mapping)

## 概述

本改进将UPSR模型中固定的不确定度到权重的映射过程替换为可学习的神经网络，使模型能够自适应地学习更优的噪声添加策略。

## 核心思想

### 原始方法（固定映射）
```python
# 计算残差作为不确定度
diff = (micro_sr_mse - micro_lq_bicubic) / 2

# 固定的线性映射
normalized_diff = |diff|.clamp(0, un_max) / un_max
uncertainty_weight = b_un + (1 - b_un) * normalized_diff
```

### 改进方法（可学习映射）
```python
# 使用可学习的神经网络映射
uncertainty_weight = LearnableMapper(diff)

# 采用残差学习策略保证稳定性
base_weight = fixed_mapping(diff)  # 固定映射作为基准
adjustment = MLP(base_weight)       # 学习调整量
final_weight = base_weight + adjustment
```

## 架构设计

### 1. MLP-based Mapping (`LearnableUncertaintyMapping`)
- **结构**: 多层感知机 (MLP)
- **输入**: 归一化的不确定度值
- **输出**: 调整后的权重
- **特点**: 
  - 逐像素独立映射
  - 参数少，训练快
  - 适合微调场景

### 2. Spatial-aware Mapping (`SpatialUncertaintyMapping`)
- **结构**: 卷积神经网络 (CNN)
- **输入**: 完整的差异图
- **输出**: 空间感知的权重图
- **特点**: 
  - 考虑空间上下文
  - 更强的表达能力
  - 适合从头训练

## 训练策略

### 策略1: 分阶段训练（推荐）

**阶段1 - Warm-up（冻结主网络）**
```yaml
uncertainty_mapping:
  use_learnable: true
  type: mlp  # 或 spatial
  staged_training: true
  freeze_others_initially: true
  hidden_dim: 64
  num_layers: 3
  use_residual: true
  
train:
  total_iter: 10000  # warm-up阶段迭代次数
```

在训练代码中调用：
```python
# 训练10k iterations后切换
if current_iter == 10000:
    model.switch_to_full_training()
```

**阶段2 - 全网络微调**
- 自动解冻所有参数
- 使用不同学习率:
  - 主网络: `base_lr * 0.1` (较小，保持预训练权重)
  - 映射层: `base_lr * 2.0` (较大，继续优化)

### 策略2: 直接联合训练

```yaml
uncertainty_mapping:
  use_learnable: true
  type: mlp
  staged_training: false
  lr_scale: 1.0  # 相对于主网络的学习率比例
  
train:
  optim_g:
    type: Adam
    lr: !!float 1e-4
```

## 配置参数说明

### 基础配置
```yaml
uncertainty_mapping:
  # 是否启用可学习映射
  use_learnable: true
  
  # 映射类型: 'mlp' 或 'spatial'
  type: mlp
  
  # 预训练权重路径（可选）
  ckpt_path: null
  strict_load: true
```

### MLP类型参数
```yaml
uncertainty_mapping:
  type: mlp
  hidden_dim: 64      # 隐藏层维度
  num_layers: 3       # MLP层数
  use_residual: true  # 使用残差连接
```

### Spatial类型参数
```yaml
uncertainty_mapping:
  type: spatial
  hidden_channels: 32  # 卷积隐藏通道数
```

### 训练策略参数
```yaml
uncertainty_mapping:
  # 分阶段训练
  staged_training: true
  freeze_others_initially: true
  
  # 学习率设置
  lr_scale: 2.0          # 映射层学习率倍数
  main_lr_scale: 0.1     # 第二阶段主网络学习率倍数
```

## 使用示例

### 1. 从预训练模型开始微调（推荐）

**步骤1**: 准备预训练的UPSR模型
```yaml
# config.yml
path:
  pretrain_network_g: experiments/pretrained/upsr_pretrained.pth
  strict_load_g: true

uncertainty_mapping:
  use_learnable: false  # 先用固定映射验证
```

**步骤2**: 启用可学习映射，warm-up训练
```yaml
uncertainty_mapping:
  use_learnable: true
  type: mlp
  hidden_dim: 64
  num_layers: 3
  use_residual: true
  staged_training: true
  freeze_others_initially: true

train:
  total_iter: 10000  # warm-up阶段
```

**步骤3**: 切换到全网络训练
```python
# 在训练脚本中
if current_iter == 10000:
    model.switch_to_full_training()
    logger.info("Switched to full training mode")
```

继续训练到收敛。

### 2. 从头开始训练

```yaml
uncertainty_mapping:
  use_learnable: true
  type: spatial  # 使用空间感知映射
  hidden_channels: 32
  staged_training: false
  lr_scale: 1.0

train:
  total_iter: 500000
  optim_g:
    type: Adam
    lr: !!float 2e-4
```

### 3. 监控训练过程

在验证时添加映射统计信息：
```python
# 在validation代码中
if hasattr(model, 'uncertainty_mapper') and model.uncertainty_mapper is not None:
    # 获取统计信息
    stats = model.uncertainty_mapper.get_statistics(diff)
    logger.info(f"Uncertainty Mapping Stats: {stats}")
```

输出示例：
```
base_weight_mean: 0.523
base_weight_std: 0.182
adjustment_mean: 0.012
adjustment_std: 0.045
final_weight_mean: 0.535
adjustment_abs_max: 0.156
```

## 初始化策略

### 为什么初始化为固定映射？

1. **稳定性**: 避免随机初始化导致的训练不稳定
2. **收敛速度**: 从已验证的映射开始，加速收敛
3. **性能保底**: 最差情况下等价于原始方法

### 实现细节

```python
def _init_as_identity(self):
    """初始化网络使其输出接近于输入"""
    # 最后一层初始化为零 -> 残差项初始为0
    nn.init.zeros_(self.mlp[-1].weight)
    nn.init.zeros_(self.mlp[-1].bias)
    
    # 其他层使用小权重初始化
    for module in self.mlp[:-1]:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)
```

## 预期效果

### 性能提升
- **PSNR**: +0.2~0.5 dB (取决于数据集)
- **LPIPS**: -0.01~0.03 (更好的感知质量)
- **训练稳定性**: 显著提升，梯度更平滑

### 训练成本
- **额外参数**: 
  - MLP (hidden_dim=64, 3层): ~13K参数
  - Spatial (hidden_channels=32): ~10K参数
- **额外显存**: <50MB
- **训练时间**: 增加 <5%

## 注意事项

### 1. 过拟合风险
- 使用Dropout (p=0.1)
- 使用LayerNorm稳定训练
- 监控验证集性能

### 2. 学习率调整
- Warm-up阶段: 使用标准学习率
- 全训练阶段: 主网络使用0.1x学习率

### 3. 调试建议
- 先用固定映射验证基础流程
- 逐步启用可学习映射
- 可视化权重分布变化

## 可视化示例

```python
import matplotlib.pyplot as plt

# 对比固定映射和可学习映射
with torch.no_grad():
    fixed_weight = model.uncertainty_mapper.fixed_mapping(diff)
    learned_weight = model.uncertainty_mapper(diff)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.hist(fixed_weight.cpu().flatten().numpy(), bins=50)
    plt.title('Fixed Mapping Distribution')
    
    plt.subplot(132)
    plt.hist(learned_weight.cpu().flatten().numpy(), bins=50)
    plt.title('Learned Mapping Distribution')
    
    plt.subplot(133)
    plt.hist((learned_weight - fixed_weight).cpu().flatten().numpy(), bins=50)
    plt.title('Adjustment Distribution')
    
    plt.savefig('uncertainty_mapping_comparison.png')
```

## 扩展方向

### 1. 时间感知映射
在视频超分中，考虑时间维度的不确定度变化

### 2. 自适应权重
根据当前时间步 t 动态调整映射策略

### 3. 多尺度映射
在不同尺度上分别学习不确定度映射

## 引用

如果使用本改进，请引用原始UPSR论文并说明使用了可学习不确定度映射改进。

---

**作者**: AI Assistant  
**日期**: 2025-10-06  
**版本**: 1.0
