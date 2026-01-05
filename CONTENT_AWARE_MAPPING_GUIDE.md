# 内容感知不确定度映射模块 - 使用指南

## 📌 概述

`ContentAwareSpatialUncertaintyMapping` 是一个改进的不确定度估计模块，通过 Cross-Attention 机制让 LQ 图像特征调制 SR 预测差异特征，实现内容自适应的不确定度估计。

## 🎯 核心思想

### 传统方法的局限

```python
# 固定映射：只使用 SR 预测差异
diff = (sr_pred - bicubic) / 2
uncertainty = fixed_mapping(diff)  # 忽略了 LQ 内容信息
```

**问题**：
- ❌ 不同内容区域（纹理/平滑/边缘）使用相同的映射策略
- ❌ 忽略 LQ 图像的退化程度和噪声水平
- ❌ 无法利用原始图像的内容先验

### 内容感知方法

```python
# 内容感知映射：同时使用 diff 和 LQ
diff = (sr_pred - bicubic) / 2
lq_hr = upsample(lq)  # 上采样到 HR 尺寸
uncertainty = content_aware_mapper(diff, lq_hr)
```

**优势**：
- ✅ 纹理丰富区域 vs 平滑区域 → 不同的不确定度策略
- ✅ 退化严重的区域 → 提高不确定度，允许更多探索
- ✅ Cross-Attention 学习最优的特征融合方式

## 🏗️ 架构设计

```
输入:
  - diff [B, 3, H, W]: SR预测差异
  - lq [B, 3, H, W]: LQ图像（[0,1]范围）

处理流程:
  1. 特征提取
     ├─ LQ Encoder: Conv → BN → ReLU → Conv → BN → ReLU
     └─ Diff Encoder: Conv → BN → ReLU → Conv → BN → ReLU
  
  2. Cross-Attention 调制
     ├─ Query: LQ特征（询问"需要多大不确定度"）
     └─ Key, Value: Diff特征（提供"SR预测质量"）
  
  3. 输出投影
     └─ Conv → BN → ReLU → Conv → 残差连接 → Clamp[min_noise, 1.0]

输出:
  - uncertainty [B, 3, H, W]: 内容感知的不确定度权重
```

### Cross-Attention 详解

```python
Q = LQ_features  # "这个位置需要多大不确定度？"
K, V = Diff_features  # "SR预测的质量信息"

Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V

# 效果：
# - 纹理区域的LQ特征 → 查询 → 调制不确定度以保留细节
# - 平滑区域的LQ特征 → 查询 → 降低不确定度以信任SR
# - 噪声区域的LQ特征 → 查询 → 提高不确定度以允许修正
```

## 🚀 使用方法

### 1. 配置文件设置

```yaml
uncertainty_mapping:
  use_learnable: true
  type: content_aware  # ✅ 使用内容感知映射
  
  # 模块参数
  hidden_channels: 64  # 特征提取隐藏层通道数
  num_heads: 4         # Cross-Attention 头数
  
  # 训练策略（可选）
  freeze_main_network: false
  mapper_lr_scale: 2.0    # 映射层学习率倍数
  main_lr_scale: 0.1      # 主网络学习率倍数
  
  # 加载预训练权重（可选）
  ckpt_path: path/to/uncertainty_mapper.pth
  strict_load: true
```

### 2. 训练命令

```bash
# 阶段1：训练映射层（可选，如果想要预训练）
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=1145 \
    train.py -opt options/stage1_content_aware_mapper.yml --launcher pytorch

# 阶段2：联合训练（推荐）
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=1145 \
    train.py -opt options/content_aware_uncertainty_mapping.yml --launcher pytorch
```

### 3. 代码集成

模块已完全集成到 UPSR 框架中，无需额外修改代码。训练和推理时会自动：

```python
# 训练时
if isinstance(self.uncertainty_mapper, ContentAwareSpatialUncertaintyMapping):
    # 上采样 LQ 到 HR 尺寸
    micro_lq_hr = F.interpolate(micro_lq, scale_factor=self.sf, mode='bicubic')
    micro_lq_hr = micro_lq_hr * 0.5 + 0.5  # 归一化到 [0, 1]
    
    # 调用内容感知映射
    micro_uncertainty = self.uncertainty_mapper(diff, micro_lq_hr)
else:
    # 普通映射：只需要 diff
    micro_uncertainty = self.uncertainty_mapper(diff)

# 推理时（自动处理）
un = self.uncertainty_mapper(diff, y0_hr)
```

## 📊 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `un_max` | float | 1.0 | 不确定度最大值 |
| `min_noise` | float | 0.4 | 最小噪声水平 |
| `channels` | int | 3 | 输入通道数（RGB） |
| `hidden_channels` | int | 64 | 特征提取隐藏层通道数 |
| `num_heads` | int | 4 | Cross-Attention 头数 |

### 参数选择建议

**hidden_channels**:
- 32: 轻量级，速度快，参数少
- 64: **推荐**，平衡性能和速度
- 128: 高性能，但计算量大

**num_heads**:
- 2: 计算量小，可能不够表达力
- 4: **推荐**，足够的多头注意力
- 8: 更强表达力，但参数量增加

## 🔬 测试验证

运行测试脚本验证模块功能：

```bash
cd /root/project/UPSR_copy
python test_content_aware_mapping.py
```

测试内容：
- ✓ 模块创建和参数统计
- ✓ 前向传播和输出验证
- ✓ 固定映射 vs 可学习映射对比
- ✓ 内容自适应性测试
- ✓ 梯度传播验证
- ✓ 内存和速度分析

## 📈 预期效果

### 不同区域的自适应行为

| 区域类型 | LQ 特征 | Diff 特征 | 预期不确定度 | 效果 |
|---------|---------|-----------|------------|------|
| **纹理丰富** | 高频细节多 | 差异大且结构化 | **适中** | 保留细节，允许适度探索 |
| **平滑区域** | 低频为主 | 差异小 | **低** | 高度信任SR结果 |
| **边缘** | 梯度大 | 差异集中 | **中等偏低** | 保持边缘锐利 |
| **噪声/退化** | 质量差 | 差异大无规律 | **高** | 允许diffusion修正 |

### 与基线方法对比

```
固定映射:
  uncertainty = f(|diff|)
  → 所有位置使用相同策略

空间映射:
  uncertainty = CNN(diff)
  → 考虑空间上下文，但忽略LQ内容

内容感知映射:
  uncertainty = Attention(LQ_feat, Diff_feat)
  → 根据LQ内容和退化程度自适应调整
```

## 🎛️ 高级功能

### 1. 可视化注意力图

```python
# 在 CrossAttentionBlock 的 forward 中添加
self.attn_weights = attn.detach()  # 保存注意力权重

# 可视化
import matplotlib.pyplot as plt
attn_map = mapper.cross_attn.attn_weights[0, 0].cpu().numpy()
plt.imshow(attn_map)
plt.title('Cross-Attention Map')
plt.savefig('attention_visualization.png')
```

### 2. 分析特征贡献

```python
# 禁用 Cross-Attention，看性能下降
mapper_no_attn = ContentAwareSpatialUncertaintyMapping(...)
# 修改 forward: modulated_feat = diff_feat  # 跳过 cross_attn

# 对比性能
```

### 3. 多尺度特征融合

可以扩展为多尺度版本：

```python
# 在不同尺度上提取特征
lq_feat_scales = [encoder1(lq), encoder2(downsample(lq)), ...]
diff_feat_scales = [encoder1(diff), encoder2(downsample(diff)), ...]

# 多尺度 Cross-Attention
for lq_f, diff_f in zip(lq_feat_scales, diff_feat_scales):
    modulated = cross_attn(lq_f, diff_f)
```

## ⚠️ 注意事项

1. **内存占用**：Cross-Attention 需要额外的内存（约 O(HW×HW)），对于大图像可能需要调整 `chop_size`

2. **学习率设置**：建议映射层使用较大学习率（2x），主网络使用较小学习率（0.1x）

3. **数据范围**：确保 LQ 图像在 [0, 1] 范围，Diff 根据实际情况归一化

4. **初始化**：输出层初始化为零，保证初始行为接近固定映射

## 🔧 故障排除

**问题1**: 内存不足
```yaml
# 减小 hidden_channels 或 batch_size
uncertainty_mapping:
  hidden_channels: 32  # 从 64 减少到 32
datasets:
  train:
    batch_size_per_gpu: 8  # 从 16 减少到 8
```

**问题2**: 训练不稳定
```yaml
# 使用更保守的学习率
train:
  optim_g:
    lr: !!float 5e-5  # 降低基础学习率
uncertainty_mapping:
  mapper_lr_scale: 1.5  # 降低映射层学习率倍数
```

**问题3**: 效果不明显
- 检查预训练模型是否正确加载
- 增加训练迭代次数
- 尝试不同的 `hidden_channels` 和 `num_heads`

## 📚 参考

- UPSR 原始论文：Uncertainty-aware Progressive Super-Resolution
- Cross-Attention 机制：Attention Is All You Need
- 残差学习：Deep Residual Learning for Image Recognition

## 📧 反馈

如有问题或建议，请在 GitHub 上提 Issue 或联系开发团队。
