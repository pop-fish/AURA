# Diff范围问题分析报告

## 问题现象

1. **可学习映射的不确定度图异常**：
   - 大部分区域是**白色**（高不确定度）
   - 边缘区域是**黑色**（低不确定度）
   - 这与预期**完全相反**（应该边缘高不确定度，平滑区域低不确定度）

2. **边缘处形成马赛克**：
   - 不是窗口边界马赛克
   - 而是图像内容边缘处的马赛克

## 根本原因分析

### 🔴 核心问题：diff的计算和归一化

在 `upsr_real_model.py` 第355行和第433行：

```python
diff = (y_hat - y_bicubic) / 2
```

**问题1：除以2的作用**
- `y_hat` 范围：`[-1, 1]`（MSE网络输出）
- `y_bicubic` 范围：`[-1, 1]`（双三次插值）
- 差值范围：最大 `[-2, 2]`
- **除以2后**：`diff ∈ [-1, 1]`

但是在**边缘区域**：
- SR网络通常能较好地预测边缘（因为有边缘先验）
- `|y_hat - y_bicubic|` 在边缘处**可能很小**
- 在**平滑区域**，由于噪声或预测不确定性，差异反而**更大**

### 🔴 问题2：固定映射的归一化

在 `uncertainty_mapping.py` 第454行：

```python
normalized_diff = torch.abs(diff).clamp_(0., self.un_max) / self.un_max
```

**关键问题**：`un_max=1.0` 的含义
- 如果 `un_max=1.0`，意味着期望 `|diff|` 的最大值为1.0
- 但实际上：
  - 如果 `|diff|` 在平滑区域为 **0.3**
  - 在边缘区域为 **0.05**
  - 归一化后：
    - 平滑区域：`0.3 / 1.0 = 0.3` → 中等不确定度
    - 边缘区域：`0.05 / 1.0 = 0.05` → 低不确定度 ⚠️

这导致：**边缘不确定度低，平滑区域不确定度高**（完全反转！）

### 🔴 问题3：可学习映射放大了问题

在 `ContentAwareSpatialUncertaintyMapping` 中：

```python
# forward函数
base_weight = self.fixed_mapping(diff)  # 已经错误的基准
adjustment = self.output_proj(modulated_feat)  # 可能进一步放大
final_weight = base_weight + adjustment
```

如果训练数据中的 `diff` 分布就是异常的（边缘小、平滑大），那么：
1. `base_weight` 已经是错误的（边缘低、平滑高）
2. 网络学习 `adjustment` 时会基于错误的分布
3. 最终结果会**更加错误**

### 🔴 问题4：为什么会有马赛克？

马赛克出现在**边缘处**，可能原因：

1. **Cross-Attention在边缘处失效**：
   - 边缘的 `diff` 值很小 → `diff_feat` 特征弱
   - Cross-Attention权重在边缘处不稳定
   - 导致 `adjustment` 在边缘处出现高频振荡

2. **BatchNorm统计问题**：
   - 边缘像素占比小
   - BatchNorm在边缘区域的统计不稳定
   - 导致边缘处的归一化异常

## 验证假设的方法

### 1. 打印实际diff的统计

在 `upsr_real_model.py` 中添加诊断代码：

```python
# 第355行后添加
if jj == 0:  # 只打印第一个micro-batch
    print(f"\n=== Diff Statistics ===")
    print(f"diff min: {diff.min():.6f}, max: {diff.max():.6f}")
    print(f"|diff| mean: {torch.abs(diff).mean():.6f}")
    
    # 检测边缘
    edge_mask = compute_edge_mask(micro_gt)  # 需要实现
    if edge_mask.sum() > 0:
        edge_diff = torch.abs(diff)[edge_mask].mean()
        non_edge_diff = torch.abs(diff)[~edge_mask].mean()
        print(f"|diff| at edges: {edge_diff:.6f}")
        print(f"|diff| at smooth: {non_edge_diff:.6f}")
        
        if edge_diff < non_edge_diff:
            print("⚠️ WARNING: Edge diff < Smooth diff (UNEXPECTED!)")
```

### 2. 检查保存的不确定度图

运行：
```bash
python check_saved_uncertainty.py
```

查看：
- 边缘处的数值（应该高，但可能实际很低）
- 平滑区域的数值（应该低，但可能实际很高）

### 3. 可视化diff分布

```python
import matplotlib.pyplot as plt

# 在validation时
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# |diff|
axes[0].imshow(torch.abs(diff[0]).mean(0).cpu(), cmap='hot')
axes[0].set_title('|diff|')

# 固定映射不确定度
axes[1].imshow(micro_uncertainty[0].mean(0).cpu(), cmap='jet', vmin=0, vmax=1)
axes[1].set_title('Uncertainty (Fixed)')

# GT用于参考
axes[2].imshow((micro_gt[0] * 0.5 + 0.5).permute(1, 2, 0).cpu())
axes[2].set_title('GT')

plt.savefig('diff_debug.png')
```

## 修复方案

### 方案1：自适应归一化（推荐）

```python
def adaptive_normalization(diff, percentile=95):
    """
    使用百分位数进行自适应归一化
    """
    abs_diff = torch.abs(diff)
    
    # 使用95百分位作为归一化因子（避免极值影响）
    normalizer = torch.quantile(abs_diff.view(abs_diff.size(0), -1), 
                                percentile/100.0, dim=1, keepdim=True)
    normalizer = normalizer.view(-1, 1, 1, 1)
    
    normalized = abs_diff / (normalizer + 1e-8)
    normalized = normalized.clamp(0, 1)
    
    return normalized
```

### 方案2：使用相对diff

```python
# 不使用绝对diff，而是相对于局部均值的diff
def compute_relative_diff(y_hat, y_bicubic):
    diff = (y_hat - y_bicubic) / 2
    
    # 计算局部标准差
    local_std = F.avg_pool2d(
        (diff - F.avg_pool2d(diff, 5, stride=1, padding=2))**2,
        5, stride=1, padding=2
    ).sqrt()
    
    # 归一化
    relative_diff = torch.abs(diff) / (local_std + 1e-4)
    
    return relative_diff
```

### 方案3：修正un_max

```python
# 在配置文件中，将un_max设置为diff的实际范围
# 例如，如果|diff|通常在[0, 0.3]，则设置un_max=0.3

# content_aware_uncertainty_mapping.yml
diffusion:
  un: 0.3  # 而不是1.0
  min_noise: 0.0
```

### 方案4：添加边缘感知损失

```python
class ContentAwareUncertaintyLoss(nn.Module):
    def forward(self, uncertainty, diff, gt):
        # 计算边缘
        edge_mask = compute_edges(gt)
        
        # 边缘处应该有更高不确定度
        edge_loss = F.mse_loss(
            uncertainty * edge_mask,
            torch.ones_like(uncertainty) * 0.8 * edge_mask
        )
        
        # 平滑区域应该有更低不确定度
        smooth_loss = F.mse_loss(
            uncertainty * (1 - edge_mask),
            torch.abs(diff) * (1 - edge_mask)  # 基于实际diff
        )
        
        return edge_loss + smooth_loss
```

## 下一步行动

1. **立即验证**：运行 `check_saved_uncertainty.py` 检查保存的图像
2. **添加诊断**：在 `upsr_real_model.py` 中打印diff统计
3. **修复un_max**：根据实际diff范围调整配置
4. **重新训练**：使用修正后的配置重新训练Stage1

## 预期结果

修复后应该看到：
- ✅ 边缘区域：高不确定度（红色/白色）
- ✅ 平滑区域：低不确定度（蓝色/黑色）
- ✅ 无马赛克效应
- ✅ 平滑过渡
