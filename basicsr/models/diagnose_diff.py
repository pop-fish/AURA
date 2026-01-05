#!/usr/bin/env python3
"""
在upsr_real_model.py中添加diff诊断代码
使用此补丁来验证diff范围假设
"""

import torch
import torch.nn.functional as F

def compute_edge_mask(image, threshold=0.1):
    """
    计算边缘mask
    
    Args:
        image: [B, C, H, W]，范围[-1, 1]
        threshold: 边缘检测阈值
    
    Returns:
        edge_mask: [B, 1, H, W]，bool类型
    """
    # Sobel边缘检测
    sobel_x = torch.tensor([
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]
    ], dtype=image.dtype, device=image.device)
    
    sobel_y = sobel_x.transpose(1, 2)
    
    # 对每个通道计算梯度
    grad_x = F.conv2d(image, sobel_x.unsqueeze(0).repeat(image.size(1), 1, 1, 1), 
                      padding=1, groups=image.size(1))
    grad_y = F.conv2d(image, sobel_y.unsqueeze(0).repeat(image.size(1), 1, 1, 1), 
                      padding=1, groups=image.size(1))
    
    # 梯度幅值
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
    
    # 平均所有通道
    grad_mag = grad_mag.mean(dim=1, keepdim=True)
    
    # 二值化
    edge_mask = grad_mag > threshold
    
    return edge_mask


def diagnose_diff_distribution(diff, gt=None, name=''):
    """
    诊断diff的分布特性
    
    Args:
        diff: [B, C, H, W]
        gt: [B, C, H, W]，可选，用于边缘检测
        name: 标识符
    """
    print(f"\n{'='*70}")
    print(f"  Diff Distribution Analysis: {name}")
    print(f"{'='*70}")
    
    abs_diff = torch.abs(diff)
    
    # 全局统计
    print(f"Global Statistics:")
    print(f"  diff min:    {diff.min():.6f}")
    print(f"  diff max:    {diff.max():.6f}")
    print(f"  |diff| mean: {abs_diff.mean():.6f}")
    print(f"  |diff| std:  {abs_diff.std():.6f}")
    print(f"  |diff| P50:  {torch.quantile(abs_diff, 0.50):.6f}")
    print(f"  |diff| P95:  {torch.quantile(abs_diff, 0.95):.6f}")
    print(f"  |diff| P99:  {torch.quantile(abs_diff, 0.99):.6f}")
    
    # 边缘vs平滑区域
    if gt is not None:
        edge_mask = compute_edge_mask(gt, threshold=0.1)
        
        if edge_mask.sum() > 0:
            # 展平mask
            edge_flat = edge_mask.view(-1)
            diff_flat = abs_diff.view(-1)
            
            edge_diff = diff_flat[edge_flat].mean()
            smooth_diff = diff_flat[~edge_flat].mean()
            
            print(f"\nSpatial Statistics:")
            print(f"  Edge pixels:     {edge_flat.sum().item()} ({100*edge_flat.sum()/edge_flat.numel():.1f}%)")
            print(f"  |diff| at edges: {edge_diff:.6f}")
            print(f"  |diff| at smooth:{smooth_diff:.6f}")
            print(f"  Edge/Smooth ratio: {edge_diff / (smooth_diff + 1e-8):.4f}")
            
            if edge_diff < smooth_diff:
                print(f"  ⚠️  WARNING: Edge diff < Smooth diff")
                print(f"      This will cause INVERTED uncertainty!")
                print(f"      (Edges will have LOW uncertainty, smooth areas HIGH)")
            else:
                print(f"  ✅ OK: Edge diff > Smooth diff")
    
    # 通道统计
    print(f"\nPer-Channel Statistics:")
    for c, channel in enumerate(['R', 'G', 'B']):
        ch_mean = abs_diff[:, c].mean()
        ch_max = abs_diff[:, c].max()
        print(f"  {channel}: mean={ch_mean:.6f}, max={ch_max:.6f}")
    
    print(f"{'='*70}\n")


def diagnose_uncertainty_quality(uncertainty, diff, gt=None, name=''):
    """
    诊断不确定度质量
    
    Args:
        uncertainty: [B, C, H, W]，不确定度图
        diff: [B, C, H, W]，原始diff
        gt: [B, C, H, W]，可选
        name: 标识符
    """
    print(f"\n{'='*70}")
    print(f"  Uncertainty Quality Check: {name}")
    print(f"{'='*70}")
    
    print(f"Uncertainty Statistics:")
    print(f"  Min:    {uncertainty.min():.6f}")
    print(f"  Max:    {uncertainty.max():.6f}")
    print(f"  Mean:   {uncertainty.mean():.6f}")
    print(f"  Std:    {uncertainty.std():.6f}")
    print(f"  Median: {torch.median(uncertainty):.6f}")
    
    # 检查饱和
    saturated_high = (uncertainty > 0.95).float().mean()
    saturated_low = (uncertainty < 0.05).float().mean()
    
    print(f"\nSaturation:")
    print(f"  >0.95 (high): {100*saturated_high:.2f}%")
    print(f"  <0.05 (low):  {100*saturated_low:.2f}%")
    
    if saturated_high > 0.3:
        print(f"  ⚠️  WARNING: >30% pixels have high uncertainty (>0.95)")
        print(f"      This indicates poor mapping!")
    
    # 边缘质量
    if gt is not None:
        edge_mask = compute_edge_mask(gt, threshold=0.1)
        
        if edge_mask.sum() > 0:
            edge_flat = edge_mask.view(-1)
            un_flat = uncertainty.mean(dim=1).view(-1)  # 平均通道
            
            edge_un = un_flat[edge_flat].mean()
            smooth_un = un_flat[~edge_flat].mean()
            
            print(f"\nSpatial Quality:")
            print(f"  Uncertainty at edges: {edge_un:.6f}")
            print(f"  Uncertainty at smooth:{smooth_un:.6f}")
            print(f"  Edge/Smooth ratio: {edge_un / (smooth_un + 1e-8):.4f}")
            
            if edge_un < smooth_un:
                print(f"  ❌ INVERTED: Edge uncertainty < Smooth uncertainty")
                print(f"     Expected behavior: Edge > Smooth")
            else:
                print(f"  ✅ CORRECT: Edge uncertainty > Smooth uncertainty")
    
    print(f"{'='*70}\n")


# 使用示例（在upsr_real_model.py中添加）:
"""
# 在训练循环中（第355行后）
if jj == 0 and self.opt.get('debug_diff', False):
    diagnose_diff_distribution(diff, gt=micro_gt, name='Training')
    diagnose_uncertainty_quality(micro_uncertainty, diff, gt=micro_gt, name='Training')

# 在验证函数中（第433行后）  
if self.opt.get('debug_diff', False):
    diagnose_diff_distribution(diff, gt=None, name='Validation')
    diagnose_uncertainty_quality(un, diff, gt=None, name='Validation')
"""

if __name__ == '__main__':
    print("""
这个脚本包含了用于诊断diff分布的工具函数。

要使用这些函数：

1. 在 upsr_real_model.py 开头添加：
   from .diagnose_diff import diagnose_diff_distribution, diagnose_uncertainty_quality

2. 在配置文件中启用调试：
   debug_diff: true

3. 在训练/验证代码中添加诊断调用（见上方示例）

4. 运行训练/测试，观察输出：
   - 如果看到 "⚠️ WARNING: Edge diff < Smooth diff"
     → 说明diff分布异常，会导致不确定度反转
   
   - 如果看到 "❌ INVERTED: Edge uncertainty < Smooth uncertainty"
     → 说明不确定度映射结果错误
     
5. 根据诊断结果调整un_max或使用自适应归一化
    """)
