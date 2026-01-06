#!/usr/bin/env python3
"""
诊断diff输入范围和不确定度映射问题
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 模拟UPSR中的diff计算
def simulate_diff_calculation():
    """
    模拟upsr_real_model.py中的diff计算过程
    
    在训练/验证中：
    diff = (y_hat - y_bicubic) / 2
    
    其中：
    - y_hat = (net_mse(y0 * 0.5 + 0.5) - 0.5) / 0.5  # MSE网络输出，范围[-1, 1]
    - y_bicubic = F.interpolate(y0, scale_factor=4, mode='bicubic')  # 范围[-1, 1]
    """
    
    # 模拟LQ输入（范围[-1, 1]）
    lq = torch.randn(1, 3, 64, 64) * 0.5  # 模拟归一化后的LQ
    
    # 模拟bicubic上采样
    bicubic = F.interpolate(lq, scale_factor=4, mode='bicubic', align_corners=False)
    
    # 模拟MSE网络输出（假设输出接近bicubic，但有差异）
    # SR网络通常会产生更锐利的结果，边缘会有更大差异
    sr_mse = bicubic + torch.randn_like(bicubic) * 0.05  # 添加噪声模拟预测差异
    
    # 在边缘添加更大的差异（模拟实际情况）
    edge_mask = create_edge_mask(bicubic)
    sr_mse = sr_mse + edge_mask * torch.randn_like(bicubic) * 0.2
    
    # 计算diff（关键公式）
    diff = (sr_mse - bicubic) / 2
    
    return diff, bicubic, sr_mse, lq


def create_edge_mask(image):
    """创建边缘mask"""
    # 使用Sobel算子检测边缘
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).float()
    sobel_y = sobel_x.transpose(1, 2)
    
    edge_x = F.conv2d(image, sobel_x.unsqueeze(0).repeat(3, 1, 1, 1), padding=1, groups=3)
    edge_y = F.conv2d(image, sobel_y.unsqueeze(0).repeat(3, 1, 1, 1), padding=1, groups=3)
    
    edge = torch.sqrt(edge_x**2 + edge_y**2)
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
    
    return edge


def analyze_diff_statistics(diff, name=''):
    """分析diff的统计特性"""
    print(f"\n{'='*60}")
    print(f"  Diff Statistics: {name}")
    print(f"{'='*60}")
    
    for c, channel in enumerate(['R', 'G', 'B']):
        diff_c = diff[0, c]
        print(f"\n{channel} Channel:")
        print(f"  Min:    {diff_c.min():.6f}")
        print(f"  Max:    {diff_c.max():.6f}")
        print(f"  Mean:   {diff_c.mean():.6f}")
        print(f"  Std:    {diff_c.std():.6f}")
        print(f"  Median: {diff_c.median():.6f}")
        
        # 百分位
        print(f"  P5:     {torch.quantile(diff_c, 0.05):.6f}")
        print(f"  P95:    {torch.quantile(diff_c, 0.95):.6f}")
        
        # |diff|的统计
        abs_diff = torch.abs(diff_c)
        print(f"  |diff| Mean: {abs_diff.mean():.6f}")
        print(f"  |diff| Max:  {abs_diff.max():.6f}")


def test_fixed_mapping(diff, un_max=1.0, min_noise=0.0):
    """
    测试固定映射
    
    un = b_un + (1 - b_un) * (|diff|.clamp(0, un_max) / un_max)
    """
    normalized_diff = torch.abs(diff).clamp_(0., un_max) / un_max
    uncertainty = min_noise + (1 - min_noise) * normalized_diff
    
    print(f"\n{'='*60}")
    print(f"  Fixed Mapping (un_max={un_max}, min_noise={min_noise})")
    print(f"{'='*60}")
    print(f"Uncertainty Min:  {uncertainty.min():.6f}")
    print(f"Uncertainty Max:  {uncertainty.max():.6f}")
    print(f"Uncertainty Mean: {uncertainty.mean():.6f}")
    print(f"Uncertainty Std:  {uncertainty.std():.6f}")
    
    return uncertainty


def visualize_diff_and_uncertainty(diff, uncertainty_fixed, save_dir='./debug_diff'):
    """可视化diff和不确定度"""
    Path(save_dir).mkdir(exist_ok=True)
    
    # 1. Diff的三个通道
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for c, channel in enumerate(['R', 'G', 'B']):
        # 原始diff（可能有负值）
        im = axes[0, c].imshow(diff[0, c].cpu().numpy(), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[0, c].set_title(f'Diff {channel} (raw)')
        plt.colorbar(im, ax=axes[0, c])
        
        # |diff|
        abs_diff = torch.abs(diff[0, c])
        im = axes[1, c].imshow(abs_diff.cpu().numpy(), cmap='hot', vmin=0, vmax=0.5)
        axes[1, c].set_title(f'|Diff| {channel}')
        plt.colorbar(im, ax=axes[1, c])
    
    # 固定映射的不确定度
    im = axes[0, 3].imshow(uncertainty_fixed[0].mean(0).cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    axes[0, 3].set_title('Fixed Uncertainty (mean)')
    plt.colorbar(im, ax=axes[0, 3])
    
    # 不确定度分布直方图
    axes[1, 3].hist(uncertainty_fixed[0].cpu().numpy().flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 3].set_title('Uncertainty Distribution')
    axes[1, 3].set_xlabel('Value')
    axes[1, 3].set_ylabel('Frequency')
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/diff_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_dir}/diff_analysis.png")
    plt.close()


def test_range_explosion():
    """
    测试关键问题：diff的范围是否导致不确定度爆炸
    
    **关键假设**：
    如果 |diff| 在边缘处很小（接近0），在平滑区域很大，
    那么归一化后会导致：
    - 边缘 → 低不确定度（黑色）
    - 平滑区域 → 高不确定度（白色）
    
    这与预期相反！
    """
    print("\n" + "="*60)
    print("  测试假设：diff范围导致不确定度反转")
    print("="*60)
    
    # 场景1：正常情况（边缘有大diff）
    print("\n场景1：边缘有大diff（正常）")
    diff_normal = torch.randn(1, 3, 256, 256) * 0.1
    edge_mask = create_edge_mask(diff_normal)
    diff_normal = diff_normal + edge_mask * 0.3  # 边缘增加diff
    
    un_normal = test_fixed_mapping(diff_normal, un_max=1.0, min_noise=0.0)
    print(f"边缘不确定度平均值: {(un_normal * edge_mask).sum() / edge_mask.sum():.6f}")
    print(f"非边缘不确定度平均值: {(un_normal * (1-edge_mask)).sum() / (1-edge_mask).sum():.6f}")
    
    # 场景2：异常情况（边缘有小diff）- 可能是您遇到的情况
    print("\n场景2：边缘有小diff（异常，可能的bug）")
    diff_abnormal = torch.randn(1, 3, 256, 256) * 0.5  # 平滑区域大噪声
    diff_abnormal = diff_abnormal * (1 - edge_mask * 0.9)  # 边缘减小diff
    
    un_abnormal = test_fixed_mapping(diff_abnormal, un_max=1.0, min_noise=0.0)
    print(f"边缘不确定度平均值: {(un_abnormal * edge_mask).sum() / edge_mask.sum():.6f}")
    print(f"非边缘不确定度平均值: {(un_abnormal * (1-edge_mask)).sum() / (1-edge_mask).sum():.6f}")
    
    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(edge_mask[0, 0].cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Edge Mask')
    
    axes[0, 1].imshow(torch.abs(diff_normal[0]).mean(0).cpu().numpy(), cmap='hot')
    axes[0, 1].set_title('|Diff| Normal (edge large)')
    
    axes[0, 2].imshow(un_normal[0].mean(0).cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    axes[0, 2].set_title('Uncertainty Normal')
    
    axes[1, 0].imshow(edge_mask[0, 0].cpu().numpy(), cmap='gray')
    axes[1, 0].set_title('Edge Mask')
    
    axes[1, 1].imshow(torch.abs(diff_abnormal[0]).mean(0).cpu().numpy(), cmap='hot')
    axes[1, 1].set_title('|Diff| Abnormal (edge small)')
    
    axes[1, 2].imshow(un_abnormal[0].mean(0).cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    axes[1, 2].set_title('Uncertainty Abnormal ⚠️')
    
    plt.tight_layout()
    plt.savefig('./debug_diff/range_explosion_test.png', dpi=150)
    print(f"\n✅ Range explosion test saved to: ./debug_diff/range_explosion_test.png")
    plt.close()


def main():
    print("\n" + "="*60)
    print("  Diff Range Diagnostic Tool")
    print("="*60)
    
    # 1. 模拟diff计算
    diff, bicubic, sr_mse, lq = simulate_diff_calculation()
    
    # 2. 分析diff统计
    analyze_diff_statistics(diff, name='Simulated')
    
    # 3. 测试固定映射
    uncertainty_fixed = test_fixed_mapping(diff, un_max=1.0, min_noise=0.0)
    
    # 4. 可视化
    visualize_diff_and_uncertainty(diff, uncertainty_fixed)
    
    # 5. 测试范围爆炸假设
    test_range_explosion()
    
    print("\n" + "="*60)
    print("  Diagnostic Suggestions")
    print("="*60)
    print("""
1. 检查SR网络输出范围：
   - 如果sr_mse在[-1,1]之外，会导致diff超出预期范围
   
2. 检查diff的除以2操作：
   diff = (y_hat - y_bicubic) / 2
   - 这会将diff范围缩小到[-1, 1]
   - 但如果y_hat和y_bicubic差异很大，仍可能超出
   
3. 检查边缘处的diff值：
   - 打印实际图像的edge_diff和smooth_diff
   - 如果edge_diff < smooth_diff，会导致不确定度反转
   
4. 可学习映射可能放大了这个问题：
   - 如果输入diff的分布异常，网络会学到错误的模式
   - adjustment可能会进一步加剧这个问题
    """)


if __name__ == '__main__':
    main()
