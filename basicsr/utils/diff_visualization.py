"""
Diff可视化工具模块

提供便捷的函数来可视化推理过程中的Diff(差异图)和Uncertainty(不确定度图)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from pathlib import Path


def save_heatmap(data, save_path, title='Heatmap', cmap='coolwarm', vmin=None, vmax=None, dpi=150):
    """
    保存单个热力图
    
    Args:
        data: numpy数组 [H, W]
        save_path: 保存路径
        title: 标题
        cmap: colormap名称
        vmin, vmax: 值域范围
        dpi: 分辨率
    """
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'{title}\nRange: [{data.min():.4f}, {data.max():.4f}], Mean: {data.mean():.4f}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def tensor_to_heatmap_data(tensor, aggregate_channels=True):
    """
    将tensor转换为可用于热力图的numpy数组
    
    Args:
        tensor: [1, C, H, W] 或 [C, H, W] 的tensor
        aggregate_channels: 是否对通道求平均
        
    Returns:
        numpy数组 [H, W] 或 [C, H, W]
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # [C, H, W]
    
    data = tensor.cpu().numpy()
    
    if aggregate_channels and data.shape[0] > 1:
        data = np.mean(data, axis=0)  # [H, W]
    
    return data


def visualize_diff_heatmap(diff_tensor, save_dir, prefix='diff', save_channels=True):
    """
    可视化Diff热力图
    
    Args:
        diff_tensor: [1, 3, H, W] 的差异图tensor
        save_dir: 保存目录
        prefix: 文件名前缀
        save_channels: 是否分别保存各通道的热力图
        
    Returns:
        保存的文件路径列表
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_files = []
    
    # 1. 保存平均热力图
    diff_avg = tensor_to_heatmap_data(diff_tensor, aggregate_channels=True)
    avg_path = os.path.join(save_dir, f'{prefix}_avg.png')
    save_heatmap(diff_avg, avg_path, title='Diff (Avg across RGB)', cmap='coolwarm')
    saved_files.append(avg_path)
    
    # 2. 分别保存各通道
    if save_channels and diff_tensor.size(1) == 3:
        channel_names = ['R', 'G', 'B']
        for i, ch_name in enumerate(channel_names):
            ch_data = diff_tensor[0, i].cpu().numpy()
            ch_path = os.path.join(save_dir, f'{prefix}_ch{ch_name}.png')
            save_heatmap(ch_data, ch_path, title=f'Diff - {ch_name} Channel', cmap='coolwarm')
            saved_files.append(ch_path)
    
    return saved_files


def visualize_uncertainty_heatmap(uncertainty_tensor, save_dir, prefix='uncertainty', save_channels=True):
    """
    可视化Uncertainty热力图
    
    Args:
        uncertainty_tensor: [1, 3, H, W] 的不确定度tensor
        save_dir: 保存目录
        prefix: 文件名前缀
        save_channels: 是否分别保存各通道的热力图
        
    Returns:
        保存的文件路径列表
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_files = []
    
    # 1. 保存平均热力图
    uncertainty_avg = tensor_to_heatmap_data(uncertainty_tensor, aggregate_channels=True)
    avg_path = os.path.join(save_dir, f'{prefix}_avg.png')
    save_heatmap(uncertainty_avg, avg_path, title='Uncertainty (Avg across RGB)', 
                 cmap='hot', vmin=0, vmax=1)
    saved_files.append(avg_path)
    
    # 2. 分别保存各通道
    if save_channels and uncertainty_tensor.size(1) == 3:
        channel_names = ['R', 'G', 'B']
        for i, ch_name in enumerate(channel_names):
            ch_data = uncertainty_tensor[0, i].cpu().numpy()
            ch_path = os.path.join(save_dir, f'{prefix}_ch{ch_name}.png')
            save_heatmap(ch_data, ch_path, title=f'Uncertainty - {ch_name} Channel', 
                        cmap='hot', vmin=0, vmax=1)
            saved_files.append(ch_path)
    
    return saved_files


def visualize_model_internals(model, save_dir, img_name='image'):
    """
    可视化模型推理过程的内部变量（Diff和Uncertainty）
    
    需要在调用sample_func时设置save_intermediate=True
    
    Args:
        model: UPSR模型实例
        save_dir: 保存目录
        img_name: 图像名称（用于文件命名）
        
    Returns:
        dict: 包含保存路径和统计信息的字典
    """
    if not hasattr(model, 'uncertainty_map'):
        print("Warning: Model does not have 'uncertainty_map' attribute.")
        print("Make sure to call sample_func with save_intermediate=True")
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算Diff
    sr_mse = model.sr_mse_pred  # [1, 3, H, W]
    bicubic = model.bicubic_upscale  # [1, 3, H, W]
    diff = (sr_mse - bicubic) / 2
    
    uncertainty = model.uncertainty_map
    
    # 可视化Diff
    diff_files = visualize_diff_heatmap(diff, save_dir, prefix=f'{img_name}_diff')
    
    # 可视化Uncertainty
    uncertainty_files = visualize_uncertainty_heatmap(uncertainty, save_dir, prefix=f'{img_name}_uncertainty')
    
    # 计算统计信息
    stats = {
        'diff': {
            'min': diff.min().item(),
            'max': diff.max().item(),
            'mean': diff.mean().item(),
            'std': diff.std().item(),
        },
        'uncertainty': {
            'min': uncertainty.min().item(),
            'max': uncertainty.max().item(),
            'mean': uncertainty.mean().item(),
            'std': uncertainty.std().item(),
        },
        'diff_files': diff_files,
        'uncertainty_files': uncertainty_files,
    }
    
    # 保存统计信息到文本文件
    stats_path = os.path.join(save_dir, f'{img_name}_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("=== Diff Statistics ===\n")
        f.write(f"Min: {stats['diff']['min']:.6f}\n")
        f.write(f"Max: {stats['diff']['max']:.6f}\n")
        f.write(f"Mean: {stats['diff']['mean']:.6f}\n")
        f.write(f"Std: {stats['diff']['std']:.6f}\n\n")
        
        f.write("=== Uncertainty Statistics ===\n")
        f.write(f"Min: {stats['uncertainty']['min']:.6f}\n")
        f.write(f"Max: {stats['uncertainty']['max']:.6f}\n")
        f.write(f"Mean: {stats['uncertainty']['mean']:.6f}\n")
        f.write(f"Std: {stats['uncertainty']['std']:.6f}\n")
    
    print(f"\n{'='*50}")
    print(f"Diff & Uncertainty visualization completed!")
    print(f"{'='*50}")
    print(f"Diff range: [{stats['diff']['min']:.4f}, {stats['diff']['max']:.4f}]")
    print(f"Uncertainty range: [{stats['uncertainty']['min']:.4f}, {stats['uncertainty']['max']:.4f}]")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*50}\n")
    
    return stats


def create_side_by_side_comparison(sr_img_path, diff_heatmap_path, uncertainty_heatmap_path, 
                                   output_path, lq_img_path=None):
    """
    创建SR图像、Diff热力图和Uncertainty热力图的并排对比图
    
    Args:
        sr_img_path: SR图像路径
        diff_heatmap_path: Diff热力图路径
        uncertainty_heatmap_path: Uncertainty热力图路径
        output_path: 输出路径
        lq_img_path: 可选的LQ图像路径
    """
    from PIL import Image
    
    # 读取图像
    sr_img = Image.open(sr_img_path)
    diff_img = Image.open(diff_heatmap_path)
    uncertainty_img = Image.open(uncertainty_heatmap_path)
    
    images = [sr_img, diff_img, uncertainty_img]
    titles = ['SR Output', 'Diff Heatmap', 'Uncertainty Map']
    
    if lq_img_path is not None:
        lq_img = Image.open(lq_img_path)
        images.insert(0, lq_img)
        titles.insert(0, 'LQ Input')
    
    # 创建并排图
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(6*n_images, 6))
    
    if n_images == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison image saved to: {output_path}")
