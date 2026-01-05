"""
可视化Diff热力图的脚本

在推理过程中获取并可视化Diff(差异图)，输出为热力图。
Diff = (SR_MSE预测 - 双三次插值上采样) / 2

使用方法：
    python visualize_diff_heatmap.py --opt options/test_xxx.yml --input_path path/to/image.png --output_dir results/heatmaps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
import cv2
import os
import argparse
from pathlib import Path
import torch.nn.functional as F
from PIL import Image

from basicsr.models import build_model
from basicsr.utils.options import parse_options
from basicsr.utils import imwrite, tensor2img, img2tensor


def create_heatmap(diff_tensor, save_path, title='Diff Heatmap', cmap='coolwarm', vmin=None, vmax=None):
    """
    创建并保存热力图
    
    Args:
        diff_tensor: [1, 3, H, W] 或 [1, 1, H, W] 的张量，值域通常在[-1, 1]
        save_path: 保存路径
        title: 图像标题
        cmap: 颜色映射 ('coolwarm', 'jet', 'viridis', 'hot', 'seismic' 等)
        vmin, vmax: 值域范围（如果为None则自动计算）
    """
    # 转换为numpy数组
    if diff_tensor.dim() == 4:
        # [1, C, H, W] -> [H, W, C] 或 [H, W]
        diff_np = diff_tensor.squeeze(0).cpu().numpy()
        if diff_np.shape[0] == 3:
            # RGB: 取所有通道的平均作为单通道热力图
            diff_np = np.mean(diff_np, axis=0)  # [H, W]
        elif diff_np.shape[0] == 1:
            diff_np = diff_np[0]  # [H, W]
    else:
        diff_np = diff_tensor.cpu().numpy()
    
    # 创建图像
    plt.figure(figsize=(12, 10))
    
    # 自动确定值域
    if vmin is None:
        vmin = diff_np.min()
    if vmax is None:
        vmax = diff_np.max()
    
    # 创建热力图
    im = plt.imshow(diff_np, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'{title}\nMin: {diff_np.min():.4f}, Max: {diff_np.max():.4f}, Mean: {diff_np.mean():.4f}')
    plt.axis('off')
    plt.tight_layout()
    
    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to: {save_path}")


def create_multi_channel_heatmap(diff_tensor, save_path_prefix, title_prefix='Channel'):
    """
    为RGB三个通道分别创建热力图
    
    Args:
        diff_tensor: [1, 3, H, W] 的张量
        save_path_prefix: 保存路径前缀（不含扩展名）
        title_prefix: 标题前缀
    """
    if diff_tensor.size(1) != 3:
        print(f"Warning: Expected 3 channels, got {diff_tensor.size(1)}")
        return
    
    channel_names = ['Red', 'Green', 'Blue']
    
    for i, ch_name in enumerate(channel_names):
        ch_diff = diff_tensor[:, i:i+1, :, :]  # [1, 1, H, W]
        save_path = f"{save_path_prefix}_{ch_name.lower()}.png"
        create_heatmap(
            ch_diff,
            save_path,
            title=f'{title_prefix} - {ch_name} Channel',
            cmap='coolwarm'
        )


def create_comparison_figure(lq_img, sr_img, diff_map, uncertainty_map, save_path):
    """
    创建包含LQ、SR、Diff热力图和Uncertainty热力图的对比图
    
    Args:
        lq_img: LQ图像 [1, 3, H, W]，值域[0, 1]或[-1, 1]
        sr_img: SR图像 [1, 3, H, W]，值域[0, 1]或[-1, 1]
        diff_map: Diff图 [1, 3, H, W]
        uncertainty_map: Uncertainty图 [1, 3, H, W]
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 归一化图像到[0, 1]
    def normalize_img(img):
        if img.min() < 0:
            img = img * 0.5 + 0.5
        return img
    
    lq_np = tensor2img(normalize_img(lq_img.clone()))
    sr_np = tensor2img(normalize_img(sr_img.clone()))
    
    # 计算平均diff和uncertainty用于热力图
    diff_np = diff_map.squeeze(0).mean(0).cpu().numpy()
    uncertainty_np = uncertainty_map.squeeze(0).mean(0).cpu().numpy()
    
    # 1. LQ图像（上采样到SR尺寸以便对比）
    axes[0, 0].imshow(lq_np)
    axes[0, 0].set_title('LQ Input (Upscaled)', fontsize=14)
    axes[0, 0].axis('off')
    
    # 2. SR输出
    axes[0, 1].imshow(sr_np)
    axes[0, 1].set_title('SR Output', fontsize=14)
    axes[0, 1].axis('off')
    
    # 3. Diff热力图
    im1 = axes[1, 0].imshow(diff_np, cmap='coolwarm', vmin=diff_np.min(), vmax=diff_np.max())
    axes[1, 0].set_title(f'Diff Heatmap\n(Min: {diff_np.min():.4f}, Max: {diff_np.max():.4f})', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 4. Uncertainty热力图
    im2 = axes[1, 1].imshow(uncertainty_np, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Uncertainty Map\n(Min: {uncertainty_np.min():.4f}, Max: {uncertainty_np.max():.4f})', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison figure saved to: {save_path}")


def visualize_diff_from_model(model, lq_tensor, output_dir, img_name='test'):
    """
    从模型推理中提取并可视化Diff
    
    Args:
        model: UPSR模型
        lq_tensor: LQ输入张量 [1, 3, H, W]，值域[0, 1]
        output_dir: 输出目录
        img_name: 图像名称（不含扩展名）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        # 归一化到[-1, 1]
        lq_normalized = (lq_tensor - 0.5) / 0.5
        
        # 执行推理（启用save_intermediate）
        sr_output = model.sample_func(
            lq_normalized, 
            noise_repeat=model.opt['val'].get('noise_repeat', False),
            save_intermediate=True
        )
        
        # 获取中间结果
        if not hasattr(model, 'uncertainty_map'):
            print("Error: Model does not have uncertainty_map. Make sure save_intermediate=True")
            return
        
        # 提取差异图（需要从模型内部计算）
        # Diff = (SR_MSE - Bicubic) / 2
        sr_mse = model.sr_mse_pred  # [1, 3, H, W]
        bicubic = model.bicubic_upscale  # [1, 3, H, W]
        diff = (sr_mse - bicubic) / 2
        
        uncertainty = model.uncertainty_map  # [1, 3, H, W]
        
        # 上采样LQ用于对比显示
        sf = model.opt['scale']
        lq_upscaled = F.interpolate(lq_tensor, scale_factor=sf, mode='bicubic', align_corners=False)
        
        # 1. 创建平均Diff热力图
        create_heatmap(
            diff,
            os.path.join(output_dir, f'{img_name}_diff_heatmap.png'),
            title='Diff Heatmap (Average across channels)',
            cmap='coolwarm'
        )
        
        # 2. 创建各通道的Diff热力图
        create_multi_channel_heatmap(
            diff,
            os.path.join(output_dir, f'{img_name}_diff'),
            title_prefix='Diff'
        )
        
        # 3. 创建Uncertainty热力图
        create_heatmap(
            uncertainty,
            os.path.join(output_dir, f'{img_name}_uncertainty_heatmap.png'),
            title='Uncertainty Map',
            cmap='hot',
            vmin=0,
            vmax=1
        )
        
        # 4. 创建各通道的Uncertainty热力图
        create_multi_channel_heatmap(
            uncertainty,
            os.path.join(output_dir, f'{img_name}_uncertainty'),
            title_prefix='Uncertainty'
        )
        
        # 5. 创建综合对比图
        create_comparison_figure(
            lq_upscaled,
            sr_output,
            diff,
            uncertainty,
            os.path.join(output_dir, f'{img_name}_comparison.png')
        )
        
        # 6. 保存SR结果
        sr_img = tensor2img(sr_output * 0.5 + 0.5)
        imwrite(sr_img, os.path.join(output_dir, f'{img_name}_sr.png'))
        
        # 7. 保存原始数据为.npy（可选，用于进一步分析）
        np.save(os.path.join(output_dir, f'{img_name}_diff.npy'), diff.cpu().numpy())
        np.save(os.path.join(output_dir, f'{img_name}_uncertainty.npy'), uncertainty.cpu().numpy())
        
        print(f"\n=== Statistics ===")
        print(f"Diff - Min: {diff.min().item():.6f}, Max: {diff.max().item():.6f}, Mean: {diff.mean().item():.6f}, Std: {diff.std().item():.6f}")
        print(f"Uncertainty - Min: {uncertainty.min().item():.6f}, Max: {uncertainty.max().item():.6f}, Mean: {uncertainty.mean().item():.6f}")
        print(f"\nAll results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Diff Heatmap during UPSR inference')
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input LQ image')
    parser.add_argument('--output_dir', type=str, default='results/diff_heatmaps', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    args = parser.parse_args()
    
    # 设置GPU
    torch.cuda.set_device(args.gpu)
    
    # 解析配置
    opt, _ = parse_options(args.opt, is_train=False)
    opt['dist'] = False
    opt['num_gpu'] = 1
    
    # 构建模型
    print("Building model...")
    model = build_model(opt)
    
    # 读取输入图像
    print(f"Loading image from: {args.input_path}")
    img_lq = cv2.imread(args.input_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
    
    # 转换为tensor
    img_lq = img2tensor(img_lq, bgr2rgb=False, float32=True)
    img_lq = img_lq.unsqueeze(0).cuda()  # [1, 3, H, W]
    
    # 可视化
    img_name = Path(args.input_path).stem
    visualize_diff_from_model(model, img_lq, args.output_dir, img_name)


if __name__ == '__main__':
    main()
