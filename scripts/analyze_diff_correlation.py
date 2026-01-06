import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)  # [1, 3, H, W]
    return img * 2 - 1  # 归一化到[-1, 1]

def analyze_diff_correlation(lq_path, gt_path, sr_path):
    """分析diff和diff_gt的相关性"""
    
    # 加载图像
    lq = load_image(lq_path)
    gt = load_image(gt_path)
    sr = load_image(sr_path)
    
    print(f"📏 Image shapes:")
    print(f"  LQ: {lq.shape}")
    print(f"  GT: {gt.shape}")
    print(f"  SR: {sr.shape}")
    
    # ✅ 如果SR尺寸大于GT（因为padding），裁剪SR到GT尺寸
    if sr.shape[-2:] != gt.shape[-2:]:
        print(f"⚠️  SR size mismatch, cropping SR from {sr.shape[-2:]} to {gt.shape[-2:]}")
        sr = sr[:, :, :gt.shape[-2], :gt.shape[-1]]
    
    # 上采样LQ
    lq_bicubic = F.interpolate(lq, size=gt.shape[-2:], mode='bicubic', align_corners=False)
    
    # 计算两种diff
    diff = (sr - lq_bicubic) / 2  # 基于bicubic
    diff_gt = (sr - gt) / 2        # 基于GT
    
    # 转换为numpy
    diff_np = diff.squeeze().abs().mean(dim=0).cpu().numpy()
    diff_gt_np = diff_gt.squeeze().abs().mean(dim=0).cpu().numpy()
    
    # 计算统计量
    print("="*60)
    print("Diff Statistics:")
    print(f"  diff (bicubic):  min={diff_np.min():.4f}, max={diff_np.max():.4f}, mean={diff_np.mean():.4f}, std={diff_np.std():.4f}")
    print(f"  diff_gt (GT):    min={diff_gt_np.min():.4f}, max={diff_gt_np.max():.4f}, mean={diff_gt_np.mean():.4f}, std={diff_gt_np.std():.4f}")
    
    # 计算相关性
    flat_diff = diff_np.flatten()
    flat_diff_gt = diff_gt_np.flatten()
    
    pearson_corr, p_value = pearsonr(flat_diff, flat_diff_gt)
    print(f"\nPearson Correlation: {pearson_corr:.4f} (p={p_value:.2e})")
    
    # 计算MSE
    mse = np.mean((flat_diff - flat_diff_gt) ** 2)
    print(f"MSE between diff and diff_gt: {mse:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Diff maps
    im1 = axes[0, 0].imshow(diff_np, cmap='hot')
    axes[0, 0].set_title('diff (bicubic-based)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(diff_gt_np, cmap='hot')
    axes[0, 1].set_title('diff_gt (GT-based)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(np.abs(diff_np - diff_gt_np), cmap='hot')
    axes[0, 2].set_title('Absolute Difference')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Row 2: Analysis
    axes[1, 0].hist(flat_diff, bins=50, alpha=0.7, label='diff', density=True)
    axes[1, 0].hist(flat_diff_gt, bins=50, alpha=0.7, label='diff_gt', density=True)
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(flat_diff[::100], flat_diff_gt[::100], alpha=0.1, s=1)
    axes[1, 1].plot([0, max(flat_diff.max(), flat_diff_gt.max())], 
                     [0, max(flat_diff.max(), flat_diff_gt.max())], 
                     'r--', label='y=x')
    axes[1, 1].set_xlabel('diff (bicubic)')
    axes[1, 1].set_ylabel('diff_gt (GT)')
    axes[1, 1].set_title(f'Correlation (r={pearson_corr:.3f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 区域统计
    bins = np.linspace(0, max(flat_diff.max(), flat_diff_gt.max()), 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    diff_means = []
    diff_gt_means = []
    
    for i in range(len(bins) - 1):
        mask = (flat_diff >= bins[i]) & (flat_diff < bins[i+1])
        if mask.sum() > 0:
            diff_means.append(flat_diff[mask].mean())
            diff_gt_means.append(flat_diff_gt[mask].mean())
        else:
            diff_means.append(np.nan)
            diff_gt_means.append(np.nan)
    
    axes[1, 2].plot(bin_centers, diff_means, 'o-', label='diff', markersize=8)
    axes[1, 2].plot(bin_centers, diff_gt_means, 's-', label='diff_gt', markersize=8)
    axes[1, 2].plot([0, bin_centers[-1]], [0, bin_centers[-1]], 'r--', alpha=0.5)
    axes[1, 2].set_xlabel('diff (bicubic) bin')
    axes[1, 2].set_ylabel('Mean value in bin')
    axes[1, 2].set_title('Binned Statistics')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/diff_correlation_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: results/diff_correlation_analysis.png")
    
    return pearson_corr, mse

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python analyze_diff_correlation.py <lq_path> <gt_path> <sr_path>")
        print("\nExample:")
        print("  python analyze_diff_correlation.py \\")
        print("    /dataset/DIV2K/valid_LR_bicubic/X4/0801x4.png \\")
        print("    /dataset/DIV2K/valid_HR/0801.png \\")
        print("    results/test_stage2_30k/visualization/0801_UPSR_x4.png")
        sys.exit(1)
    
    lq_path = sys.argv[1]
    gt_path = sys.argv[2]
    sr_path = sys.argv[3]
    
    corr, mse = analyze_diff_correlation(lq_path, gt_path, sr_path)
    
    # 评估
    print("\n" + "="*60)
    print("Evaluation:")
    if corr > 0.7:
        print(f"✅ Strong correlation (r={corr:.3f}): diff can predict diff_gt well")
    elif corr > 0.4:
        print(f"⚠️  Moderate correlation (r={corr:.3f}): diff partially predicts diff_gt")
    else:
        print(f"❌ Weak correlation (r={corr:.3f}): diff cannot predict diff_gt!")
        print("   → Current training strategy may have issues!")