"""
分析 UPSR 不确定度设计的哲学
验证：为什么"反直觉"的映射反而有效

使用方法：
    python analyze_uncertainty_philosophy.py results/test_stage2_30k/RealSRV3
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os

def analyze_regions(result_dir, max_images=3):
    """
    分析不同区域的重建质量
    
    Args:
        result_dir: 结果目录
        max_images: 最多分析的图像数量
    """
    result_path = Path(result_dir)
    
    if not result_path.exists():
        print(f"错误：目录不存在 {result_dir}")
        return
    
    sr_images = sorted([f for f in result_path.glob('*.png') if f.is_file()])
    
    if not sr_images:
        print(f"错误：在 {result_dir} 中没有找到图像")
        return
    
    print(f"\n找到 {len(sr_images)} 张图像，将分析前 {min(len(sr_images), max_images)} 张\n")
    
    for idx, img_path in enumerate(sr_images[:max_images]):
        img_name = img_path.stem
        print(f"\n{'='*70}")
        print(f"[{idx+1}/{min(len(sr_images), max_images)}] 分析图像: {img_name}")
        print(f"{'='*70}")
        
        # 读取所有相关图像
        sr_img = cv2.imread(str(img_path))
        if sr_img is None:
            print(f"⚠️  无法读取SR图像: {img_path}")
            continue
            
        sr_gray = cv2.cvtColor(sr_img, cv2.COLOR_BGR2GRAY)
        
        uncertainty_path = result_path / 'noise1' / f'{img_path.name}'
        noisy_start_path = result_path / 'noise2' / f'{img_path.name}'
        
        if not uncertainty_path.exists():
            print(f"⚠️  跳过 {img_name}：未找到不确定度图 (noise1)")
            continue
            
        if not noisy_start_path.exists():
            print(f"⚠️  跳过 {img_name}：未找到加噪起点 (noise2)")
            continue
        
        uncertainty = cv2.imread(str(uncertainty_path), cv2.IMREAD_GRAYSCALE)
        noisy_start = cv2.imread(str(noisy_start_path))
        
        if uncertainty is None or noisy_start is None:
            print(f"⚠️  无法读取中间结果")
            continue
        
        print(f"✓ 图像尺寸: {sr_img.shape[:2]}")
        
        # 计算梯度（表示纹理复杂度）
        sobelx = cv2.Sobel(sr_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(sr_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        gradient_norm = (gradient / (gradient.max() + 1e-6) * 255).astype(np.uint8)
        
        # 定义区域：使用中位数作为阈值
        grad_threshold = np.percentile(gradient, 50)
        
        flat_mask = gradient < grad_threshold  # 平坦区域（低梯度）
        texture_mask = gradient >= grad_threshold  # 纹理区域（高梯度）
        
        # 分析每个区域的不确定度
        flat_uncertainty = uncertainty[flat_mask].mean()
        texture_uncertainty = uncertainty[texture_mask].mean()
        
        flat_uncertainty_std = uncertainty[flat_mask].std()
        texture_uncertainty_std = uncertainty[texture_mask].std()
        
        # 分析噪声起点的统计特性
        noisy_gray = cv2.cvtColor(noisy_start, cv2.COLOR_BGR2GRAY)
        flat_noise_mean = noisy_gray[flat_mask].mean()
        texture_noise_mean = noisy_gray[texture_mask].mean()
        flat_noise_std = noisy_gray[flat_mask].std()
        texture_noise_std = noisy_gray[texture_mask].std()
        
        # 打印分析结果
        print(f"\n📊 区域统计分析：")
        print(f"\n1️⃣  平坦区域（低梯度，占比 {flat_mask.sum()/flat_mask.size*100:.1f}%）：")
        print(f"   - 不确定度: {flat_uncertainty:.2f} ± {flat_uncertainty_std:.2f}")
        print(f"   - 噪声水平: {flat_noise_mean:.2f} ± {flat_noise_std:.2f}")
        
        print(f"\n2️⃣  纹理区域（高梯度，占比 {texture_mask.sum()/texture_mask.size*100:.1f}%）：")
        print(f"   - 不确定度: {texture_uncertainty:.2f} ± {texture_uncertainty_std:.2f}")
        print(f"   - 噪声水平: {texture_noise_mean:.2f} ± {texture_noise_std:.2f}")
        
        # 计算比值
        ratio = flat_uncertainty / (texture_uncertainty + 1e-6)
        print(f"\n3️⃣  不确定度比值（平坦/纹理）: {ratio:.3f}")
        
        # 计算相关系数
        corr = np.corrcoef(gradient.flatten(), uncertainty.flatten())[0, 1]
        print(f"4️⃣  梯度 vs 不确定度相关系数: {corr:.3f}")
        
        # 解释结果
        print(f"\n{'='*70}")
        print(f"💡 设计哲学解释：")
        print(f"{'='*70}")
        
        if flat_uncertainty > texture_uncertainty:
            print(f"✓ 发现：平坦区域不确定度更高 ({flat_uncertainty:.1f} vs {texture_uncertainty:.1f})")
            print(f"\n  这意味着：")
            print(f"  • UPSR 对 MSE 模型在平坦区域的预测信任度较低")
            print(f"  • 扩散策略：让扩散模型在平坦区域发挥更大作用")
            print(f"  • 预期效果：防止过度平滑，生成更自然的细微纹理")
            print(f"\n  为什么这样有效：")
            print(f"  1. MSE 损失倾向于产生平滑的平均结果")
            print(f"  2. 平坦区域最容易出现 over-smoothing")
            print(f"  3. 高不确定度 → 更多扩散噪声 → 改善平滑问题")
        else:
            print(f"✓ 发现：纹理区域不确定度更高 ({texture_uncertainty:.1f} vs {flat_uncertainty:.1f})")
            print(f"\n  这意味着：")
            print(f"  • UPSR 对 MSE 模型在纹理区域的预测信任度较低")
            print(f"  • 扩散策略：让扩散模型在纹理区域发挥更大作用")
            print(f"  • 预期效果：增强细节和纹理的锐度")
        
        if abs(corr) < 0.2:
            print(f"\n  ⚠️  相关系数接近0 ({corr:.3f})：")
            print(f"  • 不确定度与梯度几乎无关")
            print(f"  • 可能使用了更复杂的映射逻辑")
        elif corr < -0.2:
            print(f"\n  📈 负相关 ({corr:.3f})：")
            print(f"  • 梯度低（平坦）→ 不确定度高")
            print(f"  • 符合'修正过度平滑'的设计理念")
        else:
            print(f"\n  📈 正相关 ({corr:.3f})：")
            print(f"  • 梯度高（纹理）→ 不确定度高")
            print(f"  • 符合'增强细节'的设计理念")
        
        print(f"{'='*70}")
        
        # ============= 可视化 =============
        try:
            fig = plt.figure(figsize=(20, 14))
            gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
            
            # 第一行：原始图像和分析
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB))
            ax1.set_title('SR Result', fontsize=13, fontweight='bold')
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(gradient_norm, cmap='hot')
            ax2.set_title('Gradient Magnitude\n(纹理强度)', fontsize=13, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            ax3 = fig.add_subplot(gs[0, 2])
            im3 = ax3.imshow(uncertainty, cmap='hot', vmin=0, vmax=255)
            ax3.set_title('Uncertainty Map\n(不确定度)', fontsize=13, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.imshow(cv2.cvtColor(noisy_start, cv2.COLOR_BGR2RGB))
            ax4.set_title('Noisy Start\n(扩散起点)', fontsize=13, fontweight='bold')
            ax4.axis('off')
            
            # 第二行：区域分析
            ax5 = fig.add_subplot(gs[1, 0])
            flat_mask_vis = flat_mask.astype(np.uint8) * 255
            ax5.imshow(flat_mask_vis, cmap='gray')
            ax5.set_title(f'平坦区域 (低梯度)\n占比: {flat_mask.sum()/flat_mask.size*100:.1f}%', 
                         fontsize=12, fontweight='bold', color='blue')
            ax5.axis('off')
            
            ax6 = fig.add_subplot(gs[1, 1])
            flat_unc_vis = np.zeros_like(uncertainty)
            flat_unc_vis[flat_mask] = uncertainty[flat_mask]
            im6 = ax6.imshow(flat_unc_vis, cmap='hot', vmin=0, vmax=255)
            ax6.set_title(f'平坦区域的不确定度\n均值: {flat_uncertainty:.1f}±{flat_uncertainty_std:.1f}', 
                         fontsize=12, color='blue')
            ax6.axis('off')
            plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
            
            ax7 = fig.add_subplot(gs[1, 2])
            texture_mask_vis = texture_mask.astype(np.uint8) * 255
            ax7.imshow(texture_mask_vis, cmap='gray')
            ax7.set_title(f'纹理区域 (高梯度)\n占比: {texture_mask.sum()/texture_mask.size*100:.1f}%', 
                         fontsize=12, fontweight='bold', color='red')
            ax7.axis('off')
            
            ax8 = fig.add_subplot(gs[1, 3])
            texture_unc_vis = np.zeros_like(uncertainty)
            texture_unc_vis[texture_mask] = uncertainty[texture_mask]
            im8 = ax8.imshow(texture_unc_vis, cmap='hot', vmin=0, vmax=255)
            ax8.set_title(f'纹理区域的不确定度\n均值: {texture_uncertainty:.1f}±{texture_uncertainty_std:.1f}', 
                         fontsize=12, color='red')
            ax8.axis('off')
            plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
            
            # 第三行：统计分析
            ax9 = fig.add_subplot(gs[2, :2])
            
            # 直方图对比
            bins = np.linspace(0, 255, 50)
            ax9.hist(uncertainty[flat_mask].flatten(), bins=bins, alpha=0.6, 
                    label=f'平坦区域 (均值={flat_uncertainty:.1f})', 
                    color='blue', density=True, edgecolor='black', linewidth=0.5)
            ax9.hist(uncertainty[texture_mask].flatten(), bins=bins, alpha=0.6,
                    label=f'纹理区域 (均值={texture_uncertainty:.1f})', 
                    color='red', density=True, edgecolor='black', linewidth=0.5)
            ax9.axvline(flat_uncertainty, color='blue', linestyle='--', linewidth=2, alpha=0.8)
            ax9.axvline(texture_uncertainty, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax9.set_xlabel('不确定度值', fontsize=12)
            ax9.set_ylabel('概率密度', fontsize=12)
            ax9.set_title('不确定度分布对比', fontsize=13, fontweight='bold')
            ax9.legend(fontsize=11, loc='upper right')
            ax9.grid(True, alpha=0.3)
            
            # 散点图：梯度 vs 不确定度
            ax10 = fig.add_subplot(gs[2, 2:])
            sample_size = min(10000, gradient.size)
            sample_indices = np.random.choice(gradient.size, sample_size, replace=False)
            grad_samples = gradient.flatten()[sample_indices]
            unc_samples = uncertainty.flatten()[sample_indices]
            
            # 按梯度大小着色
            colors = grad_samples / (grad_samples.max() + 1e-6)
            scatter = ax10.scatter(grad_samples, unc_samples, 
                                  c=colors, cmap='viridis', 
                                  alpha=0.3, s=3)
            
            # 拟合趋势线
            try:
                z = np.polyfit(grad_samples, unc_samples, 2)
                p = np.poly1d(z)
                x_line = np.linspace(grad_samples.min(), grad_samples.max(), 100)
                ax10.plot(x_line, p(x_line), "r-", linewidth=3, label='趋势线', alpha=0.8)
            except:
                pass
            
            ax10.set_xlabel('梯度强度 (纹理复杂度)', fontsize=12)
            ax10.set_ylabel('不确定度', fontsize=12)
            ax10.set_title(f'梯度 vs 不确定度散点图\n相关系数: {corr:.3f}', 
                          fontsize=13, fontweight='bold')
            ax10.legend(fontsize=11)
            ax10.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax10, label='梯度强度', fraction=0.046, pad=0.04)
            
            # 添加解释文本框
            interpretation = ""
            if corr < -0.3:
                interpretation = "🔍 负相关明显\n平坦区域 → 高不确定度\n让扩散模型改善平滑区域"
                box_color = 'lightblue'
            elif corr > 0.3:
                interpretation = "🔍 正相关明显\n纹理区域 → 高不确定度\n让扩散模型增强细节"
                box_color = 'lightcoral'
            else:
                interpretation = "🔍 弱相关\n不确定度与纹理复杂度\n关系不明显"
                box_color = 'lightyellow'
            
            ax10.text(0.05, 0.95, interpretation, transform=ax10.transAxes,
                     verticalalignment='top', fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
            
            # 总标题
            title_text = f'UPSR 不确定度设计哲学分析\n'
            title_text += f'图像: {img_name} | '
            title_text += f'不确定度比值(平坦/纹理): {ratio:.3f} | '
            title_text += f'相关系数: {corr:.3f}'
            plt.suptitle(title_text, fontsize=15, fontweight='bold', y=0.995)
            
            # 保存图像
            output_path = result_path / f'philosophy_analysis_{img_name}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"\n✅ 分析图已保存: {output_path.name}")
            
        except Exception as e:
            print(f"\n⚠️  可视化过程出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 最终总结
    print(f"\n{'='*70}")
    print(f"🎯 UPSR 设计哲学总结")
    print(f"{'='*70}")
    print(f"\n核心概念：")
    print(f"  不确定度 ≠ 区域复杂度")
    print(f"  不确定度 = 对MSE预测的信任度的倒数")
    print(f"\n为什么'反直觉'的设计反而有效：")
    print(f"  1️⃣  MSE模型的特性：")
    print(f"     • 倾向于产生平滑、安全的预测")
    print(f"     • 在平坦区域容易过度平滑（over-smoothing）")
    print(f"  2️⃣  UPSR的解决方案：")
    print(f"     • 平坦区域 → 高不确定度 → 更多扩散噪声")
    print(f"     • 修正过度平滑，生成自然纹理")
    print(f"  3️⃣  为什么指标会提升：")
    print(f"     • PSNR/SSIM: 避免过度平滑导致的失真")
    print(f"     • LPIPS: 感知质量更自然")
    print(f"     • 整体: 自适应平衡MSE和扩散")
    print(f"\n关键insight：")
    print(f"  不是'哪里复杂就加强哪里'")
    print(f"  而是'哪里MSE表现不好就用扩散补偿'")
    print(f"{'='*70}\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析UPSR不确定度设计哲学')
    parser.add_argument('result_dir', type=str, nargs='?',
                       default='results/test_stage2_30k/RealSRV3',
                       help='结果目录路径')
    parser.add_argument('--max_images', type=int, default=3,
                       help='最多分析的图像数量')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("UPSR 不确定度设计哲学分析工具")
    print("="*70)
    print(f"\n📁 分析目录: {args.result_dir}")
    print(f"🖼️  最多分析: {args.max_images} 张图像\n")
    
    if not os.path.exists(args.result_dir):
        print(f"❌ 错误：目录不存在 {args.result_dir}")
        print(f"\n💡 提示：请先运行测试生成结果")
        print(f"   python test_noise_visualization.py -opt options/test_stage2.yml")
        return
    
    analyze_regions(args.result_dir, args.max_images)
    
    print("\n✅ 所有分析完成！")
    print(f"📊 查看生成的 philosophy_analysis_*.png 图像获取详细可视化结果")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
