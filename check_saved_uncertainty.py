#!/usr/bin/env python3
"""
检查已保存的不确定度图，分析数值范围问题
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def analyze_uncertainty_image(img_path):
    """分析单张不确定度图"""
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"❌ Cannot read: {img_path}")
        return None
    
    # 转换到[0,1]范围（假设保存时是8bit）
    img_float = img.astype(np.float32) / 255.0
    
    print(f"\n{'='*70}")
    print(f"Image: {Path(img_path).name}")
    print(f"{'='*70}")
    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    print(f"Value range: [{img.min()}, {img.max()}]")
    print(f"Float range: [{img_float.min():.6f}, {img_float.max():.6f}]")
    print(f"Mean: {img_float.mean():.6f}")
    print(f"Std:  {img_float.std():.6f}")
    
    # 分析每个通道
    if len(img.shape) == 3:
        for c, channel in enumerate(['B', 'G', 'R']):  # OpenCV is BGR
            print(f"\n{channel} Channel:")
            print(f"  Mean: {img_float[:,:,c].mean():.6f}")
            print(f"  Std:  {img_float[:,:,c].std():.6f}")
            print(f"  Min:  {img_float[:,:,c].min():.6f}")
            print(f"  Max:  {img_float[:,:,c].max():.6f}")
    
    # 检测边缘
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = edges > 0
    
    gray_float = gray.astype(np.float32) / 255.0
    
    if edge_mask.sum() > 0:
        edge_mean = gray_float[edge_mask].mean()
        non_edge_mean = gray_float[~edge_mask].mean()
        
        print(f"\n边缘分析:")
        print(f"  边缘像素平均值: {edge_mean:.6f}")
        print(f"  非边缘像素平均值: {non_edge_mean:.6f}")
        print(f"  差异: {edge_mean - non_edge_mean:.6f}")
        
        if edge_mean < non_edge_mean:
            print(f"  ⚠️ 警告：边缘不确定度 < 非边缘不确定度（异常！）")
        else:
            print(f"  ✅ 正常：边缘不确定度 > 非边缘不确定度")
    
    return img_float


def compare_uncertainty_folders(folder1, folder2, name1='Fixed', name2='Learnable'):
    """对比两个文件夹的不确定度图"""
    
    # 找到匹配的图像对
    images1 = sorted(glob.glob(str(Path(folder1) / '*.png')))
    images2 = sorted(glob.glob(str(Path(folder2) / '*.png')))
    
    if not images1:
        print(f"❌ No images found in {folder1}")
        return
    
    if not images2:
        print(f"❌ No images found in {folder2}")
        return
    
    print(f"\n{'='*70}")
    print(f"Comparing: {name1} ({len(images1)} images) vs {name2} ({len(images2)} images)")
    print(f"{'='*70}")
    
    # 分析第一张图
    if images1:
        print(f"\n{'*'*70}")
        print(f"  {name1}")
        print(f"{'*'*70}")
        img1 = analyze_uncertainty_image(images1[0])
    
    if images2:
        print(f"\n{'*'*70}")
        print(f"  {name2}")
        print(f"{'*'*70}")
        img2 = analyze_uncertainty_image(images2[0])
    
    # 可视化对比
    if img1 is not None and img2 is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 转换为灰度用于显示
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
        else:
            gray1 = img1
        
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
        else:
            gray2 = img2
        
        # 第一行：固定映射
        axes[0, 0].imshow(gray1, cmap='jet', vmin=0, vmax=1)
        axes[0, 0].set_title(f'{name1} Uncertainty')
        axes[0, 0].axis('off')
        
        axes[0, 1].hist(gray1.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title(f'{name1} Distribution')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 边缘检测
        edges1 = cv2.Canny((gray1 * 255).astype(np.uint8), 50, 150)
        axes[0, 2].imshow(edges1, cmap='gray')
        axes[0, 2].set_title(f'{name1} Edges')
        axes[0, 2].axis('off')
        
        # 第二行：可学习映射
        axes[1, 0].imshow(gray2, cmap='jet', vmin=0, vmax=1)
        axes[1, 0].set_title(f'{name2} Uncertainty')
        axes[1, 0].axis('off')
        
        axes[1, 1].hist(gray2.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title(f'{name2} Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 边缘检测
        edges2 = cv2.Canny((gray2 * 255).astype(np.uint8), 50, 150)
        axes[1, 2].imshow(edges2, cmap='gray')
        axes[1, 2].set_title(f'{name2} Edges')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('./uncertainty_comparison_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ Comparison saved to: ./uncertainty_comparison_analysis.png")
        plt.close()


def main():
    print("\n" + "="*70)
    print("  Uncertainty Image Analysis Tool")
    print("="*70)
    
    # 请根据实际路径修改
    results_dir = Path('./results')
    
    # 查找noise1文件夹
    noise1_folders = list(results_dir.glob('**/noise1'))
    
    if noise1_folders:
        print(f"\n找到 {len(noise1_folders)} 个noise1文件夹:")
        for folder in noise1_folders:
            print(f"  - {folder}")
        
        # 分析第一个文件夹
        if len(noise1_folders) > 0:
            print(f"\n分析文件夹: {noise1_folders[0]}")
            images = sorted(glob.glob(str(noise1_folders[0] / '*.png')))
            
            if images:
                for i, img_path in enumerate(images[:3]):  # 分析前3张
                    analyze_uncertainty_image(img_path)
                    
                    if i < 2:  # 不是最后一张时添加分隔
                        print("\n" + "-"*70 + "\n")
            else:
                print(f"❌ 文件夹中没有PNG图像: {noise1_folders[0]}")
    else:
        print("\n❌ 未找到noise1文件夹")
        print("请手动指定不确定度图的路径")
        print("\n使用方法:")
        print("  python check_saved_uncertainty.py")
        print("\n或者修改脚本中的路径")
    
    # 如果有两个不同的结果文件夹，可以对比
    # compare_uncertainty_folders(
    #     './results/test_stage2_DIV2K_wo/noise1',
    #     './results/test_stage2_DIV2K_content_lq/noise1',
    #     name1='Fixed Mapping',
    #     name2='Content-Aware Mapping'
    # )


if __name__ == '__main__':
    main()
