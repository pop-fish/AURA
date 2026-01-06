"""
可视化对比脚本

对比显示：
- 原始LR图像
- SR重建结果
- 不确定度图（noise1）
- 加噪起点（noise2）

使用方法：
    python visualize_results.py --result_dir results/your_experiment/Set5 --image_name baby
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_image(path):
    """加载图像"""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def visualize_comparison(sr_path, noise1_path, noise2_path, output_path=None, image_name="Image"):
    """
    可视化对比图
    
    Args:
        sr_path: SR结果路径
        noise1_path: 不确定度图路径
        noise2_path: 加噪起点路径
        output_path: 保存路径（可选）
        image_name: 图像名称
    """
    # 加载图像
    sr_img = load_image(sr_path)
    noise1_img = load_image(noise1_path)
    noise2_img = load_image(noise2_path)
    
    # 检查哪些图像存在
    images = []
    titles = []
    
    if sr_img is not None:
        images.append(sr_img)
        titles.append("SR Result")
    
    if noise1_img is not None:
        images.append(noise1_img)
        titles.append("Uncertainty Map")
    
    if noise2_img is not None:
        images.append(noise2_img)
        titles.append("Noisy Start")
    
    if not images:
        print(f"错误：没有找到任何图像！")
        return
    
    # 创建子图
    n_images = len(images)
    fig = plt.figure(figsize=(6 * n_images, 6))
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(1, n_images, idx + 1)
        ax.imshow(img)
        ax.set_title(f"{title}", fontsize=16, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(f"Visualization: {image_name}", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def batch_visualize(result_dir, output_dir=None, max_images=10):
    """
    批量可视化整个目录
    
    Args:
        result_dir: 结果目录（包含SR结果、noise1、noise2）
        output_dir: 输出目录
        max_images: 最多处理的图像数量
    """
    result_path = Path(result_dir)
    
    # 查找SR图像
    sr_images = sorted([f for f in result_path.glob("*.png") if f.is_file()])
    
    if not sr_images:
        print(f"错误：在 {result_dir} 中没有找到图像！")
        return
    
    print(f"找到 {len(sr_images)} 张图像")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理每张图像
    for idx, sr_path in enumerate(sr_images[:max_images]):
        image_name = sr_path.stem
        print(f"\n处理 [{idx+1}/{min(len(sr_images), max_images)}]: {image_name}")
        
        # 构建路径
        noise1_path = result_path / "noise1" / sr_path.name
        noise2_path = result_path / "noise2" / sr_path.name
        
        # 检查文件
        print(f"  - SR: {'✓' if sr_path.exists() else '✗'}")
        print(f"  - Uncertainty: {'✓' if noise1_path.exists() else '✗'}")
        print(f"  - Noisy Start: {'✓' if noise2_path.exists() else '✗'}")
        
        # 生成可视化
        if output_dir:
            out_path = os.path.join(output_dir, f"{image_name}_comparison.png")
        else:
            out_path = None
        
        visualize_comparison(
            str(sr_path),
            str(noise1_path),
            str(noise2_path),
            output_path=out_path,
            image_name=image_name
        )


def analyze_uncertainty(noise1_path):
    """
    分析不确定度图的统计信息
    
    Args:
        noise1_path: 不确定度图路径
    """
    img = load_image(noise1_path)
    if img is None:
        print(f"无法加载图像: {noise1_path}")
        return
    
    # 转换为灰度（取平均）
    gray = img.mean(axis=2) / 255.0  # 归一化到 [0, 1]
    
    print("\n不确定度统计：")
    print(f"  - 最小值: {gray.min():.4f}")
    print(f"  - 最大值: {gray.max():.4f}")
    print(f"  - 平均值: {gray.mean():.4f}")
    print(f"  - 标准差: {gray.std():.4f}")
    print(f"  - 中位数: {np.median(gray):.4f}")
    
    # 分位数
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\n分位数：")
    for p in percentiles:
        val = np.percentile(gray, p)
        print(f"  - {p}%: {val:.4f}")
    
    # 直方图
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Uncertainty')
    plt.title('Uncertainty Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.hist(gray.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Uncertainty Value')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='可视化UPSR测试结果')
    parser.add_argument('--result_dir', type=str, required=True, 
                      help='结果目录路径，如 results/UPSR_x4/Set5')
    parser.add_argument('--image_name', type=str, default=None,
                      help='指定图像名称（不含扩展名），如 baby。留空则批量处理')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='输出目录。留空则显示而不保存')
    parser.add_argument('--max_images', type=int, default=10,
                      help='批量处理时的最大图像数量')
    parser.add_argument('--analyze', action='store_true',
                      help='分析不确定度统计信息')
    
    args = parser.parse_args()
    
    if args.image_name:
        # 单张图像处理
        result_path = Path(args.result_dir)
        
        # 查找匹配的文件
        sr_files = list(result_path.glob(f"{args.image_name}*.png"))
        if not sr_files:
            print(f"错误：找不到匹配 '{args.image_name}' 的图像")
            return
        
        sr_path = sr_files[0]
        noise1_path = result_path / "noise1" / sr_path.name
        noise2_path = result_path / "noise2" / sr_path.name
        
        print(f"处理图像: {sr_path.name}")
        
        # 可视化
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, f"{args.image_name}_comparison.png")
        else:
            out_path = None
        
        visualize_comparison(
            str(sr_path),
            str(noise1_path),
            str(noise2_path),
            output_path=out_path,
            image_name=args.image_name
        )
        
        # 分析不确定度
        if args.analyze and noise1_path.exists():
            analyze_uncertainty(str(noise1_path))
    
    else:
        # 批量处理
        print(f"批量处理目录: {args.result_dir}")
        batch_visualize(args.result_dir, args.output_dir, args.max_images)


if __name__ == '__main__':
    main()
