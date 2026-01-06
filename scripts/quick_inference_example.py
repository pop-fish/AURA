"""
快速示例：在推理时生成Diff热力图

这个脚本展示了如何在单张图像推理时生成并保存Diff热力图
"""

import torch
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from basicsr.utils import img2tensor
from basicsr.utils.diff_visualization import visualize_model_internals


def quick_inference_with_heatmap(model, image_path, output_dir='results/quick_test'):
    """
    对单张图像进行推理并生成Diff热力图
    
    Args:
        model: 已加载的UPSR模型
        image_path: 输入图像路径
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"处理图像: {image_path}")
    print(f"{'='*60}\n")
    
    # 1. 读取图像
    print("📖 读取图像...")
    img_lq = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
    
    # 2. 转换为tensor
    img_lq_tensor = img2tensor(img_lq, bgr2rgb=False, float32=True)
    if isinstance(img_lq_tensor, list):
        img_lq_tensor = img_lq_tensor[0]
    img_lq = img_lq_tensor.unsqueeze(0).cuda()  # [1, 3, H, W]
    
    print(f"✅ 输入图像尺寸: {img_lq.shape}")
    
    # 3. 执行推理（启用中间结果保存）
    print("🔄 执行推理...")
    with torch.no_grad():
        lq_normalized = (img_lq - 0.5) / 0.5
        sr_output = model.sample_func(
            lq_normalized,
            noise_repeat=model.opt['val'].get('noise_repeat', False),
            save_intermediate=True  # ✅ 关键：启用中间结果保存
        )
    
    print(f"✅ SR输出尺寸: {sr_output.shape}")
    
    # 4. 生成并保存热力图
    print("🎨 生成热力图...")
    img_name = Path(image_path).stem
    stats = visualize_model_internals(
        model,
        save_dir=output_dir,
        img_name=img_name
    )
    
    # 5. 保存SR结果
    from basicsr.utils import tensor2img, imwrite
    sr_img = tensor2img(sr_output * 0.5 + 0.5)
    sr_path = os.path.join(output_dir, f'{img_name}_sr.png')
    imwrite(sr_img, sr_path)
    print(f"✅ SR结果已保存: {sr_path}")
    
    return stats


def main():
    """主函数示例"""
    import argparse
    from basicsr.models import build_model
    from basicsr.utils.options import parse_options
    
    parser = argparse.ArgumentParser(description='Quick inference with diff heatmap')
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='results/quick_test', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    args = parser.parse_args()
    
    # 设置GPU
    torch.cuda.set_device(args.gpu)
    
    # 解析配置
    print("📋 加载配置...")
    opt, _ = parse_options(args.opt, is_train=False)
    opt['dist'] = False
    opt['num_gpu'] = 1
    
    # 构建模型
    print("🏗️  构建模型...")
    model = build_model(opt)
    
    # 执行推理并生成热力图
    stats = quick_inference_with_heatmap(model, args.input, args.output)
    
    print("\n" + "="*60)
    print("🎉 完成！")
    print("="*60)
    print(f"\n所有结果已保存到: {args.output}")
    print("\n生成的文件包括:")
    print(f"  - SR结果图像")
    print(f"  - Diff平均热力图和各通道热力图")
    print(f"  - Uncertainty平均热力图和各通道热力图")
    print(f"  - 统计信息文本文件")
    print()


if __name__ == '__main__':
    # 如果直接运行（不带参数），显示帮助信息
    if len(sys.argv) == 1:
        print("""
╔════════════════════════════════════════════════════════════════╗
║     快速推理并生成Diff热力图                                     ║
╚════════════════════════════════════════════════════════════════╝

使用方法:
    python quick_inference_example.py \\
        -opt options/test_upsr.yml \\
        --input path/to/image.png \\
        --output results/heatmaps \\
        --gpu 0

参数说明:
    -opt     : 测试配置文件路径 (必需)
    --input  : 输入LQ图像路径 (必需)
    --output : 输出目录 (默认: results/quick_test)
    --gpu    : GPU编号 (默认: 0)

示例:
    # 对单张图像生成热力图
    python quick_inference_example.py \\
        -opt options/test/test_upsr_x4.yml \\
        --input datasets/Set5/LR_bicubic/X4/baby.png \\
        --output results/baby_heatmap

输出文件:
    results/baby_heatmap/
    ├── baby_sr.png                    # SR结果
    ├── baby_diff_avg.png              # Diff平均热力图
    ├── baby_diff_chR.png              # Diff红色通道热力图
    ├── baby_diff_chG.png              # Diff绿色通道热力图
    ├── baby_diff_chB.png              # Diff蓝色通道热力图
    ├── baby_uncertainty_avg.png       # Uncertainty平均热力图
    ├── baby_uncertainty_chR.png       # Uncertainty红色通道热力图
    ├── baby_uncertainty_chG.png       # Uncertainty绿色通道热力图
    ├── baby_uncertainty_chB.png       # Uncertainty蓝色通道热力图
    └── baby_stats.txt                 # 统计信息

更多信息请参考: DIFF_HEATMAP_GUIDE.md
        """)
        sys.exit(0)
    
    main()
