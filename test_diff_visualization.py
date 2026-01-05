"""
测试Diff热力图可视化功能

快速测试脚本，验证热力图生成功能是否正常工作
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from basicsr.utils.diff_visualization import (
    visualize_diff_heatmap,
    visualize_uncertainty_heatmap,
    save_heatmap,
    tensor_to_heatmap_data
)


def test_basic_heatmap():
    """测试基本热力图生成功能"""
    print("=" * 60)
    print("测试1: 基本热力图生成")
    print("=" * 60)
    
    # 创建测试数据
    test_data = np.random.randn(256, 256) * 0.5  # [-1, 1]范围的随机数据
    
    # 保存热力图
    output_dir = 'test_heatmap_output'
    os.makedirs(output_dir, exist_ok=True)
    
    save_heatmap(
        test_data, 
        os.path.join(output_dir, 'test_basic_heatmap.png'),
        title='Test Basic Heatmap',
        cmap='coolwarm'
    )
    
    print(f"✅ 基本热力图已保存到: {output_dir}/test_basic_heatmap.png")
    print()


def test_diff_visualization():
    """测试Diff可视化功能"""
    print("=" * 60)
    print("测试2: Diff张量可视化")
    print("=" * 60)
    
    # 创建模拟的Diff张量 [1, 3, H, W]
    diff_tensor = torch.randn(1, 3, 256, 256) * 0.5
    
    output_dir = 'test_heatmap_output'
    
    # 可视化Diff
    files = visualize_diff_heatmap(
        diff_tensor,
        output_dir,
        prefix='test_diff',
        save_channels=True
    )
    
    print(f"✅ 生成了 {len(files)} 个Diff热力图文件:")
    for f in files:
        print(f"   - {f}")
    print()


def test_uncertainty_visualization():
    """测试Uncertainty可视化功能"""
    print("=" * 60)
    print("测试3: Uncertainty张量可视化")
    print("=" * 60)
    
    # 创建模拟的Uncertainty张量 [1, 3, H, W]，范围[0, 1]
    uncertainty_tensor = torch.rand(1, 3, 256, 256)
    
    output_dir = 'test_heatmap_output'
    
    # 可视化Uncertainty
    files = visualize_uncertainty_heatmap(
        uncertainty_tensor,
        output_dir,
        prefix='test_uncertainty',
        save_channels=True
    )
    
    print(f"✅ 生成了 {len(files)} 个Uncertainty热力图文件:")
    for f in files:
        print(f"   - {f}")
    print()


def test_tensor_conversion():
    """测试张量转换功能"""
    print("=" * 60)
    print("测试4: 张量转换")
    print("=" * 60)
    
    # 测试4D张量
    tensor_4d = torch.randn(1, 3, 64, 64)
    data_avg = tensor_to_heatmap_data(tensor_4d, aggregate_channels=True)
    print(f"✅ 4D张量 {tensor_4d.shape} -> {data_avg.shape} (平均)")
    
    data_separate = tensor_to_heatmap_data(tensor_4d, aggregate_channels=False)
    print(f"✅ 4D张量 {tensor_4d.shape} -> {data_separate.shape} (分离)")
    
    # 测试3D张量
    tensor_3d = torch.randn(3, 64, 64)
    data = tensor_to_heatmap_data(tensor_3d, aggregate_channels=True)
    print(f"✅ 3D张量 {tensor_3d.shape} -> {data.shape}")
    print()


def test_colormap_options():
    """测试不同的colormap选项"""
    print("=" * 60)
    print("测试5: 不同Colormap")
    print("=" * 60)
    
    test_data = np.random.randn(128, 128) * 0.5
    output_dir = 'test_heatmap_output/colormaps'
    os.makedirs(output_dir, exist_ok=True)
    
    cmaps = ['coolwarm', 'seismic', 'RdBu', 'hot', 'viridis', 'plasma']
    
    for cmap in cmaps:
        save_heatmap(
            test_data,
            os.path.join(output_dir, f'test_{cmap}.png'),
            title=f'Colormap: {cmap}',
            cmap=cmap
        )
        print(f"✅ 保存了 {cmap} colormap")
    
    print(f"\n所有colormap测试图已保存到: {output_dir}")
    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" Diff热力图可视化功能测试")
    print("=" * 60 + "\n")
    
    try:
        test_basic_heatmap()
        test_diff_visualization()
        test_uncertainty_visualization()
        test_tensor_conversion()
        test_colormap_options()
        
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n查看生成的测试图像:")
        print("  - test_heatmap_output/test_basic_heatmap.png")
        print("  - test_heatmap_output/test_diff_*.png")
        print("  - test_heatmap_output/test_uncertainty_*.png")
        print("  - test_heatmap_output/colormaps/test_*.png")
        print()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
