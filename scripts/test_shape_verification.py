"""
验证不确定度映射层的形状是否正确
"""

import torch
import sys
sys.path.insert(0, '/root/project/UPSR_copy')

from basicsr.models.uncertainty_mapping import LearnableUncertaintyMapping, SpatialUncertaintyMapping


def test_learnable_mapping():
    """测试MLP映射层"""
    print("=" * 60)
    print("测试 LearnableUncertaintyMapping")
    print("=" * 60)
    
    # 创建映射层
    mapper = LearnableUncertaintyMapping(
        un_max=1.0,
        min_noise=0.05,
        hidden_dim=64,
        num_layers=3,
        use_residual=True
    )
    
    # 测试不同形状的输入
    test_shapes = [
        (1, 3, 64, 64),    # 单张RGB图像
        (4, 3, 128, 128),  # 批量RGB图像
        (2, 1, 256, 256),  # 单通道图像
    ]
    
    for shape in test_shapes:
        print(f"\n测试输入形状: {shape}")
        
        # 创建模拟的diff
        diff = torch.randn(shape) * 0.5  # 范围大约在 [-1.5, 1.5]
        
        # 通过固定映射
        fixed_output = mapper.fixed_mapping(diff)
        print(f"  固定映射输出形状: {fixed_output.shape}")
        print(f"  固定映射值范围: [{fixed_output.min():.4f}, {fixed_output.max():.4f}]")
        
        # 通过可学习映射
        learnable_output = mapper(diff)
        print(f"  可学习映射输出形状: {learnable_output.shape}")
        print(f"  可学习映射值范围: [{learnable_output.min():.4f}, {learnable_output.max():.4f}]")
        
        # 验证形状一致性
        assert learnable_output.shape == diff.shape, "输出形状与输入不匹配！"
        assert learnable_output.shape == fixed_output.shape, "可学习输出与固定输出形状不匹配！"
        
        # 验证值范围
        assert learnable_output.min() >= mapper.min_noise - 1e-6, "输出值小于最小噪声！"
        assert learnable_output.max() <= 1.0 + 1e-6, "输出值大于1.0！"
        
        print("  ✓ 形状检查通过")
        print("  ✓ 值范围检查通过")


def test_spatial_mapping():
    """测试空间感知映射层"""
    print("\n" + "=" * 60)
    print("测试 SpatialUncertaintyMapping")
    print("=" * 60)
    
    # 创建映射层
    mapper = SpatialUncertaintyMapping(
        un_max=1.0,
        min_noise=0.05,
        channels=3,
        hidden_channels=32
    )
    
    # 测试
    test_shapes = [
        (1, 3, 64, 64),
        (2, 3, 128, 128),
    ]
    
    for shape in test_shapes:
        print(f"\n测试输入形状: {shape}")
        
        diff = torch.randn(shape) * 0.5
        
        fixed_output = mapper.fixed_mapping(diff)
        print(f"  固定映射输出形状: {fixed_output.shape}")
        
        spatial_output = mapper(diff)
        print(f"  空间映射输出形状: {spatial_output.shape}")
        print(f"  空间映射值范围: [{spatial_output.min():.4f}, {spatial_output.max():.4f}]")
        
        assert spatial_output.shape == diff.shape, "输出形状与输入不匹配！"
        assert spatial_output.min() >= mapper.min_noise - 1e-6
        assert spatial_output.max() <= 1.0 + 1e-6
        
        print("  ✓ 形状检查通过")
        print("  ✓ 值范围检查通过")


def test_gradients():
    """测试梯度流动"""
    print("\n" + "=" * 60)
    print("测试梯度流动")
    print("=" * 60)
    
    mapper = LearnableUncertaintyMapping(
        un_max=1.0,
        min_noise=0.05,
        hidden_dim=64,
        num_layers=3,
        use_residual=True
    )
    
    # 创建需要梯度的输入
    diff = torch.randn(2, 3, 32, 32, requires_grad=True)
    
    # 前向传播
    output = mapper(diff)
    
    # 计算损失（简单求和）
    loss = output.sum()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = diff.grad is not None
    print(f"输入梯度存在: {has_grad}")
    
    mlp_has_grad = all(p.grad is not None for p in mapper.mlp.parameters())
    print(f"MLP参数梯度存在: {mlp_has_grad}")
    
    assert has_grad, "输入梯度不存在！"
    assert mlp_has_grad, "MLP参数梯度不存在！"
    
    print("✓ 梯度流动检查通过")


def test_initialization():
    """测试初始化是否接近恒等映射"""
    print("\n" + "=" * 60)
    print("测试初始化策略")
    print("=" * 60)
    
    mapper = LearnableUncertaintyMapping(
        un_max=1.0,
        min_noise=0.05,
        hidden_dim=64,
        num_layers=3,
        use_residual=True
    )
    
    # 创建测试输入
    diff = torch.randn(4, 3, 64, 64)
    
    # 获取固定映射和可学习映射的输出
    with torch.no_grad():
        fixed_output = mapper.fixed_mapping(diff)
        learnable_output = mapper(diff)
    
    # 计算差异
    difference = (learnable_output - fixed_output).abs()
    max_diff = difference.max().item()
    mean_diff = difference.mean().item()
    
    print(f"固定映射 vs 可学习映射:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    
    # 初始化应该使得差异很小
    assert mean_diff < 0.1, f"平均差异过大: {mean_diff}"
    
    print("✓ 初始化检查通过（接近固定映射）")


def main():
    print("\n" + "=" * 60)
    print("不确定度映射层形状验证测试")
    print("=" * 60 + "\n")
    
    try:
        test_learnable_mapping()
        test_spatial_mapping()
        test_gradients()
        test_initialization()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
