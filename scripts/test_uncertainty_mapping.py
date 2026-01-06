"""
测试脚本：验证可学习不确定度映射的实现
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.models.uncertainty_mapping import LearnableUncertaintyMapping, SpatialUncertaintyMapping


def test_learnable_uncertainty_mapping():
    """测试MLP-based可学习映射"""
    print("="*60)
    print("Testing LearnableUncertaintyMapping...")
    print("="*60)
    
    # 参数
    un_max = 1.0
    min_noise = 0.05
    batch_size = 2
    channels = 3
    height = 64
    width = 64
    
    # 创建模型
    mapper = LearnableUncertaintyMapping(
        un_max=un_max,
        min_noise=min_noise,
        hidden_dim=64,
        num_layers=3,
        use_residual=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in mapper.parameters())}")
    
    # 创建测试数据
    diff = torch.randn(batch_size, channels, height, width) * 0.5
    
    # 前向传播
    print("\nTesting forward pass...")
    with torch.no_grad():
        # 固定映射
        fixed_weight = mapper.fixed_mapping(diff)
        print(f"Fixed mapping - mean: {fixed_weight.mean():.4f}, std: {fixed_weight.std():.4f}")
        print(f"Fixed mapping - min: {fixed_weight.min():.4f}, max: {fixed_weight.max():.4f}")
        
        # 可学习映射（初始化后应该接近固定映射）
        learned_weight = mapper(diff)
        print(f"Learned mapping - mean: {learned_weight.mean():.4f}, std: {learned_weight.std():.4f}")
        print(f"Learned mapping - min: {learned_weight.min():.4f}, max: {learned_weight.max():.4f}")
        
        # 差异
        diff_weight = (learned_weight - fixed_weight).abs()
        print(f"Difference - mean: {diff_weight.mean():.6f}, max: {diff_weight.max():.6f}")
    
    # 测试梯度
    print("\nTesting gradient flow...")
    mapper.train()
    diff.requires_grad = True
    
    learned_weight = mapper(diff)
    loss = learned_weight.mean()
    loss.backward()
    
    has_grad = sum(1 for p in mapper.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"Layers with gradients: {has_grad}/{len(list(mapper.parameters()))}")
    
    # 测试统计信息
    print("\nTesting statistics...")
    with torch.no_grad():
        stats = mapper.get_statistics(diff)
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}")
    
    print("\n✓ LearnableUncertaintyMapping test passed!")
    return True


def test_spatial_uncertainty_mapping():
    """测试Spatial-aware可学习映射"""
    print("\n" + "="*60)
    print("Testing SpatialUncertaintyMapping...")
    print("="*60)
    
    # 参数
    un_max = 1.0
    min_noise = 0.05
    batch_size = 2
    channels = 3
    height = 64
    width = 64
    
    # 创建模型
    mapper = SpatialUncertaintyMapping(
        un_max=un_max,
        min_noise=min_noise,
        channels=channels,
        hidden_channels=32
    )
    
    print(f"Model parameters: {sum(p.numel() for p in mapper.parameters())}")
    
    # 创建测试数据
    diff = torch.randn(batch_size, channels, height, width) * 0.5
    
    # 前向传播
    print("\nTesting forward pass...")
    with torch.no_grad():
        # 固定映射
        fixed_weight = mapper.fixed_mapping(diff)
        print(f"Fixed mapping - mean: {fixed_weight.mean():.4f}, std: {fixed_weight.std():.4f}")
        
        # 可学习映射
        learned_weight = mapper(diff)
        print(f"Learned mapping - mean: {learned_weight.mean():.4f}, std: {learned_weight.std():.4f}")
        
        # 差异
        diff_weight = (learned_weight - fixed_weight).abs()
        print(f"Difference - mean: {diff_weight.mean():.6f}, max: {diff_weight.max():.6f}")
    
    # 测试梯度
    print("\nTesting gradient flow...")
    mapper.train()
    diff.requires_grad = True
    
    learned_weight = mapper(diff)
    loss = learned_weight.mean()
    loss.backward()
    
    has_grad = sum(1 for p in mapper.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"Layers with gradients: {has_grad}/{len(list(mapper.parameters()))}")
    
    print("\n✓ SpatialUncertaintyMapping test passed!")
    return True


def test_initialization():
    """测试初始化是否接近恒等映射"""
    print("\n" + "="*60)
    print("Testing Initialization (Identity Mapping)...")
    print("="*60)
    
    mapper = LearnableUncertaintyMapping(
        un_max=1.0,
        min_noise=0.05,
        hidden_dim=64,
        num_layers=3,
        use_residual=True
    )
    
    # 创建不同范围的测试数据
    test_cases = [
        ("Small diff", torch.randn(1, 3, 32, 32) * 0.1),
        ("Medium diff", torch.randn(1, 3, 32, 32) * 0.5),
        ("Large diff", torch.randn(1, 3, 32, 32) * 1.0),
    ]
    
    with torch.no_grad():
        for name, diff in test_cases:
            fixed = mapper.fixed_mapping(diff)
            learned = mapper(diff)
            error = (learned - fixed).abs().mean()
            print(f"{name:15s} - Initialization error: {error:.8f}")
            
            assert error < 0.01, f"Initialization error too large: {error}"
    
    print("\n✓ Initialization test passed!")
    return True


def test_convergence():
    """测试简单优化是否收敛"""
    print("\n" + "="*60)
    print("Testing Convergence...")
    print("="*60)
    
    mapper = LearnableUncertaintyMapping(
        un_max=1.0,
        min_noise=0.05,
        hidden_dim=32,
        num_layers=2,
        use_residual=True
    )
    
    # 优化器
    optimizer = torch.optim.Adam(mapper.parameters(), lr=1e-3)
    
    # 创建简单的监督目标：让映射输出更大的权重
    diff = torch.randn(4, 3, 16, 16) * 0.5
    target_weight = torch.ones_like(diff) * 0.8  # 目标权重
    
    # 训练几步
    losses = []
    for step in range(100):
        optimizer.zero_grad()
        pred_weight = mapper(diff)
        loss = nn.MSELoss()(pred_weight, target_weight)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Loss = {loss.item():.6f}")
    
    # 检查是否收敛
    assert losses[-1] < losses[0] * 0.5, "Model did not converge"
    print(f"\nInitial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")
    print(f"Loss reduced by {(1 - losses[-1]/losses[0])*100:.1f}%")
    
    print("\n✓ Convergence test passed!")
    return True


def test_value_range():
    """测试输出值范围是否符合预期"""
    print("\n" + "="*60)
    print("Testing Value Range...")
    print("="*60)
    
    un_max = 1.0
    min_noise = 0.05
    
    mapper = LearnableUncertaintyMapping(
        un_max=un_max,
        min_noise=min_noise,
        hidden_dim=64,
        num_layers=3,
        use_residual=True
    )
    
    # 测试极端值
    test_cases = [
        ("Zero diff", torch.zeros(2, 3, 32, 32)),
        ("Small diff", torch.ones(2, 3, 32, 32) * 0.1),
        ("Large diff", torch.ones(2, 3, 32, 32) * 10.0),
        ("Negative diff", torch.ones(2, 3, 32, 32) * -1.0),
    ]
    
    with torch.no_grad():
        for name, diff in test_cases:
            weight = mapper(diff)
            print(f"{name:15s} - min: {weight.min():.4f}, max: {weight.max():.4f}, "
                  f"mean: {weight.mean():.4f}")
            
            # 检查范围
            assert weight.min() >= min_noise - 1e-6, f"Weight below min_noise: {weight.min()}"
            assert weight.max() <= 1.0 + 1e-6, f"Weight above 1.0: {weight.max()}"
    
    print("\n✓ Value range test passed!")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Starting Uncertainty Mapping Tests")
    print("="*60 + "\n")
    
    tests = [
        test_learnable_uncertainty_mapping,
        test_spatial_uncertainty_mapping,
        test_initialization,
        test_value_range,
        test_convergence,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, True, None))
        except Exception as e:
            results.append((test_func.__name__, False, str(e)))
            print(f"\n✗ {test_func.__name__} failed: {e}")
    
    # 总结
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed, error in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:40s} {status}")
        if error:
            print(f"  Error: {error}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed, _ in results if passed)
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
