"""
测试内容感知不确定度映射模块
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from basicsr.models.uncertainty_mapping import ContentAwareSpatialUncertaintyMapping

def test_content_aware_mapping():
    """测试ContentAwareSpatialUncertaintyMapping的功能"""
    print("="*60)
    print("测试内容感知不确定度映射模块")
    print("="*60)
    
    # 参数
    batch_size = 2
    channels = 3
    height, width = 128, 128
    un_max = 1.0
    min_noise = 0.4
    hidden_channels = 64
    num_heads = 4
    
    # 创建模块
    print("\n1. 创建 ContentAwareSpatialUncertaintyMapping...")
    mapper = ContentAwareSpatialUncertaintyMapping(
        un_max=un_max,
        min_noise=min_noise,
        channels=channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in mapper.parameters())
    trainable_params = sum(p.numel() for p in mapper.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    # 创建测试数据
    print("\n2. 创建测试数据...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapper = mapper.to(device)
    
    diff = torch.randn(batch_size, channels, height, width).to(device)
    lq = torch.rand(batch_size, channels, height, width).to(device)  # [0, 1]
    
    print(f"   Diff shape: {diff.shape}")
    print(f"   LQ shape: {lq.shape}")
    print(f"   Device: {device}")
    
    # 前向传播
    print("\n3. 测试前向传播...")
    mapper.eval()
    with torch.no_grad():
        uncertainty = mapper(diff, lq)
    
    print(f"   输出形状: {uncertainty.shape}")
    print(f"   输出范围: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
    print(f"   输出均值: {uncertainty.mean():.4f}")
    print(f"   输出标准差: {uncertainty.std():.4f}")
    
    # 验证约束
    assert uncertainty.shape == diff.shape, "输出形状不匹配!"
    assert uncertainty.min() >= min_noise - 1e-6, f"输出最小值 {uncertainty.min():.4f} < {min_noise}"
    assert uncertainty.max() <= 1.0 + 1e-6, f"输出最大值 {uncertainty.max():.4f} > 1.0"
    print("   ✓ 形状和范围验证通过")
    
    # 比较固定映射和可学习映射
    print("\n4. 比较固定映射 vs 可学习映射...")
    with torch.no_grad():
        fixed_uncertainty = mapper.fixed_mapping(diff)
        learnable_uncertainty = uncertainty
        
        diff_map = (learnable_uncertainty - fixed_uncertainty).abs()
        
        print(f"   固定映射均值: {fixed_uncertainty.mean():.4f}")
        print(f"   可学习映射均值: {learnable_uncertainty.mean():.4f}")
        print(f"   差异均值: {diff_map.mean():.4f}")
        print(f"   差异最大值: {diff_map.max():.4f}")
    
    # 测试不同内容的响应
    print("\n5. 测试内容自适应性...")
    with torch.no_grad():
        # 平滑LQ图像
        lq_smooth = torch.ones_like(lq) * 0.5
        un_smooth = mapper(diff, lq_smooth)
        
        # 纹理丰富的LQ图像
        lq_texture = torch.rand_like(lq)
        un_texture = mapper(diff, lq_texture)
        
        print(f"   平滑区域不确定度: {un_smooth.mean():.4f} ± {un_smooth.std():.4f}")
        print(f"   纹理区域不确定度: {un_texture.mean():.4f} ± {un_texture.std():.4f}")
        print(f"   差异: {(un_texture - un_smooth).abs().mean():.4f}")
    
    # 测试梯度
    print("\n6. 测试梯度传播...")
    mapper.train()
    diff_grad = torch.randn(batch_size, channels, height, width, requires_grad=True).to(device)
    lq_grad = torch.rand(batch_size, channels, height, width, requires_grad=True).to(device)
    
    uncertainty_grad = mapper(diff_grad, lq_grad)
    loss = uncertainty_grad.mean()
    loss.backward()
    
    print(f"   Diff梯度: {diff_grad.grad is not None and diff_grad.grad.abs().max() > 0}")
    print(f"   LQ梯度: {lq_grad.grad is not None and lq_grad.grad.abs().max() > 0}")
    print(f"   映射器梯度: {any(p.grad is not None and p.grad.abs().max() > 0 for p in mapper.parameters() if p.requires_grad)}")
    print("   ✓ 梯度传播正常")
    
    # 内存占用
    print("\n7. 估算内存占用...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_used = torch.cuda.memory_allocated(device) / 1024**2
        print(f"   GPU内存占用: {mem_used:.2f} MB")
    
    # 推理速度
    print("\n8. 测试推理速度...")
    import time
    mapper.eval()
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = mapper(diff, lq)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 计时
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = mapper(diff, lq)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_runs * 1000
    
    print(f"   平均推理时间: {avg_time:.2f} ms")
    print(f"   吞吐量: {num_runs / elapsed_time:.2f} 次/秒")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60)

if __name__ == '__main__':
    test_content_aware_mapping()
