"""
快速测试脚本：验证带语义引导的ContentAwareSpatialUncertaintyMapping
测试CLIP/DeiT编码器是否正常工作
"""

import torch
import torch.nn.functional as F
import sys
import os

# 添加路径
sys.path.insert(0, '/root/autodl-tmp/test/AURA')

from basicsr.models.uncertainty_mapping import ContentAwareSpatialUncertaintyMapping

def test_semantic_uncertainty_mapping():
    """测试语义引导的不确定度映射"""
    
    print("="*70)
    print("测试带语义引导的不确定度映射模块")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 测试配置
    batch_size = 2
    hr_size = 256  # HR图像尺寸
    
    # 创建模拟数据
    print(f"\n创建测试数据: batch_size={batch_size}, hr_size={hr_size}")
    diff = torch.randn(batch_size, 3, hr_size, hr_size).to(device) * 0.5  # 模拟diff
    lq = torch.rand(batch_size, 3, hr_size, hr_size).to(device)  # 模拟LR图像（已上采样）[0,1]
    
    print(f"  Diff shape: {diff.shape}, range: [{diff.min():.3f}, {diff.max():.3f}]")
    print(f"  LQ shape: {lq.shape}, range: [{lq.min():.3f}, {lq.max():.3f}]")
    
    # 测试1: 使用CLIP
    print("\n" + "="*70)
    print("测试1: 使用CLIP语义编码器")
    print("="*70)
    
    try:
        mapper_clip = ContentAwareSpatialUncertaintyMapping(
            un_max=1.0,
            min_noise=0.2,
            channels=3,
            hidden_channels=64,
            num_heads=4,
            window_size=8,
            use_pretrained_semantic=True,
            semantic_model='clip'
        ).to(device)
        
        mapper_clip.eval()
        
        print("\n✓ CLIP mapper创建成功")
        print(f"  - Semantic encoder: {type(mapper_clip.semantic_encoder).__name__}")
        
        with torch.no_grad():
            uncertainty_clip = mapper_clip(diff, lq)
        
        print(f"\n✓ 前向传播成功")
        print(f"  - Uncertainty shape: {uncertainty_clip.shape}")
        print(f"  - Uncertainty range: [{uncertainty_clip.min():.3f}, {uncertainty_clip.max():.3f}]")
        print(f"  - Uncertainty mean: {uncertainty_clip.mean():.3f}")
        print(f"  - Uncertainty std: {uncertainty_clip.std():.3f}")
        
        # 检查是否在合理范围
        assert uncertainty_clip.min() >= 0.19, "Uncertainty最小值应该接近min_noise=0.2"
        assert uncertainty_clip.max() <= 1.01, "Uncertainty最大值应该接近1.0"
        
        print("\n✓ CLIP测试通过!")
        
    except ImportError as e:
        print(f"\n✗ CLIP不可用: {e}")
        print("  请安装: pip install transformers")
    except Exception as e:
        print(f"\n✗ CLIP测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 使用DeiT
    print("\n" + "="*70)
    print("测试2: 使用DeiT语义编码器")
    print("="*70)
    
    try:
        mapper_deit = ContentAwareSpatialUncertaintyMapping(
            un_max=1.0,
            min_noise=0.2,
            channels=3,
            hidden_channels=64,
            num_heads=4,
            window_size=8,
            use_pretrained_semantic=True,
            semantic_model='deit'
        ).to(device)
        
        mapper_deit.eval()
        
        print("\n✓ DeiT mapper创建成功")
        print(f"  - Semantic encoder: {type(mapper_deit.semantic_encoder).__name__}")
        
        with torch.no_grad():
            uncertainty_deit = mapper_deit(diff, lq)
        
        print(f"\n✓ 前向传播成功")
        print(f"  - Uncertainty shape: {uncertainty_deit.shape}")
        print(f"  - Uncertainty range: [{uncertainty_deit.min():.3f}, {uncertainty_deit.max():.3f}]")
        print(f"  - Uncertainty mean: {uncertainty_deit.mean():.3f}")
        print(f"  - Uncertainty std: {uncertainty_deit.std():.3f}")
        
        print("\n✓ DeiT测试通过!")
        
    except ImportError as e:
        print(f"\n✗ DeiT不可用: {e}")
        print("  请安装: pip install timm")
    except Exception as e:
        print(f"\n✗ DeiT测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试3: 不使用预训练（降级方案）
    print("\n" + "="*70)
    print("测试3: 不使用预训练编码器（降级方案）")
    print("="*70)
    
    try:
        mapper_no_pretrain = ContentAwareSpatialUncertaintyMapping(
            un_max=1.0,
            min_noise=0.2,
            channels=3,
            hidden_channels=64,
            num_heads=4,
            window_size=8,
            use_pretrained_semantic=False
        ).to(device)
        
        mapper_no_pretrain.eval()
        
        print("\n✓ 无预训练mapper创建成功")
        
        with torch.no_grad():
            uncertainty_no_pretrain = mapper_no_pretrain(diff, lq)
        
        print(f"\n✓ 前向传播成功")
        print(f"  - Uncertainty shape: {uncertainty_no_pretrain.shape}")
        print(f"  - Uncertainty range: [{uncertainty_no_pretrain.min():.3f}, {uncertainty_no_pretrain.max():.3f}]")
        
        print("\n✓ 降级方案测试通过!")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试4: 梯度反向传播
    print("\n" + "="*70)
    print("测试4: 梯度反向传播")
    print("="*70)
    
    try:
        mapper_train = ContentAwareSpatialUncertaintyMapping(
            un_max=1.0,
            min_noise=0.2,
            channels=3,
            hidden_channels=32,  # 减小维度加快测试
            num_heads=4,
            window_size=8,
            use_pretrained_semantic=True,
            semantic_model='clip'
        ).to(device)
        
        mapper_train.train()
        
        # 前向传播
        uncertainty = mapper_train(diff, lq)
        
        # 计算损失
        target = torch.ones_like(uncertainty) * 0.5
        loss = F.mse_loss(uncertainty, target)
        
        print(f"\n✓ 损失计算成功: loss={loss.item():.4f}")
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        has_grad = False
        for name, param in mapper_train.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                print(f"  - {name}: grad_norm={param.grad.norm().item():.4f}")
        
        # 检查CLIP参数是否冻结
        if hasattr(mapper_train, 'semantic_encoder'):
            for param in mapper_train.semantic_encoder.parameters():
                assert param.grad is None or param.grad.abs().sum() == 0, \
                    "CLIP参数应该被冻结!"
            print("\n✓ CLIP编码器正确冻结（无梯度）")
        
        assert has_grad, "应该有可训练参数产生梯度"
        print("\n✓ 梯度反向传播测试通过!")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print("""
如果所有测试通过，说明：
1. ✓ 语义编码器（CLIP/DeiT）加载成功
2. ✓ 前向传播正常工作
3. ✓ Uncertainty输出在合理范围
4. ✓ 梯度反向传播正常
5. ✓ CLIP参数正确冻结

可以开始训练了！

训练命令：
    cd /root/project/AURA
    python basicsr/train.py -opt configs/train_with_semantic_guidance.yml
    """)

if __name__ == "__main__":
    test_semantic_uncertainty_mapping()
