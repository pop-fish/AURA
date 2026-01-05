"""
测试uncertainty mapper的保存和加载功能
"""
import torch
import sys
import os
sys.path.insert(0, '/root/project/UPSR_copy')

from basicsr.models.uncertainty_mapping import LearnableUncertaintyMapping

def test_save_load():
    """测试保存和加载功能"""
    print("=" * 60)
    print("测试 Uncertainty Mapper 保存/加载功能")
    print("=" * 60)
    
    # 创建映射层
    print("\n1. 创建原始映射层...")
    mapper_original = LearnableUncertaintyMapping(
        un_max=0.1,
        min_noise=0.4,
        hidden_dim=64,
        num_layers=3,
        use_residual=True
    )
    
    # 创建测试输入
    test_input = torch.randn(2, 3, 64, 64)
    
    # 前向传播获取输出
    with torch.no_grad():
        output_original = mapper_original(test_input)
    
    print(f"   - 输入形状: {test_input.shape}")
    print(f"   - 输出形状: {output_original.shape}")
    print(f"   - 输出均值: {output_original.mean().item():.6f}")
    print(f"   - 输出标准差: {output_original.std().item():.6f}")
    
    # 保存模型
    save_path = '/tmp/test_uncertainty_mapper.pth'
    print(f"\n2. 保存模型到 {save_path}...")
    state_dict = mapper_original.state_dict()
    save_dict = {'params': state_dict}
    torch.save(save_dict, save_path)
    print(f"   ✓ 保存成功")
    print(f"   - 参数数量: {len(state_dict)} 个张量")
    for name, param in list(state_dict.items())[:3]:
        print(f"   - {name}: {param.shape}")
    
    # 创建新的映射层
    print("\n3. 创建新的映射层并加载权重...")
    mapper_loaded = LearnableUncertaintyMapping(
        un_max=0.1,
        min_noise=0.4,
        hidden_dim=64,
        num_layers=3,
        use_residual=True
    )
    
    # 加载权重
    load_dict = torch.load(save_path)
    if 'params' in load_dict:
        load_dict = load_dict['params']
    mapper_loaded.load_state_dict(load_dict)
    print(f"   ✓ 加载成功")
    
    # 验证输出一致性
    print("\n4. 验证加载的模型输出...")
    with torch.no_grad():
        output_loaded = mapper_loaded(test_input)
    
    print(f"   - 输出形状: {output_loaded.shape}")
    print(f"   - 输出均值: {output_loaded.mean().item():.6f}")
    print(f"   - 输出标准差: {output_loaded.std().item():.6f}")
    
    # 计算差异
    diff = torch.abs(output_original - output_loaded)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print("\n5. 输出差异统计...")
    print(f"   - 最大差异: {max_diff:.10f}")
    print(f"   - 平均差异: {mean_diff:.10f}")
    
    # 判断测试结果
    print("\n" + "=" * 60)
    if max_diff < 1e-6:
        print("✅ 测试通过！保存和加载功能正常工作")
        print(f"   输出完全一致 (最大差异: {max_diff:.2e})")
    else:
        print("❌ 测试失败！输出不一致")
        print(f"   最大差异: {max_diff:.2e} (应该 < 1e-6)")
    print("=" * 60)
    
    # 清理
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"\n已清理测试文件: {save_path}")
    
    return max_diff < 1e-6

def test_parameter_groups():
    """测试参数组设置（用于差异化学习率）"""
    print("\n" + "=" * 60)
    print("测试参数组配置（差异化学习率）")
    print("=" * 60)
    
    # 创建映射层
    mapper = LearnableUncertaintyMapping(
        un_max=0.1,
        min_noise=0.4,
        hidden_dim=64,
        num_layers=3,
        use_residual=True
    )
    
    # 模拟主网络参数
    main_net_params = [torch.nn.Parameter(torch.randn(10, 10)) for _ in range(5)]
    
    # 创建参数组
    base_lr = 1e-4
    mapper_lr_scale = 2.0
    main_lr_scale = 0.5
    
    param_groups = [
        {'params': main_net_params, 'lr': base_lr * main_lr_scale},
        {'params': mapper.parameters(), 'lr': base_lr * mapper_lr_scale}
    ]
    
    print(f"\n参数组配置:")
    print(f"  - Base LR: {base_lr}")
    print(f"  - 主网络 LR: {base_lr * main_lr_scale} (scale={main_lr_scale})")
    print(f"  - 映射层 LR: {base_lr * mapper_lr_scale} (scale={mapper_lr_scale})")
    
    # 创建优化器
    try:
        optimizer = torch.optim.AdamW(param_groups)
    except AttributeError:
        optimizer = torch.optim.Adam(param_groups)  # fallback
    
    print(f"\n优化器状态:")
    for idx, group in enumerate(optimizer.param_groups):
        num_params = len(group['params'])
        lr = group['lr']
        print(f"  - 组 {idx}: {num_params} 个参数, LR = {lr}")
    
    print("\n✅ 参数组配置测试通过")
    print("=" * 60)

if __name__ == '__main__':
    # 运行测试
    success = test_save_load()
    test_parameter_groups()
    
    # 退出代码
    sys.exit(0 if success else 1)
