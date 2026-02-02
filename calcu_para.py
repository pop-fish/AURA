"""
统计UPSR模型（含可学习不确定度映射）的参数数量
"""
import torch
import yaml
from collections import OrderedDict
import sys
import os

# 添加路径以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basicsr.archs import build_network
from basicsr.models.uncertainty_mapping import (
    LearnableUncertaintyMapping,
    SpatialUncertaintyMapping,
    ContentAwareSpatialUncertaintyMapping
)


def count_parameters(model):
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        total_params: 总参数数量
        trainable_params: 可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num):
    """格式化数字为易读形式"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_uncertainty_mapper(config):
    """根据配置创建不确定度映射器"""
    mapper_config = config.get('uncertainty_mapping', {})
    
    if not mapper_config.get('use_learnable', False):
        return None, "固定映射（无额外参数）"
    
    mapper_type = mapper_config.get('type', 'mlp')
    un_max = config['diffusion'].get('un', 0.1)
    min_noise = config['diffusion'].get('min_noise', 0.4)
    
    if mapper_type == 'mlp':
        mapper = LearnableUncertaintyMapping(
            un_max=un_max,
            min_noise=min_noise,
            hidden_dim=mapper_config.get('hidden_dim', 64),
            num_layers=mapper_config.get('num_layers', 3),
            use_residual=mapper_config.get('use_residual', True)
        )
        mapper_desc = "MLP映射器（逐像素）"
    elif mapper_type == 'spatial':
        mapper = SpatialUncertaintyMapping(
            un_max=un_max,
            min_noise=min_noise,
            channels=3,
            hidden_channels=mapper_config.get('hidden_channels', 32)
        )
        mapper_desc = "空间卷积映射器"
    elif mapper_type == 'content_aware':
        mapper = ContentAwareSpatialUncertaintyMapping(
            un_max=un_max,
            min_noise=min_noise,
            channels=3,
            hidden_channels=mapper_config.get('hidden_channels', 64),
            num_heads=mapper_config.get('num_heads', 4),
            window_size=mapper_config.get('window_size', 8),
            use_pretrained_semantic=mapper_config.get('use_pretrained_semantic', True),
            semantic_model=mapper_config.get('semantic_model', 'clip')
        )
        mapper_desc = "内容感知映射器（Cross-Attention）"
    else:
        return None, f"未知类型: {mapper_type}"
    
    return mapper, mapper_desc


def main():
    # 配置文件路径（可以修改为其他配置）
    config_options = [
        ('options/train_with_semantic_perceptual_autodl.yml', '当前训练模型(语义感知)'),
        # ('options/stage1_learnable_mapper_only.yml', '阶段1：仅训练映射层'),
        # ('options/stage2_joint_training.yml', '阶段2：联合训练'),
        # ('options/content_aware_uncertainty_mapping.yml', '内容感知映射'),
    ]
    
    print("="*80)
    print("UPSR模型参数统计（含可学习不确定度映射）")
    print("="*80)
    
    for config_path, config_desc in config_options:
        full_path = config_path
        if not os.path.exists(full_path):
            print(f"\n跳过 {config_desc}: 配置文件不存在 ({full_path})")
            continue
        
        print(f"\n{'='*80}")
        print(f"配置: {config_desc}")
        print(f"文件: {config_path}")
        print(f"{'='*80}")
        
        # 加载配置
        config = load_config(full_path)

        # 获取冻结配置标志
        mapper_config = config.get('uncertainty_mapping', {})
        # 优先读取 freeze_main_network，如果没有则读取 freeze_others_initially
        freeze_g = mapper_config.get('freeze_main_network', False) or mapper_config.get('freeze_others_initially', False)
        
        # 1. 统计Network G (主生成网络)
        print(f"\n{'-'*80}")
        if freeze_g:
            print("1. Network G (扩散模型主网络) [检测到配置要求冻结]")
        else:
            print("1. Network G (扩散模型主网络)")
        print(f"{'-'*80}")
        
        net_g_opt = config['network_g']
        net_g = build_network(net_g_opt)

        if freeze_g:
            for param in net_g.parameters():
                param.requires_grad = False
        
        total_g, trainable_g = count_parameters(net_g)
        print(f"总参数数量: {total_g:,} ({format_number(total_g)})")
        print(f"可训练参数: {trainable_g:,} ({format_number(trainable_g)})")
        
        # 2. 统计Network MSE (辅助SR网络)
        print(f"\n{'-'*80}")
        print("2. Network MSE (辅助SR网络 - 冻结)")
        print(f"{'-'*80}")
        
        net_mse_opt = config['network_mse']
        net_mse = build_network(net_mse_opt)
        
        total_mse, trainable_mse = count_parameters(net_mse)
        print(f"总参数数量: {total_mse:,} ({format_number(total_mse)})")
        print(f"注意: 该网络在训练时被冻结")
        
        # 3. 统计不确定度映射器
        print(f"\n{'-'*80}")
        print("3. 不确定度映射器")
        print(f"{'-'*80}")
        
        mapper, mapper_desc = create_uncertainty_mapper(config)
        
        semantic_total = 0
        semantic_trainable = 0
        
        if mapper is None:
            print(f"类型: {mapper_desc}")
            print(f"参数数量: 0 (使用固定公式)")
            total_mapper = 0
            trainable_mapper = 0
        else:
            print(f"类型: {mapper_desc}")
            
            # --- 分离语义编码器统计 ---
            if hasattr(mapper, 'semantic_encoder'):
                print(f"检测到语义编码器: {type(mapper.semantic_encoder).__name__}")
                semantic_total, semantic_trainable = count_parameters(mapper.semantic_encoder)
                print(f"  - 语义编码器参数: {semantic_total:,} ({format_number(semantic_total)})")
                print(f"  - 语义编码器可训练: {semantic_trainable:,} ({format_number(semantic_trainable)})")
                
                # 临时将语义编码器排除出总统计，以便分别展示
                # 注意：mapper.parameters() 仍然包含它，所以我们需要做减法或者更智能的统计
            
            total_mapper_all, trainable_mapper_all = count_parameters(mapper)
            
            # 纯映射器部分（不含语义编码器）
            mapper_base_total = total_mapper_all - semantic_total
            mapper_base_trainable = trainable_mapper_all - semantic_trainable
            
            # 为了兼容之前的逻辑变量名
            total_mapper = total_mapper_all
            trainable_mapper = trainable_mapper_all
            
            print(f"总参数数量 (含语义编码器): {total_mapper:,} ({format_number(total_mapper)})")
            print(f"可训练参数: {trainable_mapper:,} ({format_number(trainable_mapper)})")
            
            if semantic_total > 0:
                print(f"映射器本体参数 (扣除语义编码器): {mapper_base_total:,} ({format_number(mapper_base_total)})")
            
            # 打印映射器配置
            mapper_config = config.get('uncertainty_mapping', {})
            print(f"\n映射器配置:")
            if mapper_config.get('type') == 'mlp':
                print(f"  - 隐藏维度: {mapper_config.get('hidden_dim', 64)}")
                print(f"  - 层数: {mapper_config.get('num_layers', 3)}")
                print(f"  - 使用残差: {mapper_config.get('use_residual', True)}")
            elif mapper_config.get('type') == 'spatial':
                print(f"  - 隐藏通道数: {mapper_config.get('hidden_channels', 32)}")
            elif mapper_config.get('type') == 'content_aware':
                print(f"  - 隐藏通道数: {mapper_config.get('hidden_channels', 64)}")
                print(f"  - 注意力头数: {mapper_config.get('num_heads', 4)}")
                print(f"  - 窗口大小: {mapper_config.get('window_size', 8)}")
                if hasattr(mapper, 'semantic_encoder'):
                    print(f"  - 语义模型: {mapper_config.get('semantic_model', 'clip')} (参数已统计)")
            
            # 详细参数分布
            print(f"\n映射器详细参数分布:")
            param_dict = {}
            for name, param in mapper.named_parameters():
                if param.requires_grad:
                    module_name = name.split('.')[0] if '.' in name else name
                    if module_name not in param_dict:
                        param_dict[module_name] = 0
                    param_dict[module_name] += param.numel()
            
            sorted_params = sorted(param_dict.items(), key=lambda x: x[1], reverse=True)
            for module_name, num_params in sorted_params:
                percentage = (num_params / trainable_mapper) * 100 if trainable_mapper > 0 else 0
                print(f"  {module_name:30s}: {num_params:10,} ({format_number(num_params):>8s}) - {percentage:5.2f}%")
        
        # 4. 总结
        print(f"\n{'='*80}")
        print("总结")
        print(f"{'='*80}")
        
        # 判断训练模式
        # mapper_config = config.get('uncertainty_mapping', {}) # already loaded
        
        # 简化逻辑：直接使用实际统计到的可训练参数
        active_trainable = trainable_g + trainable_mapper
        
        if trainable_g == 0 and trainable_mapper > 0:
             training_mode = "仅训练映射器 (Network G 已冻结)"
        elif trainable_g > 0 and trainable_mapper > 0:
             training_mode = "联合训练 (Network G 参与训练)"
        elif trainable_g > 0 and trainable_mapper == 0:
             training_mode = "仅训练主网络 (无映射器或映射器冻结)"
        else:
             training_mode = "全冻结 / 评估模式"
        
        print(f"\n训练模式: {training_mode}")
        print(f"\n模型总参数:")
        print(f"  Network G:        {total_g:12,} ({format_number(total_g):>8s})")
        print(f"  Network MSE:      {total_mse:12,} ({format_number(total_mse):>8s}) [冻结]")
        print(f"  映射器:           {total_mapper:12,} ({format_number(total_mapper):>8s})")
        print(f"  {'-'*50}")
        print(f"  总计:             {total_g + total_mse + total_mapper:12,} ({format_number(total_g + total_mse + total_mapper):>8s})")
        
        print(f"\n可训练参数:")
        print(f"  Network G:        {trainable_g:12,} ({format_number(trainable_g):>8s})")
        print(f"  映射器:           {trainable_mapper:12,} ({format_number(trainable_mapper):>8s})")
        print(f"  {'-'*50}")
        print(f"  实际训练:         {active_trainable:12,} ({format_number(active_trainable):>8s})")
        
        # 计算相对增加量
        if total_g > 0:
            mapper_percentage = (total_mapper / total_g) * 100
            print(f"\n映射器参数占主网络比例: {mapper_percentage:.3f}%")
            print(f"参数增加量: +{format_number(total_mapper)} (相对于基础UPSR)")
    
    print(f"\n{'='*80}")
    print("统计完成！")
    print(f"{'='*80}")
    
    # 保存详细统计到文件
    output_file = 'model_parameters_with_mapper_statistics.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("UPSR模型参数统计（含可学习不确定度映射）\n")
        f.write("="*80 + "\n\n")
        
        for config_path, config_desc in config_options:
            if not os.path.exists(config_path):
                continue
            
            f.write(f"\n{'='*80}\n")
            f.write(f"配置: {config_desc}\n")
            f.write(f"{'='*80}\n\n")
            
            config = load_config(config_path)
            
            # 获取冻结配置标志
            mapper_config = config.get('uncertainty_mapping', {})
            freeze_g = mapper_config.get('freeze_main_network', False) or mapper_config.get('freeze_others_initially', False)

            # Network G
            net_g_opt = config['network_g']
            net_g = build_network(net_g_opt)
            
            if freeze_g:
                for param in net_g.parameters():
                    param.requires_grad = False

            total_g, trainable_g = count_parameters(net_g)
            
            # Network MSE
            net_mse_opt = config['network_mse']
            net_mse = build_network(net_mse_opt)
            total_mse, _ = count_parameters(net_mse)
            
            # Mapper
            mapper, mapper_desc = create_uncertainty_mapper(config)
            semantic_total = 0
            if mapper:
                if hasattr(mapper, 'semantic_encoder'):
                    semantic_total, _ = count_parameters(mapper.semantic_encoder)
                total_mapper, trainable_mapper = count_parameters(mapper)
            else:
                total_mapper, trainable_mapper = 0, 0
            
            mapper_base_total = total_mapper - semantic_total
            
            active_trainable = trainable_g + trainable_mapper

            f.write(f"Network G:       Total: {total_g:12,} ({format_number(total_g):>8s}) | Trainable: {trainable_g:12,} ({format_number(trainable_g):>8s})\n")
            f.write(f"Network MSE:     Total: {total_mse:12,} ({format_number(total_mse):>8s}) | Trainable:            0 (       0)\n")
            f.write(f"映射器:          Total: {total_mapper:12,} ({format_number(total_mapper):>8s}) | Trainable: {trainable_mapper:12,} ({format_number(trainable_mapper):>8s})\n")
            if semantic_total > 0:
                 f.write(f"  - 语义编码器:  Total: {semantic_total:12,} ({format_number(semantic_total):>8s}) [冻结]\n")
                 f.write(f"  - 映射器本体:  Total: {mapper_base_total:12,} ({format_number(mapper_base_total):>8s})\n")
            
            f.write(f"{'-'*100}\n")
            f.write(f"实际参与训练参数: {active_trainable:12,} ({format_number(active_trainable):>8s})\n\n")
            
            if total_g > 0:
                mapper_percentage = (total_mapper / total_g) * 100
                f.write(f"映射器占主网络比例: {mapper_percentage:.3f}%\n")
    
    print(f"\n详细统计结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
