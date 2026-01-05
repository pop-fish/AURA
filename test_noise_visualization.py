"""
测试脚本：验证噪声可视化功能

此脚本用于测试模型在推理时是否能正确输出：
1. 内容感知映射得到的不确定度图（保存到 results/[dataset_name]/noise1/）
2. 扩散过程的加噪起点图（保存到 results/[dataset_name]/noise2/）

使用方法：
    python test_noise_visualization.py -opt options/test_upsr_real.yml

确保你的配置文件中：
1. 启用了 save_img: true
2. 配置了正确的模型权重路径
3. 配置了测试数据集路径
"""

import os
import sys
from os import path as osp

# 设置项目根目录（UPSR_view 目录）
root_path = osp.abspath(osp.dirname(__file__))
sys.path.append(root_path)

import torch
from basicsr.models import build_model
from basicsr.utils.options import dict2str, parse_options
from basicsr.data import build_dataloader, build_dataset

def main():
    # 解析配置文件（parse_options 内部会解析命令行参数）
    opt, _ = parse_options(root_path, is_train=False)
    
    print("=" * 80)
    print("噪声可视化测试")
    print("=" * 80)
    
    # 检查配置
    print("\n[配置检查]")
    model_type = opt.get('model_type') or opt.get('model_target', 'N/A')
    print(f"模型类型: {model_type}")
    
    network_type = 'N/A'
    if 'network_g' in opt:
        network_type = opt['network_g'].get('type') or opt['network_g'].get('target', 'N/A')
    print(f"网络架构: {network_type}")
    
    save_img = opt.get('val', {}).get('save_img', False) if 'val' in opt else True
    print(f"保存图片: {save_img}")
    print(f"可视化路径: {opt['path'].get('visualization', 'N/A')}")
    
    if opt.get('uncertainty_mapping'):
        print(f"\n不确定度映射配置:")
        print(f"  - 使用可学习映射: {opt['uncertainty_mapping'].get('use_learnable', False)}")
        if opt['uncertainty_mapping'].get('use_learnable', False):
            print(f"  - 映射类型: {opt['uncertainty_mapping'].get('type', 'mlp')}")
            print(f"  - 权重路径: {opt['uncertainty_mapping'].get('ckpt_path', 'N/A')}")
    
    # 构建数据集和数据加载器
    print("\n[构建数据集]")
    val_loaders = []
    for phase, dataset_opt in opt['datasets'].items():
        if phase.startswith('val'):
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed']
            )
            print(f"验证集 '{dataset_opt['name']}': {len(val_set)} 张图片")
            val_loaders.append(val_loader)
    
    if not val_loaders:
        print("错误：没有找到验证数据集！")
        return
    
    # 构建模型
    print("\n[构建模型]")
    model = build_model(opt)
    print("模型构建完成")
    
    # 运行验证
    print("\n[开始验证]")
    print("-" * 80)
    
    for val_loader in val_loaders:
        dataset_name = val_loader.dataset.opt['name']
        print(f"\n处理数据集: {dataset_name}")
        
        # 创建输出目录
        vis_path = opt['path']['visualization']
        noise1_path = os.path.join(vis_path, dataset_name, 'noise1')
        noise2_path = os.path.join(vis_path, dataset_name, 'noise2')
        
        print(f"输出路径:")
        print(f"  - SR结果: {os.path.join(vis_path, dataset_name)}")
        print(f"  - 不确定度图: {noise1_path}")
        print(f"  - 加噪起点: {noise2_path}")
        
        # 运行验证
        save_img_flag = opt.get('val', {}).get('save_img', True) if 'val' in opt else True
        model.validation(
            val_loader,
            current_iter=opt.get('name', 'test'),
            tb_logger=None,
            save_img=save_img_flag
        )
    
    print("\n" + "=" * 80)
    print("验证完成！")
    print("=" * 80)
    
    # 检查输出文件
    print("\n[检查输出文件]")
    for val_loader in val_loaders:
        dataset_name = val_loader.dataset.opt['name']
        vis_path = opt['path']['visualization']
        
        sr_path = os.path.join(vis_path, dataset_name)
        noise1_path = os.path.join(vis_path, dataset_name, 'noise1')
        noise2_path = os.path.join(vis_path, dataset_name, 'noise2')
        
        if os.path.exists(sr_path):
            sr_files = [f for f in os.listdir(sr_path) if f.endswith('.png')]
            print(f"\n数据集 '{dataset_name}':")
            print(f"  - SR图片数量: {len(sr_files)}")
            
            if os.path.exists(noise1_path):
                noise1_files = [f for f in os.listdir(noise1_path) if f.endswith('.png')]
                print(f"  - 不确定度图数量: {len(noise1_files)}")
            else:
                print(f"  - 不确定度图: 未生成（可能未启用可学习映射）")
            
            if os.path.exists(noise2_path):
                noise2_files = [f for f in os.listdir(noise2_path) if f.endswith('.png')]
                print(f"  - 加噪起点图数量: {len(noise2_files)}")
            else:
                print(f"  - 加噪起点图: 未生成")
    
    print("\n测试完成！")

if __name__ == '__main__':
    main()
