"""
训练脚本示例：使用可学习不确定度映射的UPSR模型
支持分阶段训练策略
"""

import os
import sys
import argparse
import torch
import logging
from collections import OrderedDict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def main():
    # 解析配置
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--auto_resume', action='store_true')
    
    args = parser.parse_args()
    opt, _ = parse_options(args.opt, is_train=True)
    
    # 创建目录
    make_exp_dirs(opt)
    
    # 初始化logger
    log_file = os.path.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))
    
    # 构建数据集
    logger.info('Creating training dataset...')
    train_set = build_dataset(opt['datasets']['train'])
    train_loader = build_dataloader(
        train_set,
        opt['datasets']['train'],
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed']
    )
    logger.info(f'Number of training images: {len(train_set)}, iters per epoch: {len(train_loader)}')
    
    logger.info('Creating validation dataset...')
    val_set = build_dataset(opt['datasets']['val'])
    val_loader = build_dataloader(
        val_set,
        opt['datasets']['val'],
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed']
    )
    logger.info(f'Number of validation images: {len(val_set)}')
    
    # 构建模型
    logger.info('Building model...')
    model = build_model(opt)
    
    # 检查是否使用分阶段训练
    uncertainty_mapping_opt = opt.get('uncertainty_mapping', {})
    use_staged_training = (
        uncertainty_mapping_opt.get('use_learnable', False) and
        uncertainty_mapping_opt.get('staged_training', False) and
        uncertainty_mapping_opt.get('freeze_others_initially', False)
    )
    
    if use_staged_training:
        # 获取切换到全训练的时机
        switch_iter = opt['train'].get('warmup_switch_iter', 
                                      opt['train'].get('total_iter', 10000))
        logger.info(f"Using staged training: will switch to full training at iter {switch_iter}")
    else:
        switch_iter = None
        logger.info("Using joint training from the beginning")
    
    # 训练循环
    start_iter = 0
    if opt['path'].get('resume_state'):
        start_iter = model.resume_training(opt['path']['resume_state'])
        logger.info(f'Resuming training from iteration {start_iter}')
    
    total_iters = opt['train']['total_iter']
    logger.info(f'Start training from iteration {start_iter}, total iterations: {total_iters}')
    
    data_iter = iter(train_loader)
    switched_to_full_training = False
    
    for current_iter in range(start_iter, total_iters):
        # 获取训练数据
        try:
            train_data = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            train_data = next(data_iter)
        
        # 检查是否需要切换到全训练模式
        if use_staged_training and not switched_to_full_training:
            if current_iter >= switch_iter:
                logger.info(f"\n{'='*60}")
                logger.info(f"Switching to full training mode at iteration {current_iter}")
                logger.info(f"{'='*60}\n")
                model.switch_to_full_training()
                switched_to_full_training = True
                
                # 可选：保存切换时的checkpoint
                save_filename = f'net_g_stage1_final.pth'
                save_path = os.path.join(opt['path']['models'], save_filename)
                model.save_network(model.net_g, save_path)
                
                if hasattr(model, 'uncertainty_mapper') and model.uncertainty_mapper is not None:
                    mapper_save_path = os.path.join(opt['path']['models'], 
                                                   'uncertainty_mapper_stage1_final.pth')
                    model.save_network(model.uncertainty_mapper, mapper_save_path)
                
                logger.info(f"Stage 1 models saved.")
        
        # 更新学习率
        model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
        
        # 训练一步
        model.feed_data(train_data)
        model.optimize_parameters(current_iter)
        
        # 日志输出
        if current_iter % opt['logger']['print_freq'] == 0:
            log_vars = model.get_current_log()
            message = f'[Iter {current_iter:6d}] '
            for k, v in log_vars.items():
                message += f'{k}: {v:.4e} '
            
            # 添加学习率信息
            lrs = model.get_current_learning_rate()
            if isinstance(lrs, list):
                message += f'lr: {lrs[0]:.3e} '
            else:
                message += f'lr: {lrs:.3e} '
            
            logger.info(message)
            
            # 如果使用可学习映射，输出映射统计信息
            if (hasattr(model, 'uncertainty_mapper') and 
                model.uncertainty_mapper is not None and
                current_iter % (opt['logger']['print_freq'] * 10) == 0):
                
                try:
                    # 获取最近一次的diff用于统计
                    with torch.no_grad():
                        # 简单统计
                        mapper_params = sum(p.numel() for p in model.uncertainty_mapper.parameters())
                        mapper_grad_norm = sum(
                            p.grad.norm().item() for p in model.uncertainty_mapper.parameters() 
                            if p.grad is not None
                        )
                        logger.info(f'  Uncertainty Mapper: {mapper_params} params, '
                                  f'grad_norm: {mapper_grad_norm:.4e}')
                except Exception as e:
                    pass
        
        # 验证
        if current_iter % opt['val']['val_freq'] == 0:
            logger.info(f'\n{"="*60}')
            logger.info(f'Validation at iteration {current_iter}')
            logger.info(f'{"="*60}')
            
            model.validation(
                val_loader,
                current_iter,
                tb_logger=None,
                save_img=opt['val'].get('save_img', False)
            )
            
            logger.info(f'{"="*60}\n')
        
        # 保存checkpoint
        if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
            logger.info(f'Saving models and training states at iteration {current_iter}')
            model.save(current_iter)
            
            # 单独保存可学习映射层
            if hasattr(model, 'uncertainty_mapper') and model.uncertainty_mapper is not None:
                mapper_save_path = os.path.join(
                    opt['path']['models'], 
                    f'uncertainty_mapper_{current_iter}.pth'
                )
                model.save_network(model.uncertainty_mapper, mapper_save_path)
    
    # 训练完成
    logger.info(f'\n{"="*60}')
    logger.info('Training completed!')
    logger.info(f'{"="*60}\n')
    
    # 保存最终模型
    logger.info('Saving final models...')
    model.save(epoch=-1, current_iter=total_iters)
    
    if hasattr(model, 'uncertainty_mapper') and model.uncertainty_mapper is not None:
        mapper_save_path = os.path.join(opt['path']['models'], 'uncertainty_mapper_final.pth')
        model.save_network(model.uncertainty_mapper, mapper_save_path)
    
    logger.info('All done!')


if __name__ == '__main__':
    main()
