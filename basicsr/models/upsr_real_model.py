import torch
import torch.amp as amp
import torch.nn as nn
import functools
import math
import lpips
import os
import os.path as osp
import pyiqa
import numpy as np
import random
import tqdm
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import matplotlib.pyplot as plt
from .uncertainty_mapping import (
    LearnableUncertaintyMapping, 
    SpatialUncertaintyMapping,
    ContentAwareSpatialUncertaintyMapping
)


# from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import get_obj_from_str, get_root_logger, ImageSpliterTh, imwrite, tensor2img
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from .base_model import BaseModel
from .sr_model import SRModel
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from contextlib import nullcontext
from copy import deepcopy
from torch.nn.parallel import DataParallel
from collections import OrderedDict
from torchvision import transforms
from PIL import Image


class UPSRRealModel(SRModel):
    """Diffusion SR model for single image super-resolution."""

    def __init__(self, opt):
        self.opt = opt
        logger = get_root_logger()
        
        # ⚠️ 重要：在调用 super().__init__() 之前初始化这些属性
        # 因为父类的 __init__ 会调用 init_training_settings()，而该方法需要这些属性
        uncertainty_mapping_opt = self.opt.get('uncertainty_mapping', {})
        self.use_learnable_mapping = uncertainty_mapping_opt.get('use_learnable', False)
        self.uncertainty_mapper = None  # 先设为 None，后面再初始化
        
        # 调用父类初始化（会触发 init_training_settings）
        super(UPSRRealModel, self).__init__(opt)

        self.sf = self.opt['scale']

        # define network net_mse g(\cdot)
        net_mse_opt = self.opt['network_mse']
        assert net_mse_opt['ckpt']['path'] is not None, 'ckpt_path is required for net_mse'
        logger.info(f"Restoring network_mse from {net_mse_opt['ckpt']['path']}")

        self.net_mse = build_network(net_mse_opt)
        param_key = net_mse_opt['ckpt'].get('param_key_mse', 'params_ema')
        self.load_network(self.net_mse, net_mse_opt['ckpt']['path'], net_mse_opt['ckpt'].get('strict_load_mse', True), param_key)
        self.net_mse.eval()
        for name, param in self.net_mse.named_parameters():
            param.requires_grad = False
        self.net_mse = self.net_mse.to(self.device)

        # define base_diffusion
        diff_opt = self.opt['diffusion']
        self.base_diffusion = build_network(diff_opt)
        
        # 现在初始化可学习的不确定度映射层（如果启用）
        if self.use_learnable_mapping:
            logger.info("Using learnable uncertainty mapping")
            mapping_type = uncertainty_mapping_opt.get('type', 'mlp')  # 'mlp', 'spatial', or 'content_aware'
            un_max = self.opt['diffusion'].get('un', 1.0)
            min_noise = self.opt['diffusion'].get('min_noise', 0.0)
            
            if mapping_type == 'spatial':
                self.uncertainty_mapper = SpatialUncertaintyMapping(
                    un_max=un_max,
                    min_noise=min_noise,
                    channels=3,  # RGB图像
                    hidden_channels=uncertainty_mapping_opt.get('hidden_channels', 32)
                ).to(self.device)
            elif mapping_type == 'content_aware':
                # ✅ 新增：内容感知的空间映射（带预训练语义编码器）
                logger.info("Using content-aware spatial uncertainty mapping with pretrained semantic encoder")
                self.uncertainty_mapper = ContentAwareSpatialUncertaintyMapping(
                    un_max=un_max,
                    min_noise=min_noise,
                    channels=3,  # RGB图像
                    hidden_channels=uncertainty_mapping_opt.get('hidden_channels', 64),
                    num_heads=uncertainty_mapping_opt.get('num_heads', 4),
                    window_size=uncertainty_mapping_opt.get('window_size', 8),
                    use_pretrained_semantic=uncertainty_mapping_opt.get('use_pretrained_semantic', True),
                    semantic_model=uncertainty_mapping_opt.get('semantic_model', 'clip')  # 'clip' or 'deit'
                ).to(self.device)
            else:  # 'mlp'
                self.uncertainty_mapper = LearnableUncertaintyMapping(
                    un_max=un_max,
                    min_noise=min_noise,
                    hidden_dim=uncertainty_mapping_opt.get('hidden_dim', 64),
                    num_layers=uncertainty_mapping_opt.get('num_layers', 3),
                    use_residual=uncertainty_mapping_opt.get('use_residual', True)
                ).to(self.device)
            
            # 加载预训练的映射层权重（如果提供）
            mapping_ckpt_path = uncertainty_mapping_opt.get('ckpt_path', None)
            if mapping_ckpt_path is not None and os.path.exists(mapping_ckpt_path):
                logger.info(f"Loading uncertainty mapper from {mapping_ckpt_path}")
                self.load_network(self.uncertainty_mapper, mapping_ckpt_path, 
                                uncertainty_mapping_opt.get('strict_load', True), 'params')
        else:
            logger.info("Using fixed uncertainty mapping")
        
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 160)

        # define lpips loss
        loss_lpips = self.metric_lpips = pyiqa.create_metric('lpips-vgg', as_loss=True, device=self.device)
        self.loss_lpips = loss_lpips

        if self.opt['rank'] == 0:
            self.metrics_fr = {}
            self.metrics_nr = {}

            for metric_name, metric_opt in self.opt['val']['metrics'].items():
                if metric_opt.get('fr', True):
                    self.metrics_fr[metric_name] = pyiqa.create_metric(metric_name, device=self.device)
                else:
                    self.metrics_nr[metric_name] = pyiqa.create_metric(metric_name, device=self.device)

            self.metrics_fr['psnr'] = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=self.device)
            self.metrics_fr['ssim'] = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr', device=self.device)


    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            if param_key in load_net:
                load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def init_training_settings(self):

        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = True
            self.perceptual_weight = train_opt['perceptual_opt']['lpips_weight']
        else:
            self.cri_perceptual = False

        # 🔥 新增：初始化用于 Uncertainty Guidance 的 VGG 特征提取器
        if self.use_learnable_mapping:
            logger = get_root_logger()
            logger.info("Initializing VGG feature extractor for high-res perceptual guidance...")
            import torchvision
            # 利用 torchvision 加载 VGG19
            # weights='DEFAULT' 对应 VGG19_Weights.IMAGENET1K_V1
            try:
                vgg = torchvision.models.vgg19(weights='DEFAULT')
            except:
                vgg = torchvision.models.vgg19(pretrained=True)
                
            # 取到 features[8] 大约是 relu2_2 之前，分辨率为输入的一半 (128x128)
            # 0-3: conv1_1, relu, conv1_2, relu, maxpool (->128)
            # 5-8: conv2_1, relu, conv2_2, relu
            self.vgg_features = nn.Sequential(*list(vgg.features.children())[:9]).to(self.device).eval()
            for k, v in self.vgg_features.named_parameters():
                v.requires_grad = False

        # 设置可学习映射层的训练模式
        if self.use_learnable_mapping and self.uncertainty_mapper is not None:
            self.uncertainty_mapper.train()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        self.amp_scaler = amp.GradScaler() if self.opt['train'].get('use_fp16', False) else None
    
    def setup_optimizers(self):
        """设置优化器，支持分阶段训练策略"""
        train_opt = self.opt['train']
        logger = get_root_logger()
        
        # 收集需要优化的参数
        optim_params = []
        
        # 主网络参数
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger.warning(f'Params {k} will not be optimized.')
        
        # 如果使用可学习映射且存在，则添加其参数
        uncertainty_mapping_opt = self.opt.get('uncertainty_mapping', {})
        if self.use_learnable_mapping and self.uncertainty_mapper is not None:
            # 检查是否使用分阶段训练
            use_staged_training = uncertainty_mapping_opt.get('staged_training', True)
            freeze_others_initially = uncertainty_mapping_opt.get('freeze_others_initially', True)
            
            if use_staged_training and freeze_others_initially:
                # 第一阶段：只训练映射层
                logger.info("Staged training enabled: Starting with uncertainty mapper only")
                optim_params = []  # 清空主网络参数
                
                # 冻结主网络
                for param in self.net_g.parameters():
                    param.requires_grad = False
                
                # 只添加映射层参数
                for k, v in self.uncertainty_mapper.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                        logger.info(f'Uncertainty mapper param {k} will be optimized.')
            else:
                # 联合训练或第二阶段：使用不同的学习率
                mapper_lr_scale = uncertainty_mapping_opt.get('lr_scale', 1.0)
                
                if mapper_lr_scale != 1.0:
                    # 使用参数组以设置不同学习率
                    optim_type = train_opt['optim_g'].pop('type')
                    base_lr = train_opt['optim_g'].get('lr', 1e-4)
                    
                    param_groups = [
                        {'params': optim_params, 'lr': base_lr},
                        {'params': self.uncertainty_mapper.parameters(), 
                         'lr': base_lr * mapper_lr_scale}
                    ]
                    
                    logger.info(f'Using different learning rates: base_lr={base_lr}, '
                              f'mapper_lr={base_lr * mapper_lr_scale}')
                    
                    self.optimizer_g = self.get_optimizer(optim_type, param_groups, 
                                                         **train_opt['optim_g'])
                    self.optimizers.append(self.optimizer_g)
                    return
                else:
                    # 相同学习率，直接添加参数
                    for k, v in self.uncertainty_mapper.named_parameters():
                        if v.requires_grad:
                            optim_params.append(v)
                            logger.info(f'Uncertainty mapper param {k} will be optimized.')
        
        # 创建优化器
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
    

    def backward_step(self, dif_loss_wrapper, micro_lq, micro_gt, num_grad_accumulate, tt, constraint_loss=0):
        # 输入的dif_loss_wrapper：terms(损失字典), self.decode_first_stage(x_t), self.decode_first_stage(pred_xstart)
        loss_dict = OrderedDict()

        context = amp.autocast if self.opt['train'].get('use_fp16', False) else nullcontext
        with context(device_type="cuda"):
            losses, x_t, x0_pred = dif_loss_wrapper()
            losses['loss'] = losses['mse']
            l_pix = losses['loss'].mean() / num_grad_accumulate
            # 用于计算像素空间的损失
            l_total = l_pix
            loss_dict['l_pix'] = l_pix 

            if self.cri_perceptual:
                l_lpips = self.loss_lpips(x0_pred.clamp(-1., 1.), micro_gt).to(x0_pred.dtype).view(-1)
                if torch.any(torch.isnan(l_lpips)):
                    l_lpips = torch.nan_to_num(l_lpips, nan=0.0)
                l_lpips = l_lpips.mean() / num_grad_accumulate * self.perceptual_weight

                l_total += l_lpips
                loss_dict['l_lpips'] = l_lpips
            
            # 🔥 添加约束损失（guidance + sparsity）
            if constraint_loss != 0:
                l_total += constraint_loss / num_grad_accumulate 

        # 反向传播
        if self.amp_scaler is None:
            l_total.backward()
        else:
            self.amp_scaler.scale(l_total).backward()

        return loss_dict, x_t, x0_pred

    def optimize_parameters(self, current_iter):  
        # 初始化梯度累积参数,执行完成后一轮更新结束
        current_batchsize = self.lq.shape[0]
        micro_batchsize = self.opt['datasets']['train']['micro_batchsize']
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        self.optimizer_g.zero_grad()
        # 用于计算总损失的字典
        loss_dict = OrderedDict()
        loss_dict['l_pix'] = 0
        if self.cri_perceptual:
            loss_dict['l_lpips'] = 0
        
        # 🔥 初始化约束损失项
        if self.use_learnable_mapping and self.uncertainty_mapper is not None:
            # 必须使用Tensor初始化，否则多GPU reduce时会报错
            zero_tensor = torch.tensor(0., device=self.device)
            loss_dict['l_guidance'] = zero_tensor.clone()
            loss_dict['l_contrast'] = zero_tensor.clone()
            # loss_dict['l_reg'] = zero_tensor.clone() # 已弃用，直接注释掉
        
        # 微批次循环
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_lq = self.lq[jj:jj+micro_batchsize,]
            micro_gt = self.gt[jj:jj+micro_batchsize,]


            last_batch = (jj+micro_batchsize >= current_batchsize)
            if self.opt['diffusion'].get('one_step', False):
                tt = torch.ones(
                    size=(micro_gt.shape[0],),
                    device=self.lq.device,
                    dtype=torch.int32,
                    ) * (self.base_diffusion.num_timesteps - 1)
            else:
                tt = torch.randint(
                        0, self.base_diffusion.num_timesteps,
                        size=(micro_gt.shape[0],),
                        device=self.lq.device,
                        )
            
            with torch.no_grad():        
                # y_0 插值放大的图片
                micro_lq_bicubic = torch.nn.functional.interpolate(
                        micro_lq, scale_factor=self.sf, mode='bicubic', align_corners=False,
                        )
                # g(y_0) 生成辅助SR网络的预测结果 【像素空间】
                micro_sr_mse = (self.net_mse(micro_lq * 0.5 + 0.5) - 0.5) / 0.5

                # un 不确定性估计
                if self.opt['diffusion']['un'] > 0:
                    diff = (micro_sr_mse - micro_lq_bicubic) / 2
                    
                    if self.use_learnable_mapping and self.uncertainty_mapper is not None:
                        # 检查是否为内容感知映射
                        if isinstance(self.uncertainty_mapper, ContentAwareSpatialUncertaintyMapping):
                            # ✅ 内容感知映射：需要LQ图像
                            # 将LQ上采样到HR尺寸并归一化到[0,1]
                            micro_lq_hr = F.interpolate(micro_lq, scale_factor=self.sf, mode='bicubic', align_corners=False)
                            micro_lq_hr = micro_lq_hr * 0.5 + 0.5  # 从[-1,1]转到[0,1]
                            micro_uncertainty = self.uncertainty_mapper(diff, micro_lq_hr)
                        else:
                            # 普通可学习映射：只需要diff
                            micro_uncertainty = self.uncertainty_mapper(diff)
                    else:
                        # 使用固定映射
                        un_max = self.opt['diffusion']['un']
                        b_un = self.opt['diffusion']['min_noise']
                        micro_uncertainty = torch.abs(diff).clamp_(0., un_max) / un_max
                        micro_uncertainty = b_un + (1 - b_un) * micro_uncertainty
                else:
                    micro_uncertainty = torch.ones_like(micro_sr_mse)
            
            # 🔥 改进的 Guidance Loss（High-Res Perceptual Error with VGG + Scobel）
            if self.use_learnable_mapping and self.uncertainty_mapper is not None:
                train_opt = self.opt.get('train', {})
                
                guidance_weight = train_opt.get('guidance_loss_weight', 0)
                if guidance_weight > 0:
                    with torch.no_grad():
                        # ========================================================
                        # 🔥 核心修正：构建高分辨率的感知误差图 (Target Map)
                        # ========================================================
                        
                        # Normalize inputs for VGG (expecting [0,1] then normalized)
                        # micro_sr_mse is [-1, 1], convert to [0, 1] first
                        sr_01 = micro_sr_mse * 0.5 + 0.5
                        gt_01 = micro_gt * 0.5 + 0.5
                        
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(self.device)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(self.device)
                        
                        norm_sr = (sr_01 - mean) / std
                        norm_gt = (gt_01 - mean) / std
                        
                        # 1. 计算 VGG 特征差异 (Semantic/Texture Error) [ 1/2 Resolution ]
                        # self.vgg_features 应该在 init 中已定义
                        if hasattr(self, 'vgg_features'):
                            feat_sr = self.vgg_features(norm_sr) 
                            feat_gt = self.vgg_features(norm_gt)
                            # 计算 L2 距离
                            feat_diff = (feat_sr - feat_gt).pow(2).mean(dim=1, keepdim=True).sqrt()
                            # 上采样回原分辨率
                            feat_diff_up = F.interpolate(feat_diff, size=micro_gt.shape[2:], mode='bilinear')
                        else:
                            # Fallback if vgg not initialized
                            feat_diff_up = torch.abs(micro_sr_mse - micro_gt).mean(1, keepdim=True)

                        # 2. 计算 梯度/边缘 差异 (High-Frequency Edge Error) [ Full Resolution ]
                        def get_gradient(img):
                            # 简单的梯度计算: |dx| + |dy|
                            dx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
                            dy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
                            dx = F.pad(dx, (0, 1, 0, 0))
                            dy = F.pad(dy, (0, 0, 0, 1))
                            return dx + dy
                        
                        grad_diff = torch.abs(get_gradient(micro_sr_mse) - get_gradient(micro_gt)).mean(dim=1, keepdim=True)
                        
                        # 3. 融合生成最终 Target
                        def safe_norm(x):
                            return x / (x.max().detach() + 1e-6)
                        
                        # 组合权重：0.6 VGG + 0.4 Gradient
                        target_raw = 0.6 * safe_norm(feat_diff_up) + 0.4 * safe_norm(grad_diff)
                        
                        # 4. 最终增强 (放大误差)
                        target_map = torch.tanh(target_raw * 5.0) 

                        # ========================================================
                        # 生成 Ranking Mask
                        # ========================================================
                        
                        B = target_map.shape[0]
                        flat_diff = target_map.view(B, -1)
                        
                        # Top 25% Hard, Bottom 40% Easy
                        k_hard = int(flat_diff.size(1) * 0.25)
                        k_easy = int(flat_diff.size(1) * 0.40) 

                        # Mask Hard
                        hard_val, _ = torch.topk(flat_diff, k_hard, dim=1)
                        threshold_hard = hard_val[:, -1].view(B, 1, 1, 1)
                        hard_mask = (target_map >= threshold_hard).float()
                        
                        # Mask Easy
                        easy_val, _ = torch.topk(flat_diff, k_easy, dim=1, largest=False)
                        threshold_easy = easy_val[:, -1].view(B, 1, 1, 1)
                        easy_mask = (target_map <= threshold_easy).float()
                    
                    # ✅ 计算 Loss
                    # Hard 区域的 Uncertainty 均值
                    pred_hard_mean = (micro_uncertainty * hard_mask).sum() / (hard_mask.sum() + 1e-6)
                    # Easy 区域的 Uncertainty 均值
                    pred_easy_mean = (micro_uncertainty * easy_mask).sum() / (easy_mask.sum() + 1e-6)
                    
                    # 1. Ranking Loss: Hard > Easy + margin
                    margin = 0.25
                    loss_contrast = F.relu(margin - (pred_hard_mean - pred_easy_mean))
                    
                    # 2. Anchor Loss: Easy -> 0
                    loss_anchor = (micro_uncertainty * easy_mask).mean()
                    
                    # 组合
                    loss_contrast_norm = loss_contrast  # 这里的 norm 只是命名习惯, 实际上未除以margin
                    
                    alpha = train_opt.get('contrast_weight', 1.0)
                    # 这里的权重策略：Ranking权重2.0, Anchor权重1.0
                    guidance_loss = alpha * (2.0 * loss_contrast + 1.0 * loss_anchor)
                    
                    # 记录到loss_dict
                    loss_dict['l_guidance'] += guidance_loss / num_grad_accumulate
                    loss_dict['l_contrast'] = loss_contrast.detach() # Monitor
                    
                    # 调试日志
                    if torch.rand(1).item() < 0.005:  # 0.5%概率打印
                        logger = get_root_logger()
                        logger.info(f"[Guidance] hard={pred_hard_mean.item():.3f}, easy={pred_easy_mean.item():.3f}, "
                                    f"loss_contrast={loss_contrast.item():.4f}, loss_anchor={loss_anchor.item():.4f}")
                else:
                    guidance_loss = 0
                
                constraint_loss = guidance_weight * guidance_loss
            else:
                constraint_loss = 0

            # n
            noise = torch.randn_like(micro_sr_mse)

            lq_cond = nn.PixelUnshuffle(self.sf)(torch.cat([micro_sr_mse, micro_lq_bicubic], dim=1))


            model_kwargs={'lq':lq_cond,} if self.opt['network_g']['params']['cond_lq'] else None
            # 返回：terms(损失字典), self.decode_first_stage(x_t), self.decode_first_stage(pred_xstart)
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.net_g,
                micro_gt,
                micro_lq_bicubic,
                micro_sr_mse,
                micro_uncertainty,
                tt,
                model_kwargs=model_kwargs,
                noise=noise,
            )

            if last_batch or self.opt['num_gpu'] <= 1:
                losses, x_t, x0_pred = self.backward_step(compute_losses, micro_lq, micro_gt, num_grad_accumulate, tt, constraint_loss)
            else:
                # 检查是否使用了DistributedDataParallel
                # DataParallel没有no_sync()方法，需要特殊处理
                if hasattr(self.net_g, 'no_sync'):
                    # 使用DDP时，no_sync()可以避免中间batch的梯度同步
                    with self.net_g.no_sync():
                        losses, x_t, x0_pred = self.backward_step(compute_losses, micro_lq, micro_gt, num_grad_accumulate, tt, constraint_loss)
                else:
                    # 使用DataParallel或单GPU时，直接执行backward
                    losses, x_t, x0_pred = self.backward_step(compute_losses, micro_lq, micro_gt, num_grad_accumulate, tt, constraint_loss)
            
            loss_dict['l_pix'] += losses['l_pix']
            if self.cri_perceptual:
                loss_dict['l_lpips'] += losses['l_lpips']
        # 更新网络参数    
        if self.opt['train'].get('use_fp16', False):
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()
        else:
            self.optimizer_g.step()

        self.net_g.zero_grad()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def sample_func(self, y0, noise_repeat=False, save_intermediate=False):
        # 推理阶段使用
        desired_min_size = self.opt['val']['desired_min_size']
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False

        y_bicubic = torch.nn.functional.interpolate(
            y0, scale_factor=self.sf, mode='bicubic', align_corners=False,
            )
        
        y_hat = (self.net_mse(y0 * 0.5 + 0.5) - 0.5) / 0.5
        if self.opt['diffusion']['un'] > 0:
            diff = (y_hat - y_bicubic) / 2
            
            if self.use_learnable_mapping and self.uncertainty_mapper is not None:
                # 使用可学习映射（推理时）
                self.uncertainty_mapper.eval()
                with torch.no_grad():
                    # 检查是否为内容感知映射
                    if isinstance(self.uncertainty_mapper, ContentAwareSpatialUncertaintyMapping):
                        # ✅ 内容感知映射：需要LQ图像
                        y0_hr = F.interpolate(y0, scale_factor=self.sf, mode='bicubic', align_corners=False)
                        y0_hr = y0_hr * 0.5 + 0.5  # 从[-1,1]转到[0,1]
                        un = self.uncertainty_mapper(diff, y0_hr)
                    else:
                        # 普通可学习映射：只需要diff
                        un = self.uncertainty_mapper(diff)
            else:
                # 使用固定映射
                un_max = self.opt['diffusion']['un']
                b_un = self.opt['diffusion']['min_noise']
                un = torch.abs(diff).clamp_(0., un_max) / un_max
                un = b_un + (1 - b_un) * un
        else:
            un = torch.ones_like(y_hat)

        lq_cond = nn.PixelUnshuffle(self.sf)(torch.cat([y_hat, y_bicubic], dim=1))

        model_kwargs={'lq':lq_cond,} if self.opt['network_g']['params']['cond_lq'] else None
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net = self.net_g_ema
        else:
            self.net_g.eval()
            net = self.net_g
        results, noisy_start = self.base_diffusion.ddim_sample_loop(
                y=y_bicubic,
                y_hat=y_hat,
                un=un,
                model=net,
                first_stage_model=None,
                noise=None,
                noise_repeat=noise_repeat,
                # clip_denoised=(self.autoencoder is None),
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                one_step=self.opt['diffusion'].get('one_step', False),
                return_noisy_start=save_intermediate,  # 返回加噪起点
                )    

        if flag_pad:
            results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]
            if save_intermediate and noisy_start is not None:
                un = un[:, :, :ori_h*self.sf, :ori_w*self.sf]
                noisy_start = noisy_start[:, :, :ori_h*self.sf, :ori_w*self.sf]

        # 保存中间结果用于可视化
        if save_intermediate:
            self.uncertainty_map = un
            self.noisy_start = noisy_start
            self.sr_mse_pred = y_hat  # 辅助SR网络的预测结果
            self.bicubic_upscale = y_bicubic  # 双三次插值放大的结果

        return results.clamp_(-1.0, 1.0)

    def test(self, save_intermediate=False):

        def _process_per_image(im_lq_tensor):
            if im_lq_tensor.shape[2] > self.opt['val']['chop_size'] or im_lq_tensor.shape[3] > self.opt['val']['chop_size']:
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.opt['val']['chop_size'],
                        stride=self.opt['val']['chop_stride'],
                        sf=self.opt['scale'],
                        extra_bs=self.opt['val']['chop_bs'],
                        )
                for im_lq_pch, index_infos in im_spliter:
                    im_sr_pch = self.sample_func(
                            (im_lq_pch - 0.5) / 0.5,
                            noise_repeat=self.opt['val']['noise_repeat'],
                            save_intermediate=save_intermediate,
                            )     # 1 x c x h x w, [-1, 1]
                    im_spliter.update(im_sr_pch, index_infos)
                im_sr_tensor = im_spliter.gather()
            else:
                im_sr_tensor = self.sample_func(
                        (im_lq_tensor - 0.5) / 0.5,
                        noise_repeat=self.opt['val']['noise_repeat'],
                        save_intermediate=save_intermediate,
                        )     # 1 x c x h x w, [-1, 1]

            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor
        
        self.output = _process_per_image(self.lq)


    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data, training=True):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if training and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            # USM sharpen the GT images
            if self.opt['degradation']['use_sharp'] is True:
                self.gt = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['degradation']['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['degradation']['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['degradation']['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['degradation']['gray_noise_prob']
            if np.random.uniform() < self.opt['degradation']['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['degradation']['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['degradation']['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['degradation']['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['degradation']['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['degradation']['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['degradation']['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['degradation']['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['degradation']['scale'] * scale), int(ori_w / self.opt['degradation']['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['degradation']['gray_noise_prob2']
            if np.random.uniform() < self.opt['degradation']['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['degradation']['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['degradation']['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['degradation']['scale'], ori_w // self.opt['degradation']['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['degradation']['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['degradation']['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['degradation']['scale'], ori_w // self.opt['degradation']['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['degradation']['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['degradation']['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
            # normalize
            # if self.mean is not None or self.std is not None:
            self.lq = (self.lq - 0.5) / 0.5
            self.gt = (self.gt - 0.5) / 0.5
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
            else:
                self.gt = None

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # 数据集名称
        dataset_name = dataloader.dataset.opt['name']
        # 是否计算指标
        with_metrics = self.opt['val'].get('metrics') is not None
        # 是否显示进度条
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        # 创建一个空字典，用于临时存放模型输出的图像 (img) 和真实GT图像 (img2)，以便传递给指标计算函数
        metric_data = dict()
        if use_pbar:
            pbar = tqdm.tqdm(total=len(dataloader), unit='image')

        num_img = 0

        for idx, val_data in enumerate(dataloader):
            # 更新已处理的图片总数
            num_img += len(val_data['lq_path'])
            self.feed_data(val_data, training=False)
            
            # 测试时保存中间结果
            self.test(save_intermediate=save_img)

            metric_data['img'] = self.output.clamp(0, 1)
            # metric_data['img'] = torch.clamp((self.output * 255.0).round(), 0, 255) / 255.
            metric_data['img2'] = self.gt

            if with_metrics:
                # calculate metrics
                if metric_data['img2'] is not None:
                    for name, metric in self.metrics_fr.items():
                        self.metric_results[name] += metric(metric_data['img'], metric_data['img2']).sum().item()
                for name, metric in self.metrics_nr.items():
                    self.metric_results[name] += metric(metric_data['img']).sum().item()

            visuals = self.get_current_visuals()

            sr_img = [tensor2img(visuals['result'][ii]) for ii in range(self.output.shape[0])]
            
            # 保存中间结果图片
            uncertainty_imgs = None
            noisy_start_imgs = None
            sr_mse_imgs = None
            bicubic_imgs = None
            diff_tensors = None  # 新增：用于保存diff tensor用于热力图
            
            # 新增：计算Diff
            if save_img and hasattr(self, 'sr_mse_pred') and hasattr(self, 'bicubic_upscale'):
                if self.sr_mse_pred is not None and self.bicubic_upscale is not None:
                    diff_tensors = (self.sr_mse_pred - self.bicubic_upscale) / 2
            
            # 🔥 修改：增加 save_noise1 参数控制 (Upsr代码中 noise1 文件夹对应 uncertainty_map)
            if save_img and self.opt['val'].get('save_noise1', False) and hasattr(self, 'uncertainty_map') and self.uncertainty_map is not None:
                # 将不确定度图转换为可视化图像（归一化到0-255）
                uncertainty_imgs = []
                for ii in range(self.uncertainty_map.shape[0]):
                    # 取平均通道并归一化
                    un_map = self.uncertainty_map[ii].mean(dim=0, keepdim=True)  # [1, H, W]
                    un_map = un_map.clamp(0, 1)  # 确保在[0,1]范围
                    # 转换为RGB可视化（使用灰度图）
                    un_img = un_map.repeat(3, 1, 1)  # [3, H, W]
                    un_img_np = tensor2img(un_img.unsqueeze(0).cpu())  # 转换为numpy数组
                    uncertainty_imgs.append(un_img_np)
            
            # 🔥 修改：增加 save_noise2 参数控制 (Upsr代码中 noise2 文件夹对应 noisy_start)
            if save_img and self.opt['val'].get('save_noise2', False) and hasattr(self, 'noisy_start') and self.noisy_start is not None:
                # 将加噪起点图转换为可视化图像
                noisy_start_imgs = []
                noisy_vis = self.noisy_start * 0.5 + 0.5  # 从[-1,1]转到[0,1]
                for ii in range(noisy_vis.shape[0]):
                    noisy_img_np = tensor2img(noisy_vis[ii:ii+1].cpu())
                    noisy_start_imgs.append(noisy_img_np)
            
            # 🔥 修改：增加 save_sr_mse 参数控制
            if save_img and self.opt['val'].get('save_sr_mse', False) and hasattr(self, 'sr_mse_pred') and self.sr_mse_pred is not None:
                # 保存辅助SR网络预测结果
                sr_mse_imgs = []
                sr_mse_vis = self.sr_mse_pred * 0.5 + 0.5  # 从[-1,1]转到[0,1]
                for ii in range(sr_mse_vis.shape[0]):
                    sr_mse_img_np = tensor2img(sr_mse_vis[ii:ii+1].cpu())
                    sr_mse_imgs.append(sr_mse_img_np)
            
            # 🔥 修改：增加 save_bicubic 参数控制
            if save_img and self.opt['val'].get('save_bicubic', False) and hasattr(self, 'bicubic_upscale') and self.bicubic_upscale is not None:
                # 保存双三次插值放大结果
                bicubic_imgs = []
                bicubic_vis = self.bicubic_upscale * 0.5 + 0.5  # 从[-1,1]转到[0,1]
                for ii in range(bicubic_vis.shape[0]):
                    bicubic_img_np = tensor2img(bicubic_vis[ii:ii+1].cpu())
                    bicubic_imgs.append(bicubic_img_np)
            
            if 'gt' in visuals:
                # gt_img = tensor2img([visuals['gt']])
                # metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if hasattr(self, 'uncertainty_map'):
                del self.uncertainty_map
            if hasattr(self, 'noisy_start'):
                del self.noisy_start
            if hasattr(self, 'sr_mse_pred'):
                del self.sr_mse_pred
            if hasattr(self, 'bicubic_upscale'):
                del self.bicubic_upscale

            torch.cuda.empty_cache()

            
            for ii in range(len(val_data['lq_path'])):
                if save_img:
                    img_name = osp.splitext(osp.basename(val_data['lq_path'][ii]))[0]

                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}.png')
                        # 新增：保存Diff热力图
                        if diff_tensors is not None and self.opt['val'].get('save_diff_heatmap', False):
                            from basicsr.utils.diff_visualization import visualize_diff_heatmap
                            diff_save_dir = osp.join(self.opt['path']['visualization'], 'diff_heatmap', img_name)
                            os.makedirs(diff_save_dir, exist_ok=True)
                            visualize_diff_heatmap(
                                diff_tensors[ii:ii+1], 
                                diff_save_dir, 
                                prefix=f'{img_name}_{current_iter}',
                                save_channels=True
                            )
                        
                        # 保存不确定度图和加噪起点图
                        if uncertainty_imgs is not None:
                            uncertainty_save_path = osp.join(self.opt['path']['visualization'], 'noise1', img_name,
                                                           f'{img_name}_{current_iter}.png')
                            os.makedirs(osp.dirname(uncertainty_save_path), exist_ok=True)
                            imwrite(uncertainty_imgs[ii], uncertainty_save_path)
                            # 新增：保存Uncertainty热力图
                            if self.opt['val'].get('save_uncertainty_heatmap', False):
                                from basicsr.utils.diff_visualization import visualize_uncertainty_heatmap
                                uncertainty_heatmap_dir = osp.join(self.opt['path']['visualization'], 'uncertainty_heatmap', img_name)
                                os.makedirs(uncertainty_heatmap_dir, exist_ok=True)
                                visualize_uncertainty_heatmap(
                                    self.uncertainty_map[ii:ii+1],
                                    uncertainty_heatmap_dir,
                                    prefix=f'{img_name}_{current_iter}',
                                    save_channels=True
                                )
                        if noisy_start_imgs is not None:
                            noisy_save_path = osp.join(self.opt['path']['visualization'], 'noise2', img_name,
                                                      f'{img_name}_{current_iter}.png')
                            os.makedirs(osp.dirname(noisy_save_path), exist_ok=True)
                            imwrite(noisy_start_imgs[ii], noisy_save_path)
                        # 保存辅助SR网络预测结果
                        if sr_mse_imgs is not None:
                            sr_mse_save_path = osp.join(self.opt['path']['visualization'], 'sr_mse', img_name,
                                                       f'{img_name}_{current_iter}.png')
                            os.makedirs(osp.dirname(sr_mse_save_path), exist_ok=True)
                            imwrite(sr_mse_imgs[ii], sr_mse_save_path)
                        # 保存双三次插值放大结果
                        if bicubic_imgs is not None:
                            bicubic_save_path = osp.join(self.opt['path']['visualization'], 'bicubic', img_name,
                                                        f'{img_name}_{current_iter}.png')
                            os.makedirs(osp.dirname(bicubic_save_path), exist_ok=True)
                            imwrite(bicubic_imgs[ii], bicubic_save_path)
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["val"]["suffix"]}.png')
                            # 新增：保存Diff热力图
                            if diff_tensors is not None and self.opt['val'].get('save_diff_heatmap', False):
                                from basicsr.utils.diff_visualization import visualize_diff_heatmap
                                diff_save_dir = osp.join(self.opt['path']['visualization'], dataset_name, 'diff_heatmap')
                                os.makedirs(diff_save_dir, exist_ok=True)
                                visualize_diff_heatmap(
                                    diff_tensors[ii:ii+1], 
                                    diff_save_dir, 
                                    prefix=f'{img_name}_{self.opt["val"]["suffix"]}',
                                    save_channels=True
                                )
                            
                            # 保存不确定度图和加噪起点图
                            if uncertainty_imgs is not None:
                                uncertainty_save_path = osp.join(self.opt['path']['visualization'], dataset_name, 'noise1',
                                                               f'{img_name}_{self.opt["val"]["suffix"]}.png')
                                os.makedirs(osp.dirname(uncertainty_save_path), exist_ok=True)
                                imwrite(uncertainty_imgs[ii], uncertainty_save_path)
                                # 新增：保存Uncertainty热力图
                                if self.opt['val'].get('save_uncertainty_heatmap', False):
                                    from basicsr.utils.diff_visualization import visualize_uncertainty_heatmap
                                    uncertainty_heatmap_dir = osp.join(self.opt['path']['visualization'], dataset_name, 'uncertainty_heatmap')
                                    os.makedirs(uncertainty_heatmap_dir, exist_ok=True)
                                    visualize_uncertainty_heatmap(
                                        self.uncertainty_map[ii:ii+1],
                                        uncertainty_heatmap_dir,
                                        prefix=f'{img_name}_{self.opt["val"]["suffix"]}',
                                        save_channels=True
                                    )
                            if noisy_start_imgs is not None:
                                noisy_save_path = osp.join(self.opt['path']['visualization'], dataset_name, 'noise2',
                                                          f'{img_name}_{self.opt["val"]["suffix"]}.png')
                                os.makedirs(osp.dirname(noisy_save_path), exist_ok=True)
                                imwrite(noisy_start_imgs[ii], noisy_save_path)
                            # 保存辅助SR网络预测结果
                            if sr_mse_imgs is not None:
                                sr_mse_save_path = osp.join(self.opt['path']['visualization'], dataset_name, 'sr_mse',
                                                           f'{img_name}_{self.opt["val"]["suffix"]}.png')
                                os.makedirs(osp.dirname(sr_mse_save_path), exist_ok=True)
                                imwrite(sr_mse_imgs[ii], sr_mse_save_path)
                            # 保存双三次插值放大结果
                            if bicubic_imgs is not None:
                                bicubic_save_path = osp.join(self.opt['path']['visualization'], dataset_name, 'bicubic',
                                                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                                os.makedirs(osp.dirname(bicubic_save_path), exist_ok=True)
                                imwrite(bicubic_imgs[ii], bicubic_save_path)
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["name"]}.png')
                            # 新增：保存Diff热力图
                            if diff_tensors is not None and self.opt['val'].get('save_diff_heatmap', False):
                                from basicsr.utils.diff_visualization import visualize_diff_heatmap
                                diff_save_dir = osp.join(self.opt['path']['visualization'], dataset_name, 'diff_heatmap')
                                os.makedirs(diff_save_dir, exist_ok=True)
                                visualize_diff_heatmap(
                                    diff_tensors[ii:ii+1], 
                                    diff_save_dir, 
                                    prefix=f'{img_name}_{self.opt["name"]}',
                                    save_channels=True
                                )
                            
                            # 保存不确定度图和加噪起点图
                            if uncertainty_imgs is not None:
                                uncertainty_save_path = osp.join(self.opt['path']['visualization'], dataset_name, 'noise1',
                                                               f'{img_name}_{self.opt["name"]}.png')
                                os.makedirs(osp.dirname(uncertainty_save_path), exist_ok=True)
                                imwrite(uncertainty_imgs[ii], uncertainty_save_path)
                                # 新增：保存Uncertainty热力图
                                if self.opt['val'].get('save_uncertainty_heatmap', False):
                                    from basicsr.utils.diff_visualization import visualize_uncertainty_heatmap
                                    uncertainty_heatmap_dir = osp.join(self.opt['path']['visualization'], dataset_name, 'uncertainty_heatmap')
                                    os.makedirs(uncertainty_heatmap_dir, exist_ok=True)
                                    visualize_uncertainty_heatmap(
                                        self.uncertainty_map[ii:ii+1],
                                        uncertainty_heatmap_dir,
                                        prefix=f'{img_name}_{self.opt["name"]}',
                                        save_channels=True
                                    )
                            if noisy_start_imgs is not None:
                                noisy_save_path = osp.join(self.opt['path']['visualization'], dataset_name, 'noise2',
                                                          f'{img_name}_{self.opt["name"]}.png')
                                os.makedirs(osp.dirname(noisy_save_path), exist_ok=True)
                                imwrite(noisy_start_imgs[ii], noisy_save_path)
                            # 保存辅助SR网络预测结果
                            if sr_mse_imgs is not None:
                                sr_mse_save_path = osp.join(self.opt['path']['visualization'], dataset_name, 'sr_mse',
                                                           f'{img_name}_{self.opt["name"]}.png')
                                os.makedirs(osp.dirname(sr_mse_save_path), exist_ok=True)
                                imwrite(sr_mse_imgs[ii], sr_mse_save_path)
                            # 保存双三次插值放大结果
                            if bicubic_imgs is not None:
                                bicubic_save_path = osp.join(self.opt['path']['visualization'], dataset_name, 'bicubic',
                                                            f'{img_name}_{self.opt["name"]}.png')
                                os.makedirs(osp.dirname(bicubic_save_path), exist_ok=True)
                                imwrite(bicubic_imgs[ii], bicubic_save_path)
                            
                    imwrite(sr_img[ii], save_img_path)
                    
            
            if use_pbar:
                pbar.update(1)

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= num_img
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt') and self.gt is not None:
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
    
    def save(self, epoch, current_iter):
        """保存模型，包括主网络和映射层"""
        # 如显式关闭保存主网络，则仅记录训练状态（减小存储占用）
        save_main = self.opt.get('path', {}).get('save_main_network', True)
        if save_main:
            super().save(epoch, current_iter)
        else:
            # 仍然保存优化器/调度器状态，便于继续训练
            self.save_training_state(epoch, current_iter)
        
        # 如果使用可学习映射，额外保存映射层
        if self.use_learnable_mapping and self.uncertainty_mapper is not None:
            save_filename = f'uncertainty_mapper_{current_iter}.pth'
            save_path = osp.join(self.opt['path']['models'], save_filename)
            
            # 获取裸模型（去除DataParallel包装）
            mapper = self.get_bare_model(self.uncertainty_mapper)
            
            # 保存状态字典
            state_dict = mapper.state_dict()
            save_dict = {'params': state_dict}
            
            torch.save(save_dict, save_path)
            
            logger = get_root_logger()
            logger.info(f'Saved uncertainty mapper to {save_path}')
    
    def load_uncertainty_mapper(self, load_path, strict=True, param_key='params'):
        """加载不确定度映射层权重"""
        if not hasattr(self, 'uncertainty_mapper') or self.uncertainty_mapper is None:
            logger = get_root_logger()
            logger.warning('Uncertainty mapper not initialized, skipping load')
            return
        
        logger = get_root_logger()
        mapper = self.get_bare_model(self.uncertainty_mapper)
        
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        
        if param_key is not None and param_key in load_net:
            load_net = load_net[param_key]
        
        logger.info(f'Loading uncertainty mapper from {load_path}, with param key: [{param_key}].')
        
        # 移除不必要的 'module.' 前缀
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        
        self._print_different_keys_loading(mapper, load_net, strict)
        mapper.load_state_dict(load_net, strict=strict)
