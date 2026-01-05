"""
Learnable Uncertainty Mapping Module for UPSR
可学习的不确定度映射模块
"""

import torch
import torch.nn as nn
from basicsr.utils import get_root_logger


class LearnableUncertaintyMapping(nn.Module):
    """
    可学习的不确定度到权重映射层
    
    将固定的线性映射替换为可学习的神经网络，使用残差学习策略保证训练稳定性。
    
    核心思想：
    - 原始映射是逐像素的函数 f: R -> R
    - MLP学习这个映射函数，对所有空间位置共享权重
    - 输入一个标量 -> 输出一个标量
    
    Args:
        un_max (float): 不确定度的最大值
        min_noise (float): 最小噪声水平 (b_un)
        hidden_dim (int): 隐藏层维度，默认64
        num_layers (int): MLP层数，默认3
        use_residual (bool): 是否使用残差连接，默认True
    """
    
    def __init__(self, un_max=1.0, min_noise=0.2, hidden_dim=64, num_layers=3, use_residual=True):
        super().__init__()
        self.un_max = un_max
        self.min_noise = 0.2  # 🔥 提升底噪到0.2，避免过度平滑
        self.use_residual = use_residual
        
        # 构建MLP: 输入1维标量，输出1维标量
        # 这个MLP对所有空间位置和通道共享
        layers = []
        input_dim = 1
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # 使用LayerNorm提高稳定性
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))  # 轻微dropout防止过拟合
        
        # 最后一层输出：1维标量
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化为接近恒等映射
        self._init_as_identity()
    
    def _init_as_identity(self):
        """
        初始化网络使其输出接近于输入（恒等映射）
        这样初始状态就等价于固定映射
        """
        # 最后一层初始化为零，这样残差项初始为0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        
        # 其他层使用小权重初始化
        for module in self.mlp[:-1]:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def fixed_mapping(self, diff):
        """
        原始的固定映射函数（改进版：底噪0.2 + x²指数映射）
        
        改进点：
        1. min_noise = 0.2（提升底噪，避免过度平滑）
        2. 使用指数映射 x²（更温和的非线性）
        
        Args:
            diff (Tensor): 差异图 (micro_sr_mse - micro_lq_bicubic) / 2
        
        Returns:
            Tensor: 不确定权重，范围 [0.2, 1.0]
        """
        # 归一化到 [0, 1]
        normalized_diff = torch.abs(diff).clamp_(0., self.un_max) / self.un_max
        
        # 🔥 指数映射：x²（更温和的非线性，避免过度压缩）
        normalized_diff = torch.pow(normalized_diff, 2.0)
        
        # 应用底噪（现在为0.2）
        uncertainty = self.min_noise + (1 - self.min_noise) * normalized_diff
        return uncertainty
    
    def forward(self, diff):
        """
        前向传播 - 逐像素映射，MLP权重共享
        
        Args:
            diff (Tensor): 差异图，形状 [B, C, H, W]
        
        Returns:
            Tensor: 不确定权重，形状 [B, C, H, W]，范围 [min_noise, 1.0]
        """
        # 保存原始形状
        B, C, H, W = diff.shape
        
        # 首先通过固定映射获得基准权重
        base_weight = self.fixed_mapping(diff)  # [B, C, H, W]
        
        if self.use_residual:
            # 关键：将基准权重作为MLP的输入
            # 展平为 [B*C*H*W, 1]，每个值独立通过MLP
            base_flat = base_weight.reshape(-1, 1)  # [B*C*H*W, 1]
            
            # MLP学习残差调整：对每个像素值应用相同的映射函数
            adjustment_flat = self.mlp(base_flat)  # [B*C*H*W, 1]
            
            # 重塑回原始形状
            adjustment = adjustment_flat.reshape(B, C, H, W)
            
            # 组合：基准 + 学习的调整（残差学习）
            final_weight = base_weight + adjustment
        else:
            # 不使用残差，直接学习映射
            # 归一化的diff作为输入
            normalized_diff = torch.abs(diff).clamp_(0., self.un_max) / self.un_max
            normalized_flat = normalized_diff.reshape(-1, 1)  # [B*C*H*W, 1]
            
            # MLP直接输出权重
            final_flat = self.mlp(normalized_flat)  # [B*C*H*W, 1]
            final_weight = final_flat.reshape(B, C, H, W)
            
            # 应用min_noise约束和激活
            final_weight = self.min_noise + (1 - self.min_noise) * torch.sigmoid(final_weight)
        
        # 确保权重在合理范围 [min_noise, 1.0]
        final_weight = final_weight.clamp(min=self.min_noise, max=1.0)
        
        return final_weight
    
    def get_statistics(self, diff):
        """
        获取映射统计信息，用于监控训练过程
        
        Returns:
            dict: 包含基准权重、调整量、最终权重的统计信息
        """
        with torch.no_grad():
            base_weight = self.fixed_mapping(diff)
            final_weight = self.forward(diff)
            adjustment = final_weight - base_weight
            
            stats = {
                'base_weight_mean': base_weight.mean().item(),
                'base_weight_std': base_weight.std().item(),
                'adjustment_mean': adjustment.mean().item(),
                'adjustment_std': adjustment.std().item(),
                'final_weight_mean': final_weight.mean().item(),
                'final_weight_std': final_weight.std().item(),
                'adjustment_abs_max': adjustment.abs().max().item(),
            }
            
            return stats


class SpatialUncertaintyMapping(nn.Module):
    """
    空间感知的不确定度映射
    
    使用卷积网络考虑空间上下文信息，而不是逐像素独立映射
    
    Args:
        un_max (float): 不确定度的最大值
        min_noise (float): 最小噪声水平
        channels (int): 输入通道数
        hidden_channels (int): 隐藏层通道数
    """
    
    def __init__(self, un_max=1.0, min_noise=0.2, channels=3, hidden_channels=32):
        super().__init__()
        self.un_max = un_max
        self.min_noise = 0.2  # 🔥 提升底噪到0.2，避免过度平滑
        
        # 空间感知的卷积网络
        self.conv_net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),  # 使用BatchNorm2d而不是LayerNorm
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        
        # 初始化为接近零输出
        nn.init.zeros_(self.conv_net[-1].weight)
        nn.init.zeros_(self.conv_net[-1].bias)
        
        for module in self.conv_net[:-1]:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def fixed_mapping(self, diff):
        """固定映射函数（改进版：底噪0.2 + x²指数映射）"""
        # 归一化到 [0, 1]
        normalized_diff = torch.abs(diff).clamp_(0., self.un_max) / self.un_max
        
        # 🔥 指数映射：x²（更温和的非线性，避免过度压缩）
        normalized_diff = torch.pow(normalized_diff, 2.0)
        
        uncertainty = self.min_noise + (1 - self.min_noise) * normalized_diff
        return uncertainty
    
    def forward(self, diff):
        """
        前向传播
        
        Args:
            diff (Tensor): 差异图，形状 [B, C, H, W]
        
        Returns:
            Tensor: 不确定权重，形状 [B, C, H, W]
        """
        # 基准权重
        base_weight = self.fixed_mapping(diff)
        
        # 归一化输入
        diff_normalized = torch.abs(diff).clamp_(0., self.un_max) / self.un_max
        
        # 通过卷积网络学习空间调整
        adjustment = self.conv_net(diff_normalized)
        
        # 残差连接
        final_weight = base_weight + adjustment
        
        # 约束范围
        final_weight = final_weight.clamp(min=self.min_noise, max=1.0)
        
        return final_weight


class CrossAttentionBlock(nn.Module):
    """
    Window-based Cross-Attention块（内存高效版本）
    
    使用窗口注意力机制降低内存复杂度，从 O(H²W²) 降低到 O(W²HW)
    
    Args:
        dim (int): 特征维度
        num_heads (int): 注意力头数
        window_size (int): 窗口大小
        qkv_bias (bool): 是否使用QKV偏置
    """
    
    def __init__(self, dim, num_heads=4, window_size=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Query投影（来自LQ）
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        
        # Key, Value投影（来自Diff）
        self.kv_proj = nn.Conv2d(dim, dim * 2, 1, bias=qkv_bias)
        
        # 输出投影
        self.proj = nn.Conv2d(dim, dim, 1)
        
        # LayerNorm
        self.norm_q = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.norm_kv = nn.GroupNorm(num_groups=1, num_channels=dim)
        
    def window_partition(self, x, window_size):
        """
        将特征分割成窗口
        Args:
            x: [B, C, H, W]
            window_size: int
        Returns:
            windows: [B*num_windows, C, window_size, window_size]
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = windows.view(-1, C, window_size, window_size)
        return windows
    
    def window_reverse(self, windows, window_size, H, W):
        """
        将窗口合并回特征图
        Args:
            windows: [B*num_windows, C, window_size, window_size]
            window_size: int
            H, W: 原始高度和宽度
        Returns:
            x: [B, C, H, W]
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        C = windows.shape[1]
        x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        return x
        
    def forward(self, query, key_value):
        """
        Args:
            query (Tensor): LQ特征 [B, C, H, W]
            key_value (Tensor): Diff特征 [B, C, H, W]
        
        Returns:
            Tensor: 调制后的特征 [B, C, H, W]
        """
        B, C, H, W = query.shape
        
        # 确保 H 和 W 能被 window_size 整除
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            query = torch.nn.functional.pad(query, (0, pad_w, 0, pad_h), mode='reflect')
            key_value = torch.nn.functional.pad(key_value, (0, pad_w, 0, pad_h), mode='reflect')
        
        _, _, Hp, Wp = query.shape
        
        # 应用GroupNorm
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        
        # Q, K, V投影
        q = self.q_proj(q)  # [B, C, Hp, Wp]
        kv = self.kv_proj(kv)  # [B, 2*C, Hp, Wp]
        k, v = kv.chunk(2, dim=1)  # 各 [B, C, Hp, Wp]
        
        # 分割成窗口
        q_windows = self.window_partition(q, self.window_size)  # [B*nW, C, ws, ws]
        k_windows = self.window_partition(k, self.window_size)
        v_windows = self.window_partition(v, self.window_size)
        
        # 重塑为多头 [B*nW, heads, ws*ws, C//heads]
        BnW = q_windows.shape[0]
        ws2 = self.window_size * self.window_size
        
        q_windows = q_windows.reshape(BnW, self.num_heads, C // self.num_heads, ws2).permute(0, 1, 3, 2)
        k_windows = k_windows.reshape(BnW, self.num_heads, C // self.num_heads, ws2).permute(0, 1, 3, 2)
        v_windows = v_windows.reshape(BnW, self.num_heads, C // self.num_heads, ws2).permute(0, 1, 3, 2)
        
        # 窗口内的注意力计算 [B*nW, heads, ws*ws, ws*ws]
        attn = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        out_windows = (attn @ v_windows)  # [B*nW, heads, ws*ws, C//heads]
        
        # 重塑回窗口形状
        out_windows = out_windows.permute(0, 1, 3, 2).reshape(BnW, C, self.window_size, self.window_size)
        
        # 合并窗口
        out = self.window_reverse(out_windows, self.window_size, Hp, Wp)  # [B, C, Hp, Wp]
        
        # 去除padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        
        # 输出投影
        out = self.proj(out)
        
        # 残差连接
        out = out + query[:, :, :H, :W]
        
        return out


class ContentAwareSpatialUncertaintyMapping(nn.Module):
    """
    内容感知的空间不确定度映射
    
    使用LQ图像特征通过Cross-Attention调制diff特征，
    实现内容自适应的不确定度估计
    
    核心思想：
    - LQ图像提供内容和退化信息
    - Diff提供SR预测质量信息  
    - Cross-Attention学习如何根据LQ内容调制不确定度
    
    Args:
        un_max (float): 不确定度的最大值
        min_noise (float): 最小噪声水平
        channels (int): 输入通道数（RGB=3）
        hidden_channels (int): 隐藏层通道数
        num_heads (int): 注意力头数
    """
    
    def __init__(self, un_max=1.0, min_noise=0.2, channels=3, hidden_channels=64, num_heads=4, window_size=8):
        super().__init__()
        self.un_max = un_max
        self.min_noise = 0.2  # 🔥 提升底噪到0.2，避免过度平滑
        
        # 初始化logger和调用计数器
        self.logger = get_root_logger()
        self.forward_count = 0
        self._first_call_logged = False
        
        # LQ特征提取器
        self.lq_encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),  # ✅ 修改
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),  # ✅ 修改
            nn.ReLU(inplace=True),
        )
        
        # Diff特征提取器
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),  # ✅ 修改
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),  # ✅ 修改
            nn.ReLU(inplace=True),
        )
        
        # Window-based Cross-Attention模块（内存高效）
        self.cross_attn = CrossAttentionBlock(
            dim=hidden_channels,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=True 
        )
        # 后处理平滑层 进行窗口边界融合
        self.post_smooth = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels), # 保持一致用GN
            nn.GELU() 
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, channels, 1),
            nn.Tanh() # 加入tanh激活以限制调整范围到 [-1, 1]
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        # 输出层初始化为零（残差学习）
        nn.init.zeros_(self.output_proj[-2].weight)
        nn.init.zeros_(self.output_proj[-2].bias)
        
        # 其他层使用Kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def fixed_mapping(self, diff):
        """
        原始固定映射函数（改进版：底噪0.2 + x²指数映射）
        
        改进点：
        1. min_noise = 0.2（提升底噪，避免过度平滑）
        2. 使用指数映射 x²（更温和的非线性）
        
        x² 映射相比 x³ 更温和：
        - 0.5² = 0.25 vs 0.5³ = 0.125（中等值保留更多）
        - 0.8² = 0.64 vs 0.8³ = 0.512（边缘值更高）
        加上 min_noise=0.2，最终范围 [0.2, 1.0]，避免噪声过低导致平滑
        
        Args:
            diff (Tensor): 差异图 (micro_sr_mse - micro_lq_bicubic) / 2
        
        Returns:
            Tensor: 不确定权重，范围 [0.2, 1.0]
        """
        # 归一化到 [0, 1]
        normalized_diff = torch.abs(diff).clamp_(0., self.un_max) / self.un_max
        
        # 🔥 指数映射：x²（更温和的非线性，避免过度压缩）
        normalized_diff = torch.pow(normalized_diff, 2.0)
        
        # 应用底噪（现在为0.2）
        uncertainty = self.min_noise + (1 - self.min_noise) * normalized_diff
        return uncertainty
    
    def forward(self, diff, lq):
        """
        前向传播
        
        Args:
            diff (Tensor): SR预测差异，形状 [B, C, H, W]
            lq (Tensor): LQ图像（已上采样到HR尺寸），形状 [B, C, H, W]，范围 [0, 1]
        
        Returns:
            Tensor: 内容感知的不确定度权重，形状 [B, C, H, W]
        """
        self.forward_count += 1
        
        # 第一次调用时记录详细信息
        if not self._first_call_logged:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"ContentAwareSpatialUncertaintyMapping.forward() FIRST CALL")
            self.logger.info(f"Input shapes: diff={diff.shape}, lq={lq.shape}")
            self.logger.info(f"Device: {diff.device}")
            self.logger.info(f"Config: un_max={self.un_max}, min_noise={self.min_noise}")
            self.logger.info(f"{'='*70}\n")
            self._first_call_logged = True
        
        # 获取基准权重
        base_weight = self.fixed_mapping(diff)  # [B, C, H, W]
        
        # 每500次调用记录一次统计信息
        if self.forward_count % 5000 == 1:
            self.logger.info(f"\n[ContentAwareMapping #{self.forward_count}]")
            self.logger.info(f"  Diff: min={diff.min():.4f}, max={diff.max():.4f}, mean={diff.mean():.4f}, std={diff.std():.4f}")
            self.logger.info(f"  BaseWeight: min={base_weight.min():.4f}, max={base_weight.max():.4f}, mean={base_weight.mean():.4f}")
        
        # 归一化输入
        diff_norm = torch.abs(diff).clamp_(0., self.un_max) / self.un_max
        lq_norm = lq  # LQ已经在[0,1]范围
        
        # 特征提取
        lq_feat = self.lq_encoder(lq_norm)      # [B, D, H, W]
        diff_feat = self.diff_encoder(diff_norm)  # [B, D, H, W]
        
        # Cross-Attention: LQ特征调制Diff特征
        # Query来自LQ（询问"这个位置的内容需要多大不确定度"）
        # Key, Value来自Diff（提供"SR预测质量信息"）
        modulated_feat = self.cross_attn(
            query=lq_feat,
            key_value=diff_feat
        )  # [B, D, H, W]
        
        # 🔥 [新增] 执行平滑操作，消除方块效应
        modulated_feat = self.post_smooth(modulated_feat)

        # 输出调整量 🔥[注意]降低到0.1，减少对baseweight的影响
        adjustment = self.output_proj(modulated_feat)*0.1  # [B, C, H, W]
        
        if self.forward_count % 5000 == 1:
            self.logger.info(f"  Adjustment: min={adjustment.min():.4f}, max={adjustment.max():.4f}, mean={adjustment.mean():.4f}, std={adjustment.std():.4f}")
            
            # 检查adjustment符号分布
            pos_ratio = (adjustment > 0).float().mean()
            neg_ratio = (adjustment < 0).float().mean()
            self.logger.info(f"  Adjustment signs: positive={pos_ratio*100:.1f}%, negative={neg_ratio*100:.1f}%")
        
        # 残差连接
        final_weight = base_weight + adjustment

        
        # 约束范围
        final_weight = final_weight.clamp(min=self.min_noise, max=1.0)
        
        if self.forward_count % 5000 == 1:
            self.logger.info(f"  FinalWeight: min={final_weight.min():.4f}, max={final_weight.max():.4f}, mean={final_weight.mean():.4f}")
            
            # 检查饱和情况
            saturated_low = (final_weight == self.min_noise).float().mean()
            saturated_high = (final_weight == 1.0).float().mean()
            self.logger.info(f"  Saturation: low={saturated_low*100:.2f}%, high={saturated_high*100:.2f}%")
            
            if saturated_low + saturated_high > 0.10:
                self.logger.warning(f"  ⚠️  >10% pixels saturated!")
        
        return final_weight
