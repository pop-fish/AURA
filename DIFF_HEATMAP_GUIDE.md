# Diff热力图可视化指南

本指南介绍如何在UPSR推理过程中生成并可视化Diff（差异图）热力图。

## 📋 概述

**Diff（差异图）** 定义为：
```
Diff = (SR_MSE预测 - 双三次插值上采样) / 2
```

其中：
- `SR_MSE预测`：辅助SR网络（net_mse）的预测结果
- `双三次插值上采样`：LQ图像通过双三次插值放大的结果

Diff反映了辅助SR网络相对于简单插值的改善程度，也是计算不确定度（Uncertainty）的基础。

## 🚀 使用方法

### 方法1：通过测试配置启用（推荐）

在测试配置文件（如 `options/test_upsr.yml`）中添加以下选项：

```yaml
val:
  save_img: true
  save_diff_heatmap: true          # 启用Diff热力图保存
  save_uncertainty_heatmap: true   # 启用Uncertainty热力图保存
  # ... 其他配置 ...
```

然后运行测试：
```bash
python test.py -opt options/test_upsr.yml
```

### 方法2：使用独立脚本

使用提供的可视化脚本直接对单张图像生成热力图：

```bash
python visualize_diff_heatmap.py \
    -opt options/test_upsr.yml \
    --input_path path/to/input.png \
    --output_dir results/heatmaps \
    --gpu 0
```

### 方法3：在代码中直接调用

```python
from basicsr.utils.diff_visualization import visualize_model_internals
from basicsr.models import build_model

# 构建模型
model = build_model(opt)

# 准备输入图像（已归一化到[0, 1]的tensor）
lq_tensor = ...  # [1, 3, H, W]

# 执行推理（必须启用save_intermediate）
lq_normalized = (lq_tensor - 0.5) / 0.5
sr_output = model.sample_func(
    lq_normalized, 
    noise_repeat=False,
    save_intermediate=True  # ✅ 关键：必须设置为True
)

# 生成并保存热力图
from basicsr.utils.diff_visualization import visualize_model_internals

stats = visualize_model_internals(
    model, 
    save_dir='results/heatmaps',
    img_name='my_image'
)
```

## 📁 输出文件说明

启用热力图保存后，将在可视化目录中生成以下文件：

```
results/
└── visualization/
    └── dataset_name/
        ├── image_name_sr.png              # SR输出图像
        ├── diff_heatmap/                  # Diff热力图目录
        │   ├── image_name_diff_avg.png    # RGB通道平均的Diff热力图
        │   ├── image_name_diff_chR.png    # 红色通道Diff热力图
        │   ├── image_name_diff_chG.png    # 绿色通道Diff热力图
        │   └── image_name_diff_chB.png    # 蓝色通道Diff热力图
        ├── uncertainty_heatmap/           # Uncertainty热力图目录
        │   ├── image_name_uncertainty_avg.png
        │   ├── image_name_uncertainty_chR.png
        │   ├── image_name_uncertainty_chG.png
        │   └── image_name_uncertainty_chB.png
        ├── sr_mse/                        # 辅助SR网络预测结果
        │   └── image_name_sr_mse.png
        └── bicubic/                       # 双三次插值结果
            └── image_name_bicubic.png
```

## 🎨 热力图解读

### Diff热力图
- **颜色映射**：coolwarm（蓝色→白色→红色）
- **蓝色区域**：负值，表示SR_MSE预测低于双三次插值
- **红色区域**：正值，表示SR_MSE预测高于双三次插值
- **白色区域**：接近零，表示SR_MSE预测与双三次插值接近
- **值域范围**：通常在 `[-un_max, un_max]` 之间（默认 `[-1, 1]`）

### Uncertainty热力图
- **颜色映射**：hot（黑色→红色→黄色→白色）
- **黑色区域**：低不确定度（接近min_noise），扩散过程添加较少噪声
- **白色/黄色区域**：高不确定度（接近1.0），扩散过程添加较多噪声
- **值域范围**：`[min_noise, 1.0]`（默认 `[0.0, 1.0]`）

## 🔧 API参考

### visualize_diff_heatmap

```python
from basicsr.utils.diff_visualization import visualize_diff_heatmap

visualize_diff_heatmap(
    diff_tensor,          # [1, 3, H, W] Diff张量
    save_dir,             # 保存目录
    prefix='diff',        # 文件名前缀
    save_channels=True    # 是否分别保存各通道
)
```

### visualize_uncertainty_heatmap

```python
from basicsr.utils.diff_visualization import visualize_uncertainty_heatmap

visualize_uncertainty_heatmap(
    uncertainty_tensor,   # [1, 3, H, W] Uncertainty张量
    save_dir,             # 保存目录
    prefix='uncertainty', # 文件名前缀
    save_channels=True    # 是否分别保存各通道
)
```

### visualize_model_internals

```python
from basicsr.utils.diff_visualization import visualize_model_internals

stats = visualize_model_internals(
    model,                # UPSR模型实例
    save_dir,             # 保存目录
    img_name='image'      # 图像名称
)

# 返回统计信息字典
print(stats['diff']['mean'])        # Diff均值
print(stats['uncertainty']['mean']) # Uncertainty均值
```

## 📊 统计信息

每次生成热力图时，会同时保存统计信息文件 `{image_name}_stats.txt`：

```
=== Diff Statistics ===
Min: -0.123456
Max: 0.234567
Mean: 0.012345
Std: 0.045678

=== Uncertainty Statistics ===
Min: 0.000000
Max: 1.000000
Mean: 0.345678
Std: 0.123456
```

## 💡 使用技巧

1. **分析不确定度分布**：观察Uncertainty热力图可以了解模型在哪些区域对预测更有信心（黑色区域）或更不确定（黄白色区域）

2. **对比各通道**：RGB三个通道的Diff可能有所不同，分别查看可以发现通道特异性的模式

3. **调整热力图参数**：如果默认颜色映射不够清晰，可以修改 `diff_visualization.py` 中的 `cmap` 参数：
   - Diff: `'coolwarm'`, `'seismic'`, `'RdBu'`, `'bwr'`
   - Uncertainty: `'hot'`, `'viridis'`, `'plasma'`, `'inferno'`

4. **内存优化**：对于大图像，建议使用chop模式分块处理

## 🔍 常见问题

**Q: 为什么热力图全是白色/单一颜色？**
A: 可能是值域范围太小。检查统计信息中的Min/Max值，考虑调整颜色映射范围。

**Q: 如何同时保存原始数据？**
A: 在 `visualize_diff_heatmap.py` 中已包含保存 `.npy` 文件的代码，可以用于进一步分析：
```python
diff_data = np.load('results/heatmaps/image_name_diff.npy')
```

**Q: 训练过程中能否生成热力图？**
A: 可以，只需在训练配置的 `val` 部分启用 `save_diff_heatmap: true`。但注意这会显著增加验证时间。

**Q: 热力图对GPU内存有额外要求吗？**
A: 热力图生成在CPU上进行，对GPU内存无额外要求。

## 📝 配置示例

完整的测试配置示例：

```yaml
name: upsr_test_x4
model_type: UPSRRealModel
scale: 4
num_gpu: 1
manual_seed: 10

# 数据集配置
datasets:
  val_1:
    name: Set5
    type: PairedImageDataset
    dataroot_gt: datasets/Set5/HR
    dataroot_lq: datasets/Set5/LR_bicubic/X4
    io_backend:
      type: disk

# 网络配置
network_g:
  type: SwinIR
  # ... 网络参数 ...

network_mse:
  type: SwinIR
  ckpt:
    path: experiments/pretrained_models/net_mse.pth
    param_key_mse: params_ema

# 扩散模型配置
diffusion:
  type: GaussianDiffusion
  un: 1.0              # Diff的最大值
  min_noise: 0.0       # 最小不确定度
  # ... 其他扩散参数 ...

# 验证配置
val:
  save_img: true
  save_diff_heatmap: true          # ✅ 启用Diff热力图
  save_uncertainty_heatmap: true   # ✅ 启用Uncertainty热力图
  suffix: ~
  chop_size: 256
  chop_stride: 224
  chop_bs: 1
  noise_repeat: false
  
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
    ssim:
      type: calculate_ssim
      crop_border: 4

# 路径配置
path:
  pretrain_network_g: experiments/pretrained_models/upsr_x4.pth
  param_key_g: params_ema
  strict_load_g: true
  visualization: results/visualization
```

## 📚 相关文件

- `basicsr/utils/diff_visualization.py` - 核心可视化工具模块
- `visualize_diff_heatmap.py` - 独立可视化脚本
- `basicsr/models/upsr_real_model.py` - 模型实现（包含Diff计算）
- `basicsr/models/uncertainty_mapping.py` - 不确定度映射模块

## 🎯 下一步

- 尝试可视化不同数据集上的Diff分布
- 对比固定映射和可学习映射的Uncertainty热力图差异
- 分析Diff热力图与最终SR质量的关系

## 📮 反馈与建议

如有问题或建议，请在项目仓库提交Issue。
