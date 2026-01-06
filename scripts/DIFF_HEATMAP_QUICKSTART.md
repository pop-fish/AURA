# Diff热力图功能使用说明

## 快速开始

### 方式1：测试时自动生成（最简单）

1. 编辑测试配置文件（如 `options/test_upsr_x4.yml`），添加：
```yaml
val:
  save_img: true
  save_diff_heatmap: true          # 启用Diff热力图
  save_uncertainty_heatmap: true   # 启用Uncertainty热力图
```

2. 运行测试：
```bash
python test.py -opt options/test_upsr_x4.yml
```

3. 查看结果（在 `results/visualization/dataset_name/` 目录下）：
   - `diff_heatmap/` - Diff热力图
   - `uncertainty_heatmap/` - Uncertainty热力图

### 方式2：单张图像快速测试

```bash
python quick_inference_example.py \
    -opt options/test_upsr_x4.yml \
    --input path/to/image.png \
    --output results/heatmaps
```

### 方式3：使用独立脚本

```bash
python visualize_diff_heatmap.py \
    -opt options/test_upsr_x4.yml \
    --input_path path/to/image.png \
    --output_dir results/diff_heatmaps
```

## 输出文件说明

```
results/
└── visualization/
    ├── {image_name}_sr.png           # SR输出
    ├── diff_heatmap/
    │   ├── {name}_diff_avg.png       # RGB平均
    │   ├── {name}_diff_chR.png       # R通道
    │   ├── {name}_diff_chG.png       # G通道  
    │   └── {name}_diff_chB.png       # B通道
    └── uncertainty_heatmap/
        ├── {name}_uncertainty_avg.png
        └── ...
```

## 核心概念

**Diff（差异图）**：
```
Diff = (SR_MSE预测 - 双三次插值) / 2
```
- 反映辅助SR网络相对于简单插值的改善程度
- 用于计算不确定度

**Uncertainty（不确定度）**：
```
Uncertainty = min_noise + (1 - min_noise) * |Diff|.clamp(0, un_max) / un_max
```
- 控制扩散过程添加的噪声量
- 高不确定度区域添加更多噪声

## 测试功能

运行测试脚本验证安装：
```bash
python test_diff_visualization.py
```

## 详细文档

完整说明请参考：[DIFF_HEATMAP_GUIDE.md](DIFF_HEATMAP_GUIDE.md)

## 文件清单

- `basicsr/utils/diff_visualization.py` - 核心可视化工具
- `visualize_diff_heatmap.py` - 完整可视化脚本
- `quick_inference_example.py` - 快速示例脚本
- `test_diff_visualization.py` - 功能测试脚本
- `DIFF_HEATMAP_GUIDE.md` - 详细使用指南
- `DIFF_HEATMAP_QUICKSTART.md` - 本文件

## 更新内容

已修改 `basicsr/models/upsr_real_model.py`：
- 在 `nondist_validation` 方法中添加了Diff热力图生成逻辑
- 当配置中启用 `save_diff_heatmap` 或 `save_uncertainty_heatmap` 时自动生成热力图
- 支持训练和测试阶段的热力图保存
