# 🎨 Diff热力图可视化功能

在UPSR推理过程中自动生成Diff（差异图）和Uncertainty（不确定度）的热力图。

## 🚀 快速开始（3步）

### 1️⃣ 修改配置文件
在测试配置文件中添加：
```yaml
val:
  save_img: true
  save_diff_heatmap: true          # 启用Diff热力图
  save_uncertainty_heatmap: true   # 启用Uncertainty热力图
```

### 2️⃣ 运行测试
```bash
python test.py -opt options/test_upsr.yml
```

### 3️⃣ 查看结果
热力图保存在：`results/visualization/{dataset_name}/diff_heatmap/` 和 `uncertainty_heatmap/`

---

## 📖 完整文档

- **[快速开始指南](DIFF_HEATMAP_QUICKSTART.md)** ⭐ 开始这里
- **[详细使用指南](DIFF_HEATMAP_GUIDE.md)** - 完整的API和使用方法
- **[实现总结](DIFF_HEATMAP_IMPLEMENTATION_SUMMARY.md)** - 技术细节和文件清单

## 🛠️ 工具脚本

### 独立可视化脚本
```bash
python visualize_diff_heatmap.py \
    -opt options/test_upsr.yml \
    --input_path path/to/image.png \
    --output_dir results/heatmaps
```

### 快速测试示例
```bash
python quick_inference_example.py \
    -opt options/test_upsr.yml \
    --input path/to/image.png \
    --output results/quick_test
```

### 功能测试
```bash
python test_diff_visualization.py
```

## 📊 输出示例

生成的热力图包括：
- ✅ Diff平均热力图（RGB通道平均）
- ✅ Diff各通道热力图（R、G、B）
- ✅ Uncertainty平均热力图
- ✅ Uncertainty各通道热力图
- ✅ 统计信息文本文件

## 🎨 热力图类型

| 类型 | Colormap | 含义 |
|------|----------|------|
| **Diff** | coolwarm | 蓝色=负值, 白色=零, 红色=正值 |
| **Uncertainty** | hot | 黑色=低不确定度, 黄白=高不确定度 |

## 💡 什么是Diff？

```
Diff = (SR_MSE预测 - 双三次插值上采样) / 2
```

- **SR_MSE预测**：辅助SR网络的输出
- **双三次插值**：LQ图像的简单上采样
- **Diff**：反映了SR网络相对于简单插值的改善程度

Diff用于计算Uncertainty（不确定度），进而控制扩散过程中添加的噪声量。

## 📁 新增文件

```
UPSR_view/
├── basicsr/utils/diff_visualization.py        # 核心工具
├── visualize_diff_heatmap.py                  # 完整脚本
├── quick_inference_example.py                 # 快速示例
├── test_diff_visualization.py                 # 测试脚本
├── DIFF_HEATMAP_GUIDE.md                      # 详细指南
├── DIFF_HEATMAP_QUICKSTART.md                 # 快速指南
├── DIFF_HEATMAP_IMPLEMENTATION_SUMMARY.md     # 实现总结
└── options/test_with_heatmap_example.yml      # 配置示例
```

## 🔧 配置示例

参考 [options/test_with_heatmap_example.yml](options/test_with_heatmap_example.yml) 获取完整配置。

## ❓ 常见问题

**Q: 如何只可视化单张图像？**
```bash
python quick_inference_example.py --opt config.yml --input image.png
```

**Q: 热力图在哪里？**
```
results/visualization/{dataset_name}/
├── diff_heatmap/
└── uncertainty_heatmap/
```

**Q: 如何修改colormap？**
编辑 `basicsr/utils/diff_visualization.py`，修改 `cmap` 参数。

**Q: 功能测试通过吗？**
运行 `python test_diff_visualization.py` 验证。

## 🎯 核心API

```python
from basicsr.utils.diff_visualization import visualize_model_internals

# 推理时启用中间结果保存
sr_output = model.sample_func(lq, save_intermediate=True)

# 一键生成所有热力图
stats = visualize_model_internals(model, 'output_dir', 'image_name')
```

## 📝 引用

如果这个功能对您的研究有帮助，欢迎引用UPSR论文。

## 📬 反馈

如有问题或建议，请提交Issue或Pull Request。

---

**开始使用**: 阅读 [DIFF_HEATMAP_QUICKSTART.md](DIFF_HEATMAP_QUICKSTART.md) 🚀
