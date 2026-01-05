# 修改总结

## 📋 修改内容

已成功为 UPSR 模型添加测试时输出中间结果的功能，可以保存：
1. **不确定度图（Uncertainty Map）** → `results/[dataset]/noise1/`
2. **扩散起点（Noisy Start）** → `results/[dataset]/noise2/`

---

## 📁 修改的文件

### 1. 核心模型文件
- **`basicsr/models/upsr_real_model.py`**
  - ✅ `sample_func()`: 添加 `save_intermediate` 参数，保存中间结果
  - ✅ `test()`: 传递 `save_intermediate` 参数
  - ✅ `nondist_validation()`: 转换并保存不确定度图和加噪起点图

### 2. 扩散模型文件
- **`models/gaussian_diffusion.py`**
  - ✅ `ddim_sample_loop()`: 添加 `return_noisy_start` 参数，返回加噪起点
  - ✅ `ddim_sample_loop_progressive()`: 保存并传递加噪起点

### 3. 新增文件
- ✅ **`test_noise_visualization.py`**: 测试脚本，验证噪声可视化功能
- ✅ **`visualize_results.py`**: 可视化脚本，对比显示结果
- ✅ **`NOISE_VISUALIZATION_README.md`**: 详细说明文档

---

## 🚀 使用方法

### 步骤 1: 配置文件

确保测试配置文件中设置：
```yaml
val:
  save_img: true  # ⭐ 必须启用
```

### 步骤 2: 运行测试

```bash
cd UPSR_view

# 方法 1: 使用新测试脚本（推荐）
python test_noise_visualization.py -opt options/test_upsr_real.yml

# 方法 2: 使用原始测试脚本
python basicsr/test.py -opt options/test_upsr_real.yml
```

### 步骤 3: 查看结果

测试完成后，输出目录结构：
## 📂 输出目录结构

```
results/[experiment_name]/
└── [dataset_name]/
    ├── image1.png          # SR最终结果
    ├── noise1/             # 不确定度图（灰度）
    │   └── image1.png
    ├── noise2/             # 扩散起点（RGB）
    │   └── image1.png
    ├── sr_mse/             # 辅助SR网络预测结果
    │   └── image1.png
    └── bicubic/            # 双三次插值放大结果
        └── image1.png
```

### 步骤 4: 可视化对比（可选）

```bash
# 单张图像对比
python visualize_results.py \
    --result_dir results/UPSR_x4/Set5 \
    --image_name baby \
    --analyze

# 批量可视化（生成对比图）
python visualize_results.py \
    --result_dir results/UPSR_x4/Set5 \
    --output_dir visualizations \
    --max_images 10
```

---

## 🔍 输出说明

### noise1 - 不确定度图
- **含义**: 内容感知映射生成的不确定度权重
- **格式**: 灰度图（RGB三通道相同）
- **范围**: [0, 1]，越亮表示不确定度越高
- **用途**: 
  - 分析模型对不同区域的信心
  - 识别容易出错的区域
  - 验证内容感知映射效果

### noise2 - 加噪起点
- **含义**: 扩散过程的初始状态（最高噪声水平）
- **格式**: RGB彩色图
- **范围**: [0, 1]
- **用途**:
  - 可视化扩散起点
  - 理解噪声注入过程
  - 调试采样过程

### sr_mse - 辅助SR网络预测
- **含义**: MSE网络 g(y_0) 的预测结果
- **格式**: RGB彩色图
- **范围**: [0, 1]
- **用途**:
  - 对比MSE预测 vs 最终扩散结果
  - 分析MSE网络的性能
  - 理解扩散模型的改进效果

### bicubic - 双三次插值放大
- **含义**: 简单的双三次插值上采样结果
- **格式**: RGB彩色图
- **范围**: [0, 1]
- **用途**:
  - 提供baseline对比
  - 分析相对于简单插值的改进
  - 理解不确定度的计算基础

---

## ⚙️ 工作原理

```
测试流程：
nondist_validation()
  └─> test(save_intermediate=True)
      └─> sample_func(save_intermediate=True)
          ├─> 计算不确定度图 un
          ├─> ddim_sample_loop(return_noisy_start=True)
          │   └─> 返回 (结果, 加噪起点)
          ├─> 保存 self.uncertainty_map = un
          └─> 保存 self.noisy_start = noisy_start

保存流程：
nondist_validation()
  ├─> 转换 uncertainty_map → 灰度图
  ├─> 转换 noisy_start → RGB图
  └─> 保存到 noise1/ 和 noise2/ 目录
```

---

## ✅ 功能验证

运行测试脚本验证：
```bash
python test_noise_visualization.py -opt your_config.yml
```

脚本会自动：
- ✅ 显示配置信息
- ✅ 显示处理进度
- ✅ 统计输出文件数量
- ✅ 验证文件是否正确生成

---

## 📊 代码变更统计

| 文件 | 修改行数 | 新增功能 |
|------|---------|---------|
| `upsr_real_model.py` | ~80 行 | 中间结果保存、可视化转换 |
| `gaussian_diffusion.py` | ~20 行 | 返回加噪起点 |
| `test_noise_visualization.py` | 150 行 | 新增测试脚本 |
| `visualize_results.py` | 280 行 | 新增可视化工具 |
| `NOISE_VISUALIZATION_README.md` | 500 行 | 详细文档 |

---

## 🔧 注意事项

1. **内存消耗**: 保存中间结果会增加内存使用
2. **存储空间**: 每张输入图生成3张输出，需足够存储空间
3. **仅测试启用**: 功能只在 `save_img=True` 时激活
4. **Chop模式**: 使用图像分块时，中间结果可能不完整

---

## 📚 参考文档

详细使用说明请查看：
- **`NOISE_VISUALIZATION_README.md`** - 完整功能文档
- **`test_noise_visualization.py`** - 测试脚本源码
- **`visualize_results.py`** - 可视化工具源码

---

## 🐛 故障排查

如果遇到问题：

1. **没有生成 noise1/noise2 目录**
   - 检查 `save_img: true` 是否设置
   - 检查是否启用了可学习映射

2. **图像全黑或全白**
   - 检查不确定度范围是否正常
   - 验证数值归一化是否正确

3. **报错：找不到文件**
   - 确保输出目录有写权限
   - 检查路径配置是否正确

---

## 🎉 完成！

所有修改已完成，可以开始测试了！

运行命令：
```bash
cd /root/project/UPSR_view
python test_noise_visualization.py -opt your_test_config.yml
```
