# 噪声可视化功能说明

## 修改概述

此次修改为 UPSR 模型添加了在测试/推理阶段输出中间结果的功能，具体包括：

1. **不确定度图（Uncertainty Map）**：内容感知映射生成的不确定度权重可视化
2. **加噪起点（Noisy Start）**：扩散过程的初始加噪图像

这些中间结果对于理解模型的工作机制、调试和分析非常有用。

---

## 修改的文件

### 1. `basicsr/models/upsr_real_model.py`

#### 修改点 1: `sample_func` 方法
- **新增参数**: `save_intermediate=False`
- **功能**: 当 `save_intermediate=True` 时，保存不确定度图和加噪起点
- **存储位置**: 
  - `self.uncertainty_map`: 不确定度图 tensor
  - `self.noisy_start`: 加噪起点 tensor

```python
def sample_func(self, y0, noise_repeat=False, save_intermediate=False):
    # ... 原有代码 ...
    
    # 保存中间结果用于可视化
    if save_intermediate:
        self.uncertainty_map = un
        self.noisy_start = noisy_start
    
    return results.clamp_(-1.0, 1.0)
```

#### 修改点 2: `test` 方法
- **新增参数**: `save_intermediate=False`
- **功能**: 将 `save_intermediate` 参数传递给 `sample_func`

```python
def test(self, save_intermediate=False):
    def _process_per_image(im_lq_tensor):
        # ... 调用 sample_func 时传递 save_intermediate
        im_sr_tensor = self.sample_func(
            (im_lq_tensor - 0.5) / 0.5,
            noise_repeat=self.opt['val']['noise_repeat'],
            save_intermediate=save_intermediate,
        )
```

#### 修改点 3: `nondist_validation` 方法
- **功能**: 
  1. 调用 `self.test(save_intermediate=save_img)` 启用中间结果保存
  2. 将不确定度图和加噪起点转换为可视化图像
  3. 保存到指定目录

- **输出目录结构**:
```
results/
└── [dataset_name]/
    ├── image1_suffix.png          # SR结果
    ├── image2_suffix.png
    ├── noise1/                    # 不确定度图
    │   ├── image1_suffix.png
    │   └── image2_suffix.png
    └── noise2/                    # 加噪起点
        ├── image1_suffix.png
        └── image2_suffix.png
```

- **可视化处理**:
  - 不确定度图：取RGB三通道平均，归一化到 [0, 1]，转换为灰度图
  - 加噪起点：从 [-1, 1] 范围转换到 [0, 1]，保持RGB三通道

### 2. `models/gaussian_diffusion.py`

#### 修改点 1: `ddim_sample_loop` 方法
- **新增参数**: `return_noisy_start=False`
- **功能**: 控制是否返回加噪起点
- **返回值**: 
  - 原来: `result`
  - 现在: `result, noisy_start_decoded` 或 `result, None`

```python
def ddim_sample_loop(self, ..., return_noisy_start=False):
    # ... 采样过程 ...
    
    result = nn.PixelShuffle(self.sf)(final)
    if return_noisy_start and noisy_start is not None:
        noisy_start_decoded = nn.PixelShuffle(self.sf)(noisy_start)
        return result, noisy_start_decoded
    else:
        return result, None
```

#### 修改点 2: `ddim_sample_loop_progressive` 方法
- **新增参数**: `return_noisy_start=False`
- **功能**: 在第一次迭代时保存加噪起点 `x_sample`
- **传递**: 在每次 yield 的 `out` 字典中添加 `"noisy_start"` 键

```python
def ddim_sample_loop_progressive(self, ..., return_noisy_start=False):
    # ... 初始化 ...
    
    x_sample = self.prior_sample(y, y_hat, un, noise)
    noisy_start = x_sample.clone() if return_noisy_start else None
    
    for i in indices:
        # ... 采样步骤 ...
        if return_noisy_start:
            out["noisy_start"] = noisy_start
        yield out
```

---

## 使用方法

### 1. 配置文件设置

确保你的测试配置文件（如 `options/test_upsr_real.yml`）中包含：

```yaml
# 验证设置
val:
  save_img: true  # ⭐ 必须设置为 true
  suffix: ~  # 或者设置自定义后缀
  # ... 其他配置 ...

# 路径设置
path:
  visualization: results/your_experiment_name  # 输出目录

# 不确定度映射设置（如果使用内容感知映射）
uncertainty_mapping:
  use_learnable: true
  type: content_aware  # 或 'spatial' 或 'mlp'
  ckpt_path: path/to/uncertainty_mapper.pth  # 如果有预训练权重
```

### 2. 运行测试

#### 方法 A: 使用原始测试脚本
```bash
cd UPSR_view
python basicsr/test.py -opt options/test_upsr_real.yml
```

#### 方法 B: 使用新的测试脚本（推荐）
```bash
cd UPSR_view
python test_noise_visualization.py -opt options/test_upsr_real.yml
```

新测试脚本会：
- 显示配置信息
- 显示处理进度
- 自动检查输出文件
- 统计生成的图片数量

### 3. 查看结果

测试完成后，检查输出目录：

```bash
ls -la results/your_experiment_name/[dataset_name]/
ls -la results/your_experiment_name/[dataset_name]/noise1/
ls -la results/your_experiment_name/[dataset_name]/noise2/
```

---

## 输出说明

### 不确定度图（noise1）

- **含义**: 表示模型对每个像素位置的不确定程度
- **范围**: [0, 1]，越亮（接近1）表示不确定度越高
- **格式**: 灰度图（RGB三通道相同）
- **用途**:
  - 分析模型对不同区域的信心
  - 识别容易出错的区域
  - 验证内容感知映射的效果

### 加噪起点（noise2）

- **含义**: 扩散过程的初始状态（最高噪声水平）
- **范围**: [0, 1]（从 [-1, 1] 转换后）
- **格式**: RGB彩色图
- **用途**:
  - 可视化扩散起点
  - 理解噪声注入过程
  - 调试采样过程

---

## 代码流程

```
nondist_validation()
    ├─> feed_data()                    # 加载数据
    ├─> test(save_intermediate=True)   # 测试
    │   └─> sample_func(save_intermediate=True)
    │       ├─> 计算不确定度图 un
    │       ├─> ddim_sample_loop(return_noisy_start=True)
    │       │   └─> ddim_sample_loop_progressive(return_noisy_start=True)
    │       │       ├─> 保存 x_sample 为 noisy_start
    │       │       └─> 执行扩散采样
    │       ├─> self.uncertainty_map = un
    │       └─> self.noisy_start = noisy_start
    │
    ├─> 转换为可视化图像
    │   ├─> uncertainty_imgs (灰度图)
    │   └─> noisy_start_imgs (RGB图)
    │
    └─> 保存图片
        ├─> imwrite(sr_img, save_path)
        ├─> imwrite(uncertainty_imgs, noise1_path)
        └─> imwrite(noisy_start_imgs, noise2_path)
```

---

## 注意事项

1. **内存消耗**: 保存中间结果会增加内存消耗，特别是处理高分辨率图像时
   
2. **存储空间**: 每张输入图会生成3张输出（SR + uncertainty + noisy），确保有足够存储空间

3. **仅在测试时启用**: 此功能只在 `save_img=True` 且非训练模式下激活

4. **可学习映射**: 如果没有启用可学习的不确定度映射，`noise1` 目录可能为空或包含固定映射结果

5. **Chop模式**: 如果使用图像分块处理（chop），中间结果可能只保存最后一块的结果

---

## 调试建议

如果输出的中间结果不符合预期：

1. **检查配置**:
   ```bash
   grep -A 5 "save_img" your_config.yml
   ```

2. **检查模型属性**:
   在 `nondist_validation` 中添加打印：
   ```python
   print(f"Has uncertainty_map: {hasattr(self, 'uncertainty_map')}")
   print(f"Has noisy_start: {hasattr(self, 'noisy_start')}")
   ```

3. **检查文件权限**:
   确保输出目录有写权限

4. **验证图像范围**:
   ```python
   print(f"Uncertainty range: [{self.uncertainty_map.min()}, {self.uncertainty_map.max()}]")
   print(f"Noisy start range: [{self.noisy_start.min()}, {self.noisy_start.max()}]")
   ```

---

## 示例输出

假设你测试了 Set5 数据集，输出目录结构如下：

```
results/UPSR_ContentAware_x4/
└── Set5/
    ├── baby_UPSR_ContentAware_x4.png       # SR结果
    ├── bird_UPSR_ContentAware_x4.png
    ├── butterfly_UPSR_ContentAware_x4.png
    ├── head_UPSR_ContentAware_x4.png
    ├── woman_UPSR_ContentAware_x4.png
    ├── noise1/                              # 不确定度图
    │   ├── baby_UPSR_ContentAware_x4.png
    │   ├── bird_UPSR_ContentAware_x4.png
    │   ├── butterfly_UPSR_ContentAware_x4.png
    │   ├── head_UPSR_ContentAware_x4.png
    │   └── woman_UPSR_ContentAware_x4.png
    └── noise2/                              # 加噪起点
        ├── baby_UPSR_ContentAware_x4.png
        ├── bird_UPSR_ContentAware_x4.png
        ├── butterfly_UPSR_ContentAware_x4.png
        ├── head_UPSR_ContentAware_x4.png
        └── woman_UPSR_ContentAware_x4.png
```

---

## 扩展建议

如果你需要更多可视化功能，可以考虑：

1. **保存更多中间步骤**: 修改 `ddim_sample_loop_progressive` 保存每个时间步的结果
2. **热力图可视化**: 使用 matplotlib 的 colormap 渲染不确定度图
3. **差异图**: 保存 `diff = (y_hat - y_bicubic) / 2` 的可视化
4. **注意力图**: 如果使用 ContentAwareSpatialUncertaintyMapping，可视化注意力权重

---

如有问题，请检查上述配置和代码流程。
