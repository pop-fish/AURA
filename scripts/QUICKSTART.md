# 快速使用指南

## 🚀 快速开始

### 方法 1: 使用测试脚本（推荐）

```bash
cd /root/project/UPSR_view

# 直接运行
python test_noise_visualization.py -opt options/test_stage2.yml

# 或使用 bash 脚本
chmod +x run_test.sh
./run_test.sh
```

### 方法 2: 使用原始测试脚本

```bash
cd /root/project/UPSR_view
python basicsr/test.py -opt options/test_stage2.yml
```

---

## 📂 输出位置

测试完成后，结果会保存在：

```
results/test_stage2_30k/
├── RealSRV3/              # 第一个验证集
│   ├── image1.png         # SR结果
│   ├── image2.png
│   ├── noise1/            # 不确定度图
│   │   ├── image1.png
│   │   └── image2.png
│   └── noise2/            # 加噪起点
│       ├── image1.png
│       └── image2.png
└── DIV2K/                 # 第二个验证集
    ├── ...
    ├── noise1/
    └── noise2/
```

---

## ⚙️ 配置说明

测试使用的配置文件：`options/test_stage2.yml`

### 关键配置项：

1. **模型权重**（必须设置）：
   ```yaml
   path:
     pretrain_network_g: experiments/stage1_content_aware_10k_best9k/models/net_g_9000.pth
   ```

2. **保存图片**（必须开启）：
   ```yaml
   val:
     save_img: true  # ⭐ 必须为 true
   ```

3. **数据集路径**：
   ```yaml
   datasets:
     val_1:
       name: RealSRV3
       dataroot_gt: /dataset/real2019/Test_HR_100
       dataroot_lq: /dataset/real2019/Test_LR_100_bicubic
   ```

---

## 🔍 查看结果

### 命令行查看

```bash
# 查看输出文件
ls -la results/test_stage2_30k/RealSRV3/
ls -la results/test_stage2_30k/RealSRV3/noise1/
ls -la results/test_stage2_30k/RealSRV3/noise2/

# 统计文件数量
echo "SR结果: $(ls results/test_stage2_30k/RealSRV3/*.png 2>/dev/null | wc -l) 张"
echo "不确定度图: $(ls results/test_stage2_30k/RealSRV3/noise1/*.png 2>/dev/null | wc -l) 张"
echo "加噪起点: $(ls results/test_stage2_30k/RealSRV3/noise2/*.png 2>/dev/null | wc -l) 张"
```

### 可视化对比

```bash
# 单张图像对比
python visualize_results.py \
    --result_dir results/test_stage2_30k/RealSRV3 \
    --image_name 0001 \
    --analyze

# 批量可视化
python visualize_results.py \
    --result_dir results/test_stage2_30k/RealSRV3 \
    --output_dir visualizations \
    --max_images 10
```

---

## 🐛 常见问题

### 问题 1: 模型权重文件不存在

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'experiments/xxx/models/net_g_xxx.pth'
```

**解决方法**：
1. 检查配置文件中的 `pretrain_network_g` 路径
2. 确保模型已经训练并保存
3. 使用绝对路径或相对于 `UPSR_view` 目录的路径

### 问题 2: 数据集路径不存在

**错误信息**：
```
AssertionError: dataroot_gt path does not exist
```

**解决方法**：
修改配置文件中的数据集路径，指向实际存在的目录

### 问题 3: 没有生成 noise1 或 noise2

**可能原因**：
- `save_img: false` - 需要设置为 `true`
- 没有启用内容感知映射 - 检查模型是否加载了 `uncertainty_mapper`

---

## 📊 输出说明

### noise1 - 不确定度图
- **含义**: 模型对每个像素的不确定程度
- **格式**: 灰度图，越亮表示越不确定
- **范围**: [0, 1]

### noise2 - 加噪起点  
- **含义**: 扩散过程的初始状态
- **格式**: RGB彩色图
- **范围**: [0, 1]

---

## 📝 详细文档

更多信息请查看：
- `NOISE_VISUALIZATION_README.md` - 完整技术文档
- `MODIFICATION_SUMMARY.md` - 修改总结

---

## ✅ 测试检查清单

运行测试前请确认：

- [ ] 配置文件存在：`options/test_stage2.yml`
- [ ] 模型权重存在：检查 `path.pretrain_network_g`
- [ ] 数据集路径正确：检查 `datasets.val_X.dataroot_*`
- [ ] 保存图片已启用：`val.save_img: true`
- [ ] 当前目录：`/root/project/UPSR_view`

全部确认后运行测试！
