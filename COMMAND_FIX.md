# ⚠️ 命令行参数修正说明

## 问题
在之前的文档中，使用了 `--opt`（双横杠），但实际应该使用 `-opt`（单横杠）。

## 正确的命令格式

### ✅ 正确
```bash
python test.py -opt options/test_stage2.yml
```

### ❌ 错误
```bash
python test.py --opt options/test_stage2.yml  # 会报错！
```

## 已修正的文件

所有文档和脚本已经修正：

### 修正的文档
- ✅ `DIFF_HEATMAP_QUICKSTART.md`
- ✅ `DIFF_HEATMAP_GUIDE.md`
- ✅ `DIFF_HEATMAP_IMPLEMENTATION_SUMMARY.md`
- ✅ `README_DIFF_HEATMAP.md`

### 修正的脚本
- ✅ `visualize_diff_heatmap.py` - 使用 `-opt` 参数
- ✅ `quick_inference_example.py` - 使用 `-opt` 参数

## 正确的使用示例

### 1. 测试脚本
```bash
python test.py -opt options/test_stage2.yml
```

### 2. 可视化脚本
```bash
python visualize_diff_heatmap.py \
    -opt options/test_upsr.yml \
    --input_path path/to/image.png \
    --output_dir results/heatmaps
```

### 3. 快速示例
```bash
python quick_inference_example.py \
    -opt options/test_upsr.yml \
    --input path/to/image.png \
    --output results/quick_test
```

## 参数说明

注意区分：
- `-opt` : 单横杠，用于配置文件路径参数（必需）
- `--input`, `--output`, `--gpu` 等：双横杠，用于其他选项参数

这是 argparse 的标准用法：
- 单横杠 `-` 通常用于短选项或必需参数
- 双横杠 `--` 通常用于长选项或可选参数

## 现在可以正常运行了！

```bash
# 运行测试
python test.py -opt options/test_stage2.yml

# 或者使用你自己的配置
python test.py -opt options/your_config.yml
```

所有文档已同步更新，可以放心使用！
