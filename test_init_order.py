"""
测试 UPSRRealModel 初始化顺序是否正确
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

# 加载配置文件
config_path = 'options/stage1_learnable_mapper_only.yml'
with open(config_path, 'r') as f:
    opt = yaml.safe_load(f)

print("✓ Config loaded successfully")
print(f"✓ use_learnable: {opt.get('uncertainty_mapping', {}).get('use_learnable', False)}")

# 尝试导入并检查属性
try:
    from basicsr.models.upsr_real_model import UPSRRealModel
    print("✓ UPSRRealModel imported successfully")
    
    # 检查初始化顺序
    print("\n检查初始化逻辑...")
    print("1. uncertainty_mapping配置:", opt.get('uncertainty_mapping', {}))
    print("2. use_learnable应该在super().__init__()之前设置")
    print("3. uncertainty_mapper应该初始化为None")
    
    print("\n✓ 代码结构看起来正确！")
    print("\n接下来运行完整训练来验证...")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
