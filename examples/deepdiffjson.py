import json
import sys
from deepdiff import DeepDiff

def serialize_deepdiff(obj):
    """自定义序列化函数，处理 DeepDiff 中的特殊类型"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, set):
        return list(obj)
    else:
        # 对于无法序列化的对象，转换为字符串
        return str(obj)

if len(sys.argv) != 3:
    print(f"用法: {sys.argv[0]} file1.json file2.json")
    sys.exit(1)

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data1 = json.load(f)
with open(sys.argv[2], 'r', encoding='utf-8') as f:
    data2 = json.load(f)

diff = DeepDiff(data1, data2, ignore_order=True)

# 使用 DeepDiff 的内置方法转换为可序列化的格式
try:
    # 优先使用 DeepDiff 的 to_json 方法
    if hasattr(diff, 'to_json'):
        print(diff.to_json(indent=2))
    else:
        # 备用方案：使用自定义序列化
        print(json.dumps(diff, default=serialize_deepdiff, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"序列化错误: {e}")
    # 最后的备用方案：直接打印 diff 对象
    print("差异结果:")
    print(diff)