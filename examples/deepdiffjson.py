# #!/usr/bin/env python
# import json
# import sys
# from deepdiff import DeepDiff

# def flatten_keys(d, parent_key=""):
#     """递归提取 JSON 所有键路径"""
#     keys = []
#     if isinstance(d, dict):
#         for k, v in d.items():
#             full_key = f"{parent_key}.{k}" if parent_key else k
#             keys.extend(flatten_keys(v, full_key))
#     elif isinstance(d, list):
#         for i, v in enumerate(d):
#             full_key = f"{parent_key}[{i}]"
#             keys.extend(flatten_keys(v, full_key))
#     else:
#         keys.append(parent_key)
#     return keys

# if len(sys.argv) != 3:
#     print(f"用法: {sys.argv[0]} file1.json file2.json")
#     sys.exit(1)

# with open(sys.argv[1], 'r', encoding='utf-8') as f:
#     data1 = json.load(f)
# with open(sys.argv[2], 'r', encoding='utf-8') as f:
#     data2 = json.load(f)

# diff = DeepDiff(data1, data2, ignore_order=True)

# # 计算结构相似度
# keys1 = set(flatten_keys(data1))
# keys2 = set(flatten_keys(data2))
# common_keys = keys1 & keys2
# total_keys = keys1 | keys2
# structure_similarity = len(common_keys) / len(total_keys) * 100 if total_keys else 100

# # 计算值相似度（仅对共有键）
# same_values = 0
# for key in common_keys:
#     try:
#         v1 = eval(f"data1{''.join([f'[{repr(k)}]' if not k.startswith('[') else k for k in key.split('.')])}")
#         v2 = eval(f"data2{''.join([f'[{repr(k)}]' if not k.startswith('[') else k for k in key.split('.')])}")
#         if v1 == v2:
#             same_values += 1
#     except Exception:
#         pass
# value_similarity = same_values / len(common_keys) * 100 if common_keys else 100

# # 输出统计信息
# print(f"结构相似度: {structure_similarity:.2f}%")
# print(f"值相似度: {value_similarity:.2f}%")
# print(f"综合相似度: {(structure_similarity*0.5 + value_similarity*0.5):.2f}%")
#!/usr/bin/env python

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
