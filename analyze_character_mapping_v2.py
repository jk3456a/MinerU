#!/usr/bin/env python3
import json
import re
from collections import defaultdict

def analyze_character_mapping(ocr_file, no_ocr_file):
    """分析OCR和No-OCR版本之间的字符映射差异"""
    
    def load_json_data(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    print("加载文件...")
    ocr_data = load_json_data(ocr_file)
    no_ocr_data = load_json_data(no_ocr_file)
    
    # 提取所有文本内容
    def extract_text_content(data):
        texts = []
        for key, value in data.items():
            if 'content' in key and isinstance(value, str) and len(value) > 0:
                texts.append((key, value))
        return texts
    
    ocr_texts = extract_text_content(ocr_data)
    no_ocr_texts = extract_text_content(no_ocr_data)
    
    print(f"OCR版本文本数量: {len(ocr_texts)}")
    print(f"No-OCR版本文本数量: {len(no_ocr_texts)}")
    
    # 创建键值映射
    ocr_dict = dict(ocr_texts)
    no_ocr_dict = dict(no_ocr_texts)
    
    # 分析字符映射问题
    character_mapping = defaultdict(list)
    encoding_issues = []
    
    # 对比相同的键
    common_keys = set(ocr_dict.keys()) & set(no_ocr_dict.keys())
    print(f"共同键数量: {len(common_keys)}")
    
    # 分析每个字符的映射
    for key in list(common_keys)[:100]:  # 只分析前100个
        ocr_text = ocr_dict[key]
        no_ocr_text = no_ocr_dict[key]
        
        if ocr_text != no_ocr_text:
            # 逐字符对比
            min_len = min(len(ocr_text), len(no_ocr_text))
            for i in range(min_len):
                ocr_char = ocr_text[i]
                no_ocr_char = no_ocr_text[i]
                
                if ocr_char != no_ocr_char:
                    character_mapping[no_ocr_char].append(ocr_char)
                    encoding_issues.append({
                        'key': key,
                        'position': i,
                        'ocr_char': ocr_char,
                        'no_ocr_char': no_ocr_char,
                        'ocr_code': ord(ocr_char),
                        'no_ocr_code': ord(no_ocr_char),
                        'ocr_text': ocr_text,
                        'no_ocr_text': no_ocr_text
                    })
    
    # 生成报告
    print("\n=== 字符映射分析报告 ===")
    
    print(f"\n1. 发现 {len(character_mapping)} 个不同的字符映射:")
    for no_ocr_char, ocr_chars in list(character_mapping.items())[:20]:
        if len(ocr_chars) > 0:
            most_common = max(set(ocr_chars), key=ocr_chars.count)
            print(f"  '{no_ocr_char}' (U+{ord(no_ocr_char):04X}) -> '{most_common}' (U+{ord(most_common):04X}) [出现{len(ocr_chars)}次]")
    
    print(f"\n2. 编码问题详情 (前10个):")
    for issue in encoding_issues[:10]:
        print(f"  键: {issue['key']}")
        print(f"  位置{issue['position']}: '{issue['no_ocr_char']}' (U+{issue['no_ocr_code']:04X}) -> '{issue['ocr_char']}' (U+{issue['ocr_code']:04X})")
        print(f"  OCR文本: {issue['ocr_text']}")
        print(f"  No-OCR文本: {issue['no_ocr_text']}")
        print()
    
    # 分析Unicode范围
    print("\n3. Unicode范围分析:")
    unicode_ranges = defaultdict(int)
    for issue in encoding_issues:
        no_ocr_code = issue['no_ocr_code']
        if 0x4E00 <= no_ocr_code <= 0x9FFF:  # 中文汉字
            unicode_ranges['CJK统一汉字'] += 1
        elif 0xFF00 <= no_ocr_code <= 0xFFEF:  # 全角字符
            unicode_ranges['全角字符'] += 1
        elif 0x3000 <= no_ocr_code <= 0x303F:  # 中文标点
            unicode_ranges['中文标点'] += 1
        elif 0x2000 <= no_ocr_code <= 0x206F:  # 一般标点
            unicode_ranges['一般标点'] += 1
        elif 0x0020 <= no_ocr_code <= 0x007F:  # ASCII
            unicode_ranges['ASCII'] += 1
        else:
            unicode_ranges[f'其他 (U+{no_ocr_code:04X})'] += 1
    
    for range_name, count in unicode_ranges.items():
        print(f"  {range_name}: {count}个字符")
    
    return character_mapping, encoding_issues

def create_character_fix_mapping(character_mapping):
    """创建字符修复映射表"""
    fix_mapping = {}
    
    for no_ocr_char, ocr_chars in character_mapping.items():
        if len(ocr_chars) > 0:
            # 选择最常见的映射
            most_common = max(set(ocr_chars), key=ocr_chars.count)
            fix_mapping[no_ocr_char] = most_common
    
    return fix_mapping

def create_enhanced_character_fix_function():
    """创建增强的字符修复函数"""
    fix_function = '''
def fix_character_encoding(text: str) -> str:
    """修复No-OCR版本中的字符编码问题"""
    
    # 基础字符映射
    char_mapping = {
        # 全角数字转半角
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
        
        # 全角字母转半角
        'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
        'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
        'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
        'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
        'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y', 'Ｚ': 'Z',
        'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e',
        'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j',
        'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o',
        'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't',
        'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y', 'ｚ': 'z',
        
        # 全角标点转半角
        '（': '(', '）': ')', '【': '[', '】': ']', '｛': '{', '｝': '}',
        '，': ',', '。': '.', '：': ':', '；': ';', '！': '!', '？': '?',
        '＂': '"', '＇': "'", '－': '-', '～': '~',
        
        # 特殊字符映射 (根据分析结果添加)
        '陨': 'I', '杂': 'S', '月': 'B', '晕': 'N', '怨': '9', '苑': '7', '愿': '8',
        '原': '0', '缘': '1', '园': '2', '员': '3', '猿': '5', '愿': '8',
        '悦': 'C', '陨': 'I', '孕': 'P', '圆': '0', '园': '2', '远': '6', '园': '2',
        '时': '时', '改': '政', '经': '济', '所': '出', '业': '版', '版': '社',
        '汉': '政', '经': '济', '业': '版', '版': '社',
    }
    
    # 应用字符映射
    for old_char, new_char in char_mapping.items():
        text = text.replace(old_char, new_char)
    
    return text
'''
    return fix_function

def main():
    ocr_file = "output_test_ocr/CZJJ00-60371752_flattened.json"
    no_ocr_file = "output_test_no_ocr/CZJJ00-60371752_flattened.json"
    
    print("开始分析字符映射问题...")
    character_mapping, encoding_issues = analyze_character_mapping(ocr_file, no_ocr_file)
    
    # 创建修复映射
    fix_mapping = create_character_fix_mapping(character_mapping)
    
    print(f"\n4. 建议的字符修复映射:")
    for no_ocr_char, ocr_char in list(fix_mapping.items())[:20]:
        print(f"  '{no_ocr_char}' -> '{ocr_char}'")
    
    # 保存修复映射到文件
    with open('character_fix_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(fix_mapping, f, ensure_ascii=False, indent=2)
    
    # 创建增强的修复函数
    fix_function = create_enhanced_character_fix_function()
    with open('character_fix_function.py', 'w', encoding='utf-8') as f:
        f.write(fix_function)
    
    print(f"\n修复映射已保存到 character_fix_mapping.json")
    print(f"修复函数已保存到 character_fix_function.py")

if __name__ == "__main__":
    main()
