#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from typing import List, Dict, Optional

# —— 配置区 —— #
# 本项目的根目录即为当前文件所在目录
BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "dataset")
DICT_CSV    = os.path.join(DATA_DIR, "THUOCL_law_pruned.csv")
# —— end 配置 —— #

def load_terms(csv_path: str = DICT_CSV) -> List[str]:
    """
    从 CSV 加载术语列表（第一列）。假设无表头，列格式为 term,freq
    """
    terms = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            term = row[0].strip()
            if term:
                terms.append(term)
    return terms

def is_edit_distance_leq_one(s: str, t: str) -> bool:
    """
    判断两个字符串 s, t 的 Levenshtein 距离是否 ≤1。
    只处理插入、删除、替换三种操作。
    """
    if s == t:
        return True
    ls, lt = len(s), len(t)
    # 长度差 >1，必然 >1 次编辑
    if abs(ls - lt) > 1:
        return False
    # 确保 s 较短
    if ls > lt:
        return is_edit_distance_leq_one(t, s)
    # 此时 ls <= lt
    # 如果等长，只需检查 ≤1 个替换
    if ls == lt:
        diff = sum(1 for cs, ct in zip(s, t) if cs != ct)
        return diff <= 1
    # 如果 lt = ls + 1，只需检查插入/删除一次
    # 双指针扫描
    i = j = diff = 0
    while i < ls and j < lt:
        if s[i] == t[j]:
            i += 1
            j += 1
        else:
            diff += 1
            if diff > 1:
                return False
            j += 1
    return True  # 末尾多出的那一个字符也算一次编辑

def filter_candidates(
    candidates: List[str],
    terms: Optional[List[str]] = None
) -> List[Dict]:
    """
    对 candidates 列表中的每个词做合法性校验。
    返回每个词的匹配结果字典：
      { 'candidate': str,
        'flag': bool,        # 是否匹配到合法术语
        'matched_term': str,  # 命中的合法术语（或 None）
      }
    """
    if terms is None:
        terms = load_terms()
    results = []
    for cand in candidates:
        matched = None
        # 1) 精确匹配
        if cand in terms:
            matched = cand
        else:
            # 2) 近似匹配：编辑距离 ≤1
            for term in terms:
                if is_edit_distance_leq_one(cand, term):
                    matched = term
                    break
        results.append({
            'candidate': cand,
            'flag': bool(matched),
            'matched_term': matched
        })
    return results

# —— 测试示例 —— #
if __name__ == "__main__":
    # 假设你在主流程里得到了下面这个候选列表
    sample = ["合同", "判决", "违约金", "证谑"]  # “证谑” 应近似命中 “证据”
    terms = load_terms()
    out = filter_candidates(sample, terms)
    for item in out:
        print(item)
    # 运行效果示例：
    # {'candidate': '合同',  'flag': True, 'matched_term': '合同'}
    # {'candidate': '判决',  'flag': True, 'matched_term': '判决'}
    # {'candidate': '违约金','flag': False,'matched_term': None}
    # {'candidate': '证谑',  'flag': True, 'matched_term': '证据'}
