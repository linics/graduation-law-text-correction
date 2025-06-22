#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd


def load_and_filter_terms(
        input_path: str,
        output_path: str,
        freq_threshold: int = 10000,
        top_n: int = None
):
    """
    加载 THUOCL_law.txt，清洗 freq 列，按频次排序，
    然后根据阈值或取前 N 条进行剪枝，最后保存到 CSV。

    :param input_path:  原始词典文件路径，制表符分隔，每行 term\tfreq
    :param output_path: 剪枝后词典保存路径（CSV 格式）
    :param freq_threshold: 只保留 freq >= freq_threshold 的行
    :param top_n:         若不为 None，则优先取前 top_n 条（忽略 freq_threshold）
    """
    # 1. 读取并清洗
    df = pd.read_csv(
        input_path,
        sep='\t',
        names=['term', 'freq'],
        header=None,
        encoding='utf-8',
        dtype={'term': str, 'freq': str}
    )
    # 将无法转换的 freq 设为 0
    df['freq'] = pd.to_numeric(df['freq'], errors='coerce').fillna(0).astype(int)

    # 2. 排序
    df = df.sort_values('freq', ascending=False).reset_index(drop=True)

    # 3. 剪枝
    if top_n is not None:
        df_pruned = df.head(top_n)
    else:
        df_pruned = df[df['freq'] >= freq_threshold]

    # 4. 保存
    df_pruned.to_csv(
        output_path,
        sep=',',
        index=False,
        encoding='utf-8',
        columns=['term', 'freq']
    )
    print(f"已保存 {len(df_pruned)} 条术语到 {output_path}")


if __name__ == '__main__':
    # 示例用法
    load_and_filter_terms(
        input_path='../dataset/THUOCL_law.txt',
        output_path='../dataset/THUOCL_law_pruned.csv',
        freq_threshold=10000,
        top_n=None
    )
