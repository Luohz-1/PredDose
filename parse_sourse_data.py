"""
处理原始 DMS 数据
"""

import pandas as pd
import re
from os.path import isfile


#
# 创建一个氨基酸的映射字典

amino_acid_map = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V', 'Ter': 'X'
}

# # 定义一个函数，将氨基酸的三字符表示转换为单字符表示
def convert_to_single_letter(hgvs_pro):
    # 是否要去除同一变异
    # 提取氨基酸的三字符表示
    match = re.match(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', hgvs_pro)   # 使用正则表达式判断，分别会捕获三个括号里面的与之匹配的内容，可以和使用match.groups获得
    if match:
        original_aa, position, mutated_aa = match.groups()
        return f"{amino_acid_map[original_aa]}{position}{amino_acid_map[mutated_aa]}"
    elif '=' in hgvs_pro:
        return None
    else:
        print(f'{hgvs_pro}cant precessing')
        raise

#
# # 定义一个函数，计算变异的数量
def count_mutations(hgvs_pro):
    # 如果存在逗号分隔的变异，计算数量
    if ',' in hgvs_pro:
        return len(hgvs_pro.split(','))
    return 1

# 定义评分与华法林计量映射关系
def map_AA(x):
    if 0 <= x < 1:
        return 1.5*x + 0.5
    elif 1 <= x <= 2:
        return 2*x
    else:
        print(f"err: Unexpected inputs: {x}")
        raise

def map_GA(x):
    if 0 <= x < 0.5:
        return 3*x + 0.5
    elif 0.5 <= x < 1.5:
        return 2*x + 1
    elif 1.5 <= x <= 2:
        return 4*x - 2
    else:
        print(f"err: Unexpected inputs: {x}")
        raise

def map_GG(x):
    if 0 <= x < 0.5:
        return 3.5*x + 1.25
    elif 0.5 <= x < 1.5:
        return 2*x + 2
    elif 1.5 <= x <= 2:
        return 4*x -1
    else:
        print(f"err: Unexpected inputs: {x}")
        raise

def main():

    source_fn = "source_data/urn_mavedb_00000095-a-1_scores.csv"
    out_fn = "data/cyp2c9/cyp2c9_dg.tsv"


    if isfile(out_fn):
        print("err: parsed cyp2c9 dataset already exists: {}".format(out_fn))
        return

    # load the source data
    data = pd.read_csv(source_fn, sep=",") # 此时的表头为 infer 模式，即推断表头内容

    data['variant'] = data['hgvs_pro'].apply(convert_to_single_letter)
    data = data.dropna(subset=['variant']) # 去除所有同义变异
    # 计算变异数量
    data ['num_mutations'] = data ['hgvs_pro'].apply(count_mutations)

    # 保留需要的列
    result_df = data[['variant', 'num_mutations', 'score']].copy()    # 得分是否要对其归一化
    result_df['score_clip_dp'] = result_df['score'].clip(lower=0, upper=1) * 2
    result_df['score_AA'] = result_df['score_clip_dp'].apply(map_AA)
    result_df['score_GA'] = result_df['score_clip_dp'].apply(map_GA)
    result_df['score_GG'] = result_df['score_clip_dp'].apply(map_GG)
    result_df.to_csv(out_fn, sep="\t", index=False)


if __name__ == "__main__":
    main()
