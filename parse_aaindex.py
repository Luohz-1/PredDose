""" Parsing the AAIndex database to use as input features in the neural nets """

from os.path import isfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def parse_raw_data(aaindex_fn="source_data/aaindex1"):   # 读取 AAindex1 文件，返回22列的数据框
    """ load and parse the raw aa index data """
    # read the aa index file

    with open(aaindex_fn) as f:
        lines = f.readlines()   # 读取文件所有内容

    # set up an empty dataframe (will append to it)
    data = pd.DataFrame([], columns=["accession number", "description", "A", "C", "D", "E", "F", "G", "H", "I", "K",
                                     "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"])    # 预先建立数据框的 header

    # the order of amino acids in the aaindex file
    line_1_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I"]
    line_2_order = ["L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    all_entries = []
    current_entry = {}
    reading_aa_props = 0
    for line in lines:  # 里面的多个 elif 语句，当遇到第一个判断正确的条件，后续的elif语句将会被跳过
        if line.startswith("//"):
            all_entries.append(current_entry)   # 每结束一个性质数据，便对数据进行保存和清空
            current_entry = {}
        elif line.startswith("H"):  # 索引编号标签
            current_entry.update({"accession number": line.split()[1]})
        elif line.startswith("D"):  # 描述信息
            current_entry.update({"description": " ".join(line.split()[1:])})
        elif line.startswith("I"):  # 信息行
            reading_aa_props = 1    # 不要第一行的信息
        elif reading_aa_props == 1:
            current_entry.update({k: v if v != "NA" else 0 for k, v in zip(line_1_order, line.split())})
            reading_aa_props = 2
        elif reading_aa_props == 2:
            current_entry.update({k: v if v != "NA" else 0 for k, v in zip(line_2_order, line.split())})
            reading_aa_props = 0

    data = pd.concat([data, pd.DataFrame(all_entries)], ignore_index=True)  # 以列表的形式嵌套了 566 个字典，每个字典表示一种特征数据，并加入到 data 数据框中

    return data


def my_pca(pcs, n_components):  # 使用 PCA 降维
    np.random.seed(7)
    pca = PCA(n_components=n_components)    # 初始化
    principal_components = pca.fit_transform(pcs)   # 计算
    print("Captured variance:", sum(pca.explained_variance_ratio_)) # 计算不同主成分贡献度
    return principal_components


def gen_pca_from_raw_data():

    n_components = 19   # 需要降低的维度直接指定，需要探究一下多少数值合适，在 jupyter 中实现
    out_fn = "../data/pca-{}.npy".format(n_components)
    if isfile(out_fn):
        raise FileExistsError("aaindex pca already exists: {}".format(out_fn))

    data = parse_raw_data() # 读取 AAindex1 文件

    aas = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    # standardize each aa feature onto unit scale
    aa_features = data.loc[:, aas].values.astype(np.float32)    # loc 通过行或列的标签来索引，iloc 通过位置索引
    # for standardization and PCA, we need it in [n_samples, n_features] format
    aa_features = aa_features.transpose()   # 转置
    # standardize
    aa_features = StandardScaler().fit_transform(aa_features)   # 归一化和标准化

    # pca
    pcs = my_pca(aa_features, n_components=n_components)
    np.save(out_fn, pcs)


def main():
    gen_pca_from_raw_data()


if __name__ == "__main__":
    main()
