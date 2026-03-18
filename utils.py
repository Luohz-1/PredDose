""" general utility functions used throughput codebase """

import os
from os.path import join, isfile, isdir, basename
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import constants


__version__ = '1.0.0'

def mkdir(d):   # 创建 log 目录
    """ creates given dir if it does not already exist """
    if not isdir(d):
        os.makedirs(d)



def load_dataset(ds_name=None, ds_fn=None): # 以 pandas 格式加载数据集
    """ load a dataset as pandas dataframe """
    if ds_name is None and ds_fn is None:
        raise ValueError("must provide either ds_name or ds_fn to load a dataset")

    if ds_fn is None:
        ds_fn = constants.DATASETS[ds_name]["ds_fn"]

    if not isfile(ds_fn):
        raise FileNotFoundError("can't load dataset, file doesn't exist: {}".format(ds_fn))

    ds = pd.read_csv(ds_fn, sep="\t")
    return ds




# 数据迭代器
def create_dataloader(dt, lab, vk=None, batch_size=64, shuffle=True):
    # 将节点特征和标签组合成TensorDataset
    if vk is None:
        dataset = TensorDataset(dt, lab)
    else:
        dataset = TensorDataset(dt, vk, lab)
    # 创建DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

