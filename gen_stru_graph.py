""" Generating structure graphs for graph convolutional neural networks """
import os
from os.path import isfile
from enum import Enum, auto

import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from biopandas.pdb import PandasPdb

import constants
import utils


class GraphType(Enum):  # 生成图的几个类型
    LINEAR = auto()     # 线性图
    COMPLETE = auto()   # 全连接图
    DISCONNECTED = auto()   # 不连接图，只有点没有边
    DIST_THRESH = auto()    # 距离阈值图
    DIST_THRESH_SHUFFLED = auto()   # 打乱的距离阈值图


def save_graph(g, fn):  # 保存图
    """ Saves graph to file """
    nx.write_gexf(g, fn)


def load_graph(fn):
    """ Loads graph from file """
    g = nx.read_gexf(fn, node_type=int)
    return g


def shuffle_nodes(g, seed=7):   # 随机打乱图的节点，但是拓扑结构不变（边连接关系）
    """ Shuffles the nodes of the given graph and returns a copy of the shuffled graph """
    # get the list of nodes in this graph
    nodes = g.nodes()

    # create a permuted list of nodes
    np.random.seed(seed)
    nodes_shuffled = np.random.permutation(nodes)   # 打乱节点

    # create a dictionary mapping from old node label to new node label
    mapping = {n: ns for n, ns in zip(nodes, nodes_shuffled)}   # 建立映射关系，

    g_shuffled = nx.relabel_nodes(g, mapping, copy=True)    # 重新映射节点

    return g_shuffled


def linear_graph(num_residues): # 得到线性无向图
    """ Creates a linear graph where each each node is connected to its sequence neighbor in order """
    g = nx.Graph()  # 初始化无向图的类
    g.add_nodes_from(np.arange(0, num_residues))    # 以迭代器的形式为无向图增加点
    for i in range(num_residues-1):
        g.add_edge(i, i+1)  # 循环增加边，线性连接
    return g


def complete_graph(num_residues):
    """ Creates a graph where each node is connected to all other nodes"""
    g = nx.complete_graph(num_residues) # 全连接图
    return g


def disconnected_graph(num_residues):   # 不连接图，只有点，没有边
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, num_residues))
    return g


def dist_thresh_graph(dist_mtx, threshold): # 产生结构图
    """ Creates undirected graph based on a distance threshold """
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, dist_mtx.shape[0]))

    # loop through each residue
    for rn1 in range(len(dist_mtx)):
        # find all residues that are within threshold distance of current
        rns_within_threshold = np.where(dist_mtx[rn1] < threshold)[0]   # np.where 条件判断，返回邻接矩阵中，大于阈值点的索引

        # add edges from current residue to those that are within threshold
        for rn2 in rns_within_threshold:    # 叠代加入小于阈值的点
            # don't add self edges
            if rn1 != rn2:
                g.add_edge(rn1, rn2)    # 没有为边赋值
    return g


def ordered_adjacency_matrix(g):
    """ returns the adjacency matrix ordered by node label in increasing order as a numpy array """
    node_order = sorted(g.nodes())
    adj_mtx = nx.to_numpy_array(g, nodelist=node_order)
    return np.asarray(adj_mtx).astype(np.float32)


# 添加全局节点
def expand_adj(g):
    """
    original_adj: [490, 490]
    返回扩展后的邻接矩阵 [491, 491]
    """
    node_order = sorted(g.nodes())
    # 添加全局节点
    global_node = 490
    g.add_node(global_node)

    # 将全局节点与所有其他节点相连
    for node in node_order:
        g.add_edge(global_node, node)

    return g

def cbeta_distance_matrix(pdb_fn, start=0, end=None):

    ppdb = PandasPdb().read_pdb(pdb_fn) # 读取 pdb 文件

    # group by residue number
    grouped = ppdb.df["ATOM"].groupby("residue_number")     # groupby 对数据框数据按类别分类

    # a list of coords for the cbeta or calpha of each residue
    coords = []

    # loop through each residue and find the coordinates of cbeta
    for i, (residue_number, values) in enumerate(grouped):  # 按照 residue_number 残基顺序读取数据框
        # 循环提取每个残基中的对应 CB 或者 CA 的三维坐标位置
        # skip residues not in the range
        end_index = (len(grouped) if end is None else end)  # 最后一位残基的索引位置
        if i not in range(start, end_index):
            continue

        residue_group = grouped.get_group(residue_number)   # 得到 residue_number 的数据框

        atom_names = residue_group["atom_name"] # 得到原子代码
        if "CB" in atom_names.values:   # 优先选用 CB 原子，其次选用 CA 原子
            # print("Using CB...")
            atom_name = "CB"
        elif "CA" in atom_names.values:
            # print("Using CA...")
            atom_name = "CA"
        else:
            raise ValueError("Couldn't find CB or CA for residue {}".format(residue_number))

        # get the coordinates of cbeta (or calpha)
        coords.append(
            residue_group[residue_group["atom_name"] == atom_name][["x_coord", "y_coord", "z_coord"]].values[0])

    # stack the coords into a numpy array where each row has the x,y,z coords for a different residue
    coords = np.stack(coords)   # 将堆叠的数据转换成 ndarry

    # compute pairwise euclidean distance between all cbetas
    dist_mtx = cdist(coords, coords, metric="euclidean")    # 计算临接矩阵

    return dist_mtx


def gen_graph(graph_type, res_dist_mtx, dist_thresh=7, shuffle_seed=7, graph_save_dir=None, save=False):
    """ generate the specified structure graph using the specified residue distance matrix """
    if graph_type is GraphType.LINEAR:
        g = linear_graph(len(res_dist_mtx)) # 得到线性无向图
        save_fn = None if not save else os.path.join(graph_save_dir, "linear.graph")    # 保存线性图

    elif graph_type is GraphType.COMPLETE:
        g = complete_graph(len(res_dist_mtx))
        save_fn = None if not save else os.path.join(graph_save_dir, "complete.graph")

    elif graph_type is GraphType.DISCONNECTED:
        g = disconnected_graph(len(res_dist_mtx))
        save_fn = None if not save else os.path.join(graph_save_dir, "disconnected.graph")

    elif graph_type is GraphType.DIST_THRESH:
        g = dist_thresh_graph(res_dist_mtx, dist_thresh)    # 距离阈值图
        save_fn = None if not save else os.path.join(graph_save_dir, "dist_thresh_{}.graph".format(dist_thresh))

    elif graph_type is GraphType.DIST_THRESH_SHUFFLED:
        g = dist_thresh_graph(res_dist_mtx, dist_thresh)    # 距离阈值图
        g = shuffle_nodes(g, seed=shuffle_seed) # 将里面的点随机打乱，但连接性不变
        save_fn = None if not save else \
            os.path.join(graph_save_dir, "dist_thresh_{}_shuffled_r{}.graph".format(dist_thresh, shuffle_seed))

    else:
        raise ValueError("Graph type {} is not implemented".format(graph_type))

    if save:
        if isfile(save_fn):
            print("err: graph already exists: {}. to overwrite, delete the existing file first".format(save_fn))
        else:
            utils.mkdir(graph_save_dir)
            save_graph(g, save_fn)

    return g


def gen_all_graphs():
    """ generate all structure graphs for all datasets """
    thresholds = [11, 12, 13, 14, 15, 16, 17]
    shuffle_seed = 7
    for ds_name in constants.DATASETS.keys():
        cbeta_mtx = cbeta_distance_matrix(constants.DATASETS[ds_name]["pdb_fn"])    # 计算 pdb 文件的邻接矩阵
        for graph_type in GraphType:    # 循环生成无向图、全连接图、不联通图、距离阈值图、打乱的距离阈值图
            if graph_type in [GraphType.DIST_THRESH, GraphType.DIST_THRESH_SHUFFLED]:
                for threshold in thresholds:    # 循环得到不同阈值的图
                    gen_graph(graph_type, cbeta_mtx, dist_thresh=threshold, shuffle_seed=shuffle_seed,
                              graph_save_dir="data/cyp2c9/graphs", save=True)
            else:
                gen_graph(graph_type, cbeta_mtx, graph_save_dir="data/cyp2c9/graphs", save=True)


def main():
    gen_all_graphs()


if __name__ == "__main__":
    main()
