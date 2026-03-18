"""
实现umap
"""

# this jupyter notebook is running inside of the "notebooks" directory
# for relative paths to work properly, we need to set the current working directory to the root of the project
# for imports to work properly, we need to add the code folder to the system path
# import os
# from os.path import abspath, join, isdir, isfile
# import sys
# if not isdir("notebooks"):
#     # if there's a "notebooks" directory in the cwd, we've already set the cwd so no need to do it again
#     os.chdir("../bin")
# os.getcwd()
# module_path = abspath("bin")
# if module_path not in sys.path:
#     sys.path.append(module_path)
# os.getcwd()
#
# import torch
# import build_model
# import utils
# import umap
# import encode as enc
# import constants
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
#
# # 加载数据并编码
# ds = utils.load_dataset(ds_fn='../data/cyp2c9/cyp2c9_dg.tsv')  # 以 pandas 格式加载数据集
#
# wt_aa = constants.DATASETS["cyp2c9"]["wt_aa"]  # 野生氨基酸序列
# wt_ofs = constants.DATASETS["cyp2c9"]["wt_ofs"]
#
# # create the dataset dictionary, containing encoded data, scores, etc,
# data = {}
# vk = ['score_GG', 'score_GA', 'score_AA']
# data["scores"] = []
#
# data["ds"] = ds
# data["variants"] = ds['variant'].tolist()
# for i in vk:
#     data["scores"].append(ds[f"{i}"].to_numpy(dtype='float32'))
# data["scores"] = np.array(np.array(data["scores"]).flatten().tolist(), dtype=np.float32)
# data["encoded_data"] = enc.encode(encoding='one_hot,aa_index', graph_fn="",
#                                   variants=data["variants"],
#                                   wt_aa=wt_aa, wt_offset=wt_ofs)
#
# # 加载模型
# loaded_model = torch.load('./output/cnn_3X128_full_model.pth', weights_only=False)
# loaded_model.eval()
#
#
# def extract_features(loaded_model, dataloader, device):
#     features_dict = {'layer1': [], 'layer2': [], 'layer3': [], 'layer_cat':[]}
#     labels = []
#
#     with torch.no_grad():
#         for x, y in dataloader:
#             x = x.to(device)  # (batch_size, 490, 40)
#             _, features = loaded_model(x, return_features=True)
#             for layer in features:
#                 # 展平特征 (batch_size, channels, length) -> (batch_size, -1)
#                 flattened = features[layer].view(x.size(0), -1).cpu().numpy()
#                 features_dict[layer].append(flattened)
#             labels.append(y.cpu().numpy())
#
#     # 合并所有batch
#     for layer in features_dict:
#         features_dict[layer] = np.concatenate(features_dict[layer], axis=0)
#     labels = np.concatenate(labels)
#
#     return features_dict, labels
#
#
# def apply_umap(features_dict):
#     umap_results = {}
#     for layer, feats in features_dict.items():
#         # UMAP降维
#         reducer = umap.UMAP(n_components=2, random_state=42)
#         umap_results[layer] = reducer.fit_transform(feats)
#     return umap_results
#
# def plot_umap_1(umap_results, labels, title_suffix=""):
#     fig, ax = plt.subplots(1, 1, figsize=(7, 7))
#
#     # for (layer, emb), ax in zip(umap_results.items(), axes):
#     divnorm = colors.TwoSlopeNorm(vmin=labels.min(), vcenter=3.25, vmax=labels.max())   # 初始化颜色条
#     scatter = ax.scatter(umap_results['layer_cat'][:, 0], umap_results['layer_cat'][:, 1], c=labels, cmap="coolwarm", norm=divnorm, s=2, alpha=0.5,
#                          marker="o", linewidth=0)
#     ax.set_xticks([])  # 移除x轴刻度
#     ax.set_yticks([])  # 移除y轴刻度
#     ax.set_xticklabels([])  # 移除x轴标签
#     ax.set_yticklabels([])  # 移除y轴标签
#     ax.spines['top'].set_visible(False)  # 移除上边框
#     ax.spines['right'].set_visible(False)  # 移除右边框
#     # ax.spines['bottom'].set_visible(False)  # 移除下边框
#     # ax.spines['left'].set_visible(False)  # 移除左边框
#     # 加粗右边和下边框
#     ax.spines['left'].set_linewidth(3)  # 加粗右边框
#     ax.spines['bottom'].set_linewidth(3) # 加粗下边框
#     # ax.set_title(f'UMAP - {layer} {title_suffix}')
#     # 添加颜色条并设置格式
#     cbar = fig.colorbar(scatter, ax=ax)
#
#     # 设置颜色条标签字号（加大到14）
#     cbar.ax.tick_params(labelsize=24)
#
#     # 设置刻度长度（默认是5，这里设为8）
#     cbar.ax.tick_params(size=10, width=2)  # size控制长度，width控制粗细
#     plt.tight_layout()
#     plt.show()
#
# def plot_umap(umap_results, labels, title_suffix=""):
#     fig, axes = plt.subplots(2, 2, figsize=(10, 8))
#     axes = axes.ravel()
#     for (layer, emb), ax in zip(umap_results.items(), axes):
#         divnorm = colors.TwoSlopeNorm(vmin=labels.min(), vcenter=3.25, vmax=labels.max())   # 初始化颜色条
#         scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="coolwarm", norm=divnorm, s=2, alpha=0.5,
#                              marker="o", linewidth=0)
#         # ax.set_xticks([])  # 移除x轴刻度
#         # ax.set_yticks([])  # 移除y轴刻度
#         # ax.set_xticklabels([])  # 移除x轴标签
#         # ax.set_yticklabels([])  # 移除y轴标签
#         ax.spines['top'].set_visible(False)  # 移除上边框
#         ax.spines['right'].set_visible(False)  # 移除右边框
#         # ax.spines['bottom'].set_visible(False)  # 移除下边框
#         # ax.spines['left'].set_visible(False)  # 移除左边框
#         # 加粗右边和下边框
#         ax.spines['left'].set_linewidth(1.5)  # 加粗右边框
#         ax.spines['bottom'].set_linewidth(1.5) # 加粗下边框
#         # ax.set_title(f'UMAP - {layer} {title_suffix}')
#         # 添加颜色条并设置格式
#         cbar = fig.colorbar(scatter, ax=ax)
#
#         # 设置颜色条标签字号（加大到14）
#         cbar.ax.tick_params(labelsize=14)
#
#         # 设置刻度长度（默认是5，这里设为8）
#         cbar.ax.tick_params(size=8, width=1.5)  # size控制长度，width控制粗细
#     plt.tight_layout()
#     plt.show()
#
# # 将数据转换为迭代器加载到GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # set the encoded data to its own var to make things cleaner
# train_loader = utils.create_dataloader(torch.tensor(data["encoded_data"]["encoded_data"]),
#                                        torch.tensor(data["scores"]), batch_size=8)
#
# # 使用示例
# features_dict, labels = extract_features(loaded_model, train_loader, device)
# umap_results = apply_umap(features_dict)
# plot_umap(umap_results, labels, "(cnn_3X128 Model)")
# plot_umap_1(umap_results, labels, "(cnn_3X128 Model)")

"""
实现对回归分析数据润色
"""
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# def plot_true_vs_predicted(csv_file):
#     """
#     读取CSV文件并绘制true_label与predicted_value的散点图
#
#     参数:
#         csv_file (str): CSV文件路径
#     """
#     # 读取CSV文件
#     data = pd.read_csv(csv_file)
#     model_name = 'sgcn_5X64     WR3'
#
#
#     # 画图
#     plt.figure(figsize=(7, 7))
#
#     # 散点图（真实值 vs 预测值）
#     sns.scatterplot(x=data['true_label'], y=data['predicted_value'], s=30, linewidth=0)
#     plt.xlim(0, 7.5)  # min_val和max_val是你要的范围
#     plt.ylim(0, 7.5)
#     # 添加理想线（y=x，预测完全正确时应重合）
#     min_val = 0
#     max_val = 7.5
#     plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2)
#
#     plt.xlabel('')  # 移除x轴标题
#     plt.ylabel('')  # 移除y轴标题
#     plt.legend()
#     plt.grid(False)   # 图像中的网格线
#
#     # 获取当前坐标轴对象
#     ax = plt.gca()
#     ax.set_yticks([0, 2, 4, 6])  # 这将只显示0,2,4,6这几个刻度
#     ax.set_xticks([0, 2, 4, 6])  # 这将只显示0,2,4,6这几个刻度
#
#     # 设置x轴和y轴标签的样式（字号+加粗）
#     ax.tick_params(
#         axis='both',  # 同时修改x轴和y轴
#         which='major',  # 主刻度标签
#         labelsize=36,  # 字号14
#         length=10,  # 刻度线长度（新增，默认是5-8左右）
#         pad=10, # 标签位移距离
#         width=3  # 刻度线粗细
#     )
#
#     # 加粗坐标轴边框（上下左右四边）
#     for spine in ax.spines.values():
#         spine.set_linewidth(3)  # 边框粗细2pt
#     # 自动调整布局
#     plt.tight_layout()
#     # 保存图片（可选）
#     plt.savefig(f"./{model_name}_visual.png", dpi=300, bbox_inches='tight')
#     plt.show()
#     print(1)
#
#
# # 使用示例
# if __name__ == "__main__":
#     csv_file = './output/sgcn_5X64_pred_labels.csv'
#     plot_true_vs_predicted(csv_file)


'''
结构图
'''

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import networkx as nx
#
# import gen_stru_graph as gsg
# from matplotlib.colors import ListedColormap
# import constants
# import utils
#
# # calculate the residue distance matrix
# pdb_fn = "../data/cyp2c9/AF-P11712-F1-model_v4.pdb"
# dist_mtx = gsg.cbeta_distance_matrix(pdb_fn)
#
# min_val = dist_mtx.min()
# max_val = dist_mtx.max()
#
# # fig, ax = plt.subplots(1)
# # ax = sns.heatmap(dist_mtx, ax=ax, square=True, cmap='rocket_r')
# #
# # ax.set(xlabel="Residue Num", ylabel="Residue Num", title="Residue distance matrix (cyp2c9)")
# # plt.show()
# # plt.close(fig)
#
# mask = np.triu(np.ones_like(dist_mtx, dtype=np.bool), 1)
# all_distances = dist_mtx[mask]
# # fig, ax = plt.subplots(1)
# # ax = sns.distplot(all_distances, bins=50, hist=True, kde=False, ax=ax)
# # ax.set(xlabel="Distance ($\AA$)", ylabel="Count", title="Histogram of res-res distances (cyp2c9)")
# # plt.show()
# # plt.close(fig)
#
# # Visualize the contact maps at various thresholds
# # thresholds = [4, 6, 8, 10, 12, 14, 16, 18]
# # cmap = ListedColormap(['white', '#802934'])
# # fig, axes = plt.subplots(nrows=2, ncols=4, squeeze=True, sharey=True, sharex=True, figsize=(12, 6), dpi=300)
# # axes = axes.flatten()
# # for i, t in enumerate(thresholds):
# #     ax = axes[i]
# #     ax = sns.heatmap(np.where(dist_mtx < t, 1, 0), square=True, cbar=False, cmap=cmap, ax=ax)
# #     ax.set(title="t={}$\AA$".format(t))
# # fig.suptitle("Contact maps at various thresholds (cyp2c9)", y=1.02)
# # plt.tight_layout()
# # plt.show()
# # plt.close(fig)
#
# # Visualize the contact maps at various thresholds
# # thresholds = [14,]
# # cmap = ListedColormap(['white', '#802934'])
# # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), dpi=300)
# # ax = sns.heatmap(np.where(dist_mtx < 14, 1, 0), square=True, cbar=False, cmap=cmap, ax=ax)
# # ax.set(xticks=[], yticks=[], xlabel=None, ylabel=None, title=None)
# # ax.tick_params(left=False, bottom=False)  # 去除刻度线
# #
# # plt.tight_layout()
# # plt.savefig("./maps_thresholds.png", bbox_inches='tight', dpi=300)
# # plt.show()
# # plt.close(fig)
#
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
# t = [6]
# g = gsg.dist_thresh_graph(dist_mtx, t)
# node_values = [node + 1 for node in g.nodes()]
# node_layout = nx.spring_layout(g, seed=1) #节点随机影响，模拟真实图结构
# nx.draw(g, cmap=plt.get_cmap("Blues"), with_labels=False, node_color=node_values,
#         node_size=100, edgecolors="black",  width=1, linewidths=0.4, pos=node_layout, ax=ax)
# # ax.set(title="t={}$\AA$".format(t))
# # ax.axis("on")
# # fig.suptitle("Structure graphs at various thresholds (cyp2c9)", y=1.02, fontsize=16)
# plt.tight_layout()
# plt.savefig(f"./t=6A.png", dpi=300, bbox_inches='tight')
# plt.show()
# print(0)
# # plt.close(fig)

"""
示意图
"""
#
# import matplotlib.pyplot as plt
# # 示例数据
# categories = [0, 0.5, 1, 1.5, 2]
# values_GG1 = [0.5, 3, 3, 5, 5]  # 第一组数据（底部）
# values_GG2 = [1.5, 1, 1, 2, 2]   # 第二组数据（堆叠在顶部）
#
# values_GA1 = [0.5, 0.5, 3, 3, 5]  # 第一组数据（底部）
# values_GA2 = [1.5, 1.5, 1, 1, 2]   # 第二组数据（堆叠在顶部）
#
# values_AA1 = [0.5, 0.5, 0.5, 3, 3]  # 第一组数据（底部）
# values_AA2 = [1.5, 1.5, 1.5, 1, 1]   # 第二组数据（堆叠在顶部）
#
#
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 8), dpi=100, sharex=True)
#
# for ax in axes:
#     # 加粗所有边框（线宽设为2）
#     for spine in ax.spines.values():
#         spine.set_linewidth(1)
#
#     # 设置刻度标签大小
#     ax.tick_params(axis='both', which='major', labelsize=11)
#
# # 定义折线图的点
# line_points_GG = [(0, 1.25), (0.5, 3), (1, 4), (1.5, 5), (2, 7)]
# line_points_GA = [(0, 0.5), (0.5, 2), (1, 3), (1.5, 4), (2, 6)]
# line_points_AA = [(0, 0.5), (0.5, 1.25), (1, 2), (1.5, 3), (2, 4)]
#
# # 提取x和y坐标
# x_GG, y_GG = zip(*line_points_GG)
# x_GA, y_GA = zip(*line_points_GA)
# x_AA, y_AA = zip(*line_points_AA)
#
# ax = axes[0]
# ax.bar(categories, values_GG1, width=0.25, color='white')  # 柱状图
# ax.bar(categories, values_GG2, bottom=values_GG1, width=0.25, color='#B1CBE7')
# # 添加折线图
# ax.plot(x_GG, y_GG, color='#376eae', linewidth=2, marker='o', markersize=4)
# ax.grid(True, linestyle='--', alpha=0.6)
# ax.set_ylim(0, 8)
# # 设置标签字体大小
# ax.xaxis.label.set_size(10)
# ax.yaxis.label.set_size(10)
# ax.set_ylabel("VKORC1=GG(mg)", fontsize=14)  # 明确设置y轴标签大小
# ax.set_xticks(categories)
#
# ax = axes[1]
# ax.bar(categories, values_GA1, width=0.25, color='white')
# ax.bar(categories, values_GA2, bottom=values_GA1, width=0.25, color='#B1CBE7')
# # 添加折线图
# ax.plot(x_GA, y_GA, color='#376eae', linewidth=2, marker='o', markersize=4)
# ax.grid(True, linestyle='--', alpha=0.6)
# ax.set_xticks(categories)
# ax.set_ylim(0, 8)
# # 设置标签字体大小
# ax.xaxis.label.set_size(10)
# ax.yaxis.label.set_size(10)
# ax.set_ylabel("VKORC1=GA(mg)", fontsize=14)
#
# ax = axes[2]
# ax.bar(categories, values_AA1, width=0.25, color='white')
# ax.bar(categories, values_AA2, bottom=values_AA1, width=0.25, color='#B1CBE7')
# # 添加折线图
# ax.plot(x_AA, y_AA, color='#376eae', linewidth=2, marker='o', markersize=4)
# ax.grid(True, linestyle='--', alpha=0.6)
# ax.set_xticks(categories)
# ax.set_ylim(0, 8)
# # 设置标签字体大小
# ax.xaxis.label.set_size(10)
# ax.yaxis.label.set_size(10)
# ax.set_ylabel("VKORC1=AA(mg)", fontsize=14)
# ax.set_xlabel("Activity score", fontsize=14)  # 设置x轴标签大小
#
# plt.tight_layout(rect=[0, 0.02, 1, 1])
# # plt.show()
# plt.savefig('../data/cyp2c9/figue/warfarin.png', dpi=300, bbox_inches='tight')
# print(1)


"""
将cnn中batch_size为64的行去除
"""
# import os
# import pandas as pd
#
# # 设置文件夹路径
# folder_path = './output/training_logs/cnn'  # 请替换为实际的文件夹路径
# output_path = './output/training_logs/all'
# # 获取文件夹中所有的CSV文件
# csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
#
# # 遍历每个CSV文件
# for csv_file in csv_files:
#     # 构建完整的文件路径
#     file_path = os.path.join(folder_path, csv_file)
#     output_file_path = os.path.join(output_path, csv_file)
#
#     # 读取CSV文件
#     df = pd.read_csv(file_path)
#
#     # 检查是否存在batchsize列
#     if 'batch_size' in df.columns:
#         # 去除batchsize列中所有值为64的行
#         df = df[df['batch_size'] != 64]
#
#         # 以原始文件名保存（覆盖原文件）
#         df.to_csv(output_file_path, index=False)
#         print(f"已处理文件: {csv_file}")
#     else:
#         print(f"文件 {csv_file} 中没有batch_size列，跳过处理")
#
# print("所有文件处理完成！")