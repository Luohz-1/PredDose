import logging
import os.path
import random
import time
import uuid
from datetime import datetime
from os.path import join, basename, isdir
import gen_stru_graph as gsg
from scipy.stats import pearsonr, spearmanr

# 3rd party
import numpy as np
import pandas as pd
import shortuuid
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import constants
import encode as enc
import build_model as bm
import matplotlib.pyplot as plt
import seaborn as sns
# mine
import utils
from parse_args import get_parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CYPpred." + __name__)
logger.setLevel(logging.INFO)


def run_trainin(data, args):  # 循环训练，用于确定最佳超参数
    logger.info("setting random seeds py={}, np={}, tf={}".format(args.py_rseed, args.np_rseed, args.np_rseed))
    random.seed(args.py_rseed)
    np.random.seed(args.np_rseed)
    torch.manual_seed(args.np_rseed)
    torch.cuda.manual_seed_all(args.np_rseed)  # 设置随机种子

    # results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=args.np_rseed)

    if len(args.graph_fn) != 0:

        columns = [
            'graph_thresholds', 'lr', 'batch_size', 'loss', 'early_stop',
            'pearson_r_mean', 'pearson_r_sd', 'pearson_r_1', 'pearson_r_2', 'pearson_r_3', 'pearson_r_4',
            'pearson_r_5', 'r2_mean', 'r2_sd', 'r2_1', 'r2_2', 'r2_3', 'r2_4', 'r2_5',
            'spearman_r_mean', 'spearman_r_sd', 'spearman_r_1', 'spearman_r_2', 'spearman_r_3', 'spearman_r_4',
            'spearman_r_5',
        ]
        train_df = pd.DataFrame(columns=columns)
        test_df = pd.DataFrame(columns=columns)

        graph_filedir = args.graph_fn.split(",")
        lr_list = [float(x) for x in args.learning_rate.split(",")]
        batch_list = [int(x) for x in args.batch_size.split(",")]

        for graph_file in graph_filedir:
            g = gsg.load_graph(graph_file)  # 加载在 geg 代码中生成的图
            g = gsg.expand_adj(g)

            adj_mtx = gsg.ordered_adjacency_matrix(g)  # 得到邻接热图
            adj = torch.tensor(adj_mtx, dtype=torch.float32)  # 这个才是邻接矩阵

            for lr in lr_list:
                for batch in batch_list:
                    i = 1
                    data_dict = {
                        "train": {
                            "loss": [],
                            "pearson_r": [],
                            "R2": [],
                            "spearman_r": []
                        },
                        "test": {
                            "loss": [],
                            "pearson_r": [],
                            "R2": [],
                            "spearman_r": []
                        },
                        "early_stats": []
                    }

                    for fold, (train_idx, val_idx) in enumerate(kf.split(data["encoded_data"]["encoded_data"])):

                        # 早停
                        early_stopping = EarlyStopping()

                        # set the encoded data to its own var to make things cleaner
                        train_loader = utils.create_dataloader(
                            torch.tensor(data["encoded_data"]["encoded_data"][train_idx]),
                            torch.tensor(data["scores"][train_idx]),
                            vk=torch.tensor(data["encoded_data"]["vk_coding"][train_idx]),
                            batch_size=batch)
                        val_loader = utils.create_dataloader(
                            torch.tensor(data["encoded_data"]["encoded_data"][val_idx]),
                            torch.tensor(data["scores"][val_idx]),
                            vk=torch.tensor(data["encoded_data"]["vk_coding"][val_idx]),
                            batch_size=batch, shuffle=False)

                        # 将数据加载到GPU
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        # model = bm.TFStyleGCN(num_features=40, adj_matrix=adj.to(device)).to(device)

                        model = bm.GCNdp5(num_features=data["encoded_data"]["encoded_data"].shape[3],
                                          vk=data["encoded_data"]["vk_coding"].shape[1],
                                          adj_matrix=adj.to(device)).to(device)

                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        criterion = torch.nn.MSELoss()

                        # 训练循环

                        for epoch in range(args.epochs):
                            val_loss = train_gc(model, train_loader, device, optimizer, criterion)
                            val_metrics = early_stopping(val_loss, model, train_loader, val_loader, device, criterion)

                            if early_stopping.early_stop:
                                break
                        # val_metrics = eval_gc(model, train_loader, val_loader, device, criterion)

                        data_dict["early_stats"].append(epoch)
                        data_dict["train"]["loss"].append(val_metrics[0])
                        data_dict["test"]["loss"].append(val_metrics[1])
                        data_dict["train"]["pearson_r"].append(val_metrics[2])
                        data_dict["train"]["R2"].append(val_metrics[3])
                        data_dict["train"]["spearman_r"].append(val_metrics[4])
                        data_dict["test"]["pearson_r"].append(val_metrics[5])
                        data_dict["test"]["R2"].append(val_metrics[6])
                        data_dict["test"]["spearman_r"].append(val_metrics[7])

                        # 每次训练，输出打印最终参数
                        current_time = datetime.now().strftime("%m-%d %H:%M:%S")
                        print(
                            f"[{current_time}]第{len(train_df) + 1}-{i}次训练，graph: {graph_file}, lr: {lr}, batch_size: {batch}\t"
                            f"loss_train: {val_metrics[0]:.4}, pearson_train: {val_metrics[2]:.4},\t"
                            f"loss_test: {val_metrics[1]:.4}, pearson_test: {val_metrics[5]:.4}")
                        i = i + 1

                    # 将相关参数加载到数据框中
                    train_df.loc[len(train_df)] = {
                        # 固定参数
                        'graph_thresholds': os.path.basename(graph_file),
                        'lr': lr,
                        'batch_size': batch,
                        'loss': round(np.mean(data_dict["train"]["loss"]), 4),
                        'early_stop': round(np.mean(data_dict["early_stats"]), 4),

                        # Pearson R结果
                        'pearson_r_mean': round(np.mean(data_dict["train"]["pearson_r"]), 4),
                        'pearson_r_sd': round(np.std(data_dict["train"]["pearson_r"]), 4),
                        **{f'pearson_r_{i + 1}': data_dict["train"]["pearson_r"][i] for i in range(0, 5)},
                        # ** 含义是用于解开字典，如果不使用这个方法，那么程序将作为字典输入函数，如果解开字典，那么将以字典的键值对输入程序
                        # R²结果
                        'r2_mean': round(np.mean(data_dict["train"]["R2"]), 4),
                        'r2_sd': round(np.std(data_dict["train"]["R2"]), 4),
                        **{f'r2_{i + 1}': data_dict["train"]["R2"][i] for i in range(0, 5)},

                        # Spearman_r结果
                        'spearman_r_mean': round(np.mean(data_dict["train"]["spearman_r"]), 4),
                        'spearman_r_sd': round(np.std(data_dict["train"]["spearman_r"]), 4),
                        **{f'spearman_r_{i + 1}': data_dict["train"]["spearman_r"][i] for i in range(0, 5)},
                    }
                    test_df.loc[len(test_df)] = {
                        # 固定参数
                        'graph_thresholds': os.path.basename(graph_file),
                        'lr': lr,
                        'batch_size': batch,
                        'loss': round(np.mean(data_dict["test"]["loss"]), 4),
                        'early_stop': round(np.mean(data_dict["early_stats"]), 4),

                        # Pearson R结果
                        'pearson_r_mean': round(np.mean(data_dict["test"]["pearson_r"]), 4),
                        'pearson_r_sd': round(np.std(data_dict["test"]["pearson_r"]), 4),
                        **{f'pearson_r_{i + 1}': data_dict["test"]["pearson_r"][i] for i in range(0, 5)},

                        # R²结果
                        'r2_mean': round(np.mean(data_dict["test"]["R2"]), 4),
                        'r2_sd': round(np.std(data_dict["test"]["R2"]), 4),
                        **{f'r2_{i + 1}': data_dict["test"]["R2"][i] for i in range(0, 5)},

                        # Spearman_r结果
                        'spearman_r_mean': round(np.mean(data_dict["test"]["spearman_r"]), 4),
                        'spearman_r_sd': round(np.std(data_dict["test"]["spearman_r"]), 4),
                        **{f'spearman_r_{i + 1}': data_dict["test"]["spearman_r"][i] for i in range(0, 5)},
                    }

        # 排序并打印训练的关键信息
        print("train")
        train_df = rank_print_metrics(train_df)
        print("test")
        test_df = rank_print_metrics(test_df)

        train_df.to_csv(f"{args.log_dir_base}/{os.path.basename(args.net_file)}_train.csv", index=False)
        test_df.to_csv(f"{args.log_dir_base}/{os.path.basename(args.net_file)}_test.csv", index=False)
    else:  # 处理序列数据

        columns = [
            'kernel_size', 'lr', 'batch_size', 'loss', 'early_stop',
            'pearson_r_mean', 'pearson_r_sd', 'pearson_r_1', 'pearson_r_2', 'pearson_r_3', 'pearson_r_4',
            'pearson_r_5', 'r2_mean', 'r2_sd', 'r2_1', 'r2_2', 'r2_3', 'r2_4', 'r2_5',
            'spearman_r_mean', 'spearman_r_sd', 'spearman_r_1', 'spearman_r_2', 'spearman_r_3', 'spearman_r_4',
            'spearman_r_5',
        ]
        train_df = pd.DataFrame(columns=columns)
        test_df = pd.DataFrame(columns=columns)

        kernel_list = [x for x in args.kernel_size.split(",")] # 专利训练需要
        # kernel_list = [int(x) for x in args.kernel_size.split(",")]  # 全连接与回归分析需要去除这个循环
        lr_list = [float(x) for x in args.learning_rate.split(",")]
        batch_list = [int(x) for x in args.batch_size.split(",")]

        for kernel in kernel_list:
            for lr in lr_list:
                for batch in batch_list:
                    i = 1
                    data_dict = {
                        "train": {
                            "loss": [],
                            "pearson_r": [],
                            "R2": [],
                            "spearman_r": []
                        },
                        "test": {
                            "loss": [],
                            "pearson_r": [],
                            "R2": [],
                            "spearman_r": []
                        },
                        "early_stats": []
                    }

                    for fold, (train_idx, val_idx) in enumerate(kf.split(data["encoded_data"]["encoded_data"])):

                        # 早停
                        early_stopping = EarlyStoppingCnn()

                        # set the encoded data to its own var to make things cleaner
                        train_loader = utils.create_dataloader(
                            torch.tensor(data["encoded_data"]["encoded_data"][train_idx]),
                            torch.tensor(data["scores"][train_idx]), batch_size=batch)
                        val_loader = utils.create_dataloader(
                            torch.tensor(data["encoded_data"]["encoded_data"][val_idx]),
                            torch.tensor(data["scores"][val_idx]), batch_size=batch, shuffle=False)

                        # 将数据加载到GPU
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                        # 专利模型
                        # model = bm.create_transformer_model(model_type=kernel, input_shape=(2, 490, 43)).to(device)
                        # Inception-ResNet
                        # config = bm.InceptionResNetConfig('standard')
                        model = bm.create_inception_resnet(model_type=kernel).to(device)

                        # print(model)

                        # 序列卷积使用这条命令
                        # model = bm.GCNdp6(num_features=data["encoded_data"]["encoded_data"].shape[3],
                        #                   kernel_size=kernel).to(device)

                        # 全连接与线性回归使用这条命令
                        # model = bm.GCNdp7().to(device)

                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        criterion = torch.nn.MSELoss()

                        # 训练循环

                        for epoch in range(args.epochs):
                            val_loss = train_cn(model, train_loader, device, optimizer, criterion)
                            val_metrics = early_stopping(val_loss, model, train_loader, val_loader, device, criterion)

                            if early_stopping.early_stop:
                                break

                        data_dict["early_stats"].append(epoch)
                        data_dict["train"]["loss"].append(val_metrics[0])
                        data_dict["test"]["loss"].append(val_metrics[1])
                        data_dict["train"]["pearson_r"].append(val_metrics[2])
                        data_dict["train"]["R2"].append(val_metrics[3])
                        data_dict["train"]["spearman_r"].append(val_metrics[4])
                        data_dict["test"]["pearson_r"].append(val_metrics[5])
                        data_dict["test"]["R2"].append(val_metrics[6])
                        data_dict["test"]["spearman_r"].append(val_metrics[7])

                        # 每次训练，输出打印最终参数
                        current_time = datetime.now().strftime("%m-%d %H:%M:%S")
                        print(
                            f"[{current_time}]：第{len(train_df) + 1}-{i}次训练，graph: {kernel}, lr: {lr}, batch_size: {batch}\t"
                            f"loss_train: {val_metrics[0]:.4}, pearson_train: {val_metrics[2]:.4},\t"
                            f"loss_test: {val_metrics[1]:.4}, pearson_test: {val_metrics[5]:.4}")
                        i = i + 1

                    # 将相关参数加载到数据框中
                    train_df.loc[len(train_df)] = {
                        # 固定参数
                        'kernel_size': kernel,
                        'lr': lr,
                        'batch_size': batch,
                        'loss': round(np.mean(data_dict["train"]["loss"]), 4),
                        'early_stop': round(np.mean(data_dict["early_stats"]), 4),

                        # Pearson R结果
                        'pearson_r_mean': round(np.mean(data_dict["train"]["pearson_r"]), 4),
                        'pearson_r_sd': round(np.std(data_dict["train"]["pearson_r"]), 4),
                        **{f'pearson_r_{i + 1}': data_dict["train"]["pearson_r"][i] for i in range(0, 5)},
                        # ** 含义是用于解开字典，如果不使用这个方法，那么程序将作为字典输入函数，如果解开字典，那么将以字典的键值对输入程序
                        # R²结果
                        'r2_mean': round(np.mean(data_dict["train"]["R2"]), 4),
                        'r2_sd': round(np.std(data_dict["train"]["R2"]), 4),
                        **{f'r2_{i + 1}': data_dict["train"]["R2"][i] for i in range(0, 5)},

                        # Spearman_r结果
                        'spearman_r_mean': round(np.mean(data_dict["train"]["spearman_r"]), 4),
                        'spearman_r_sd': round(np.std(data_dict["train"]["spearman_r"]), 4),
                        **{f'spearman_r_{i + 1}': data_dict["train"]["spearman_r"][i] for i in range(0, 5)},
                    }
                    test_df.loc[len(test_df)] = {
                        # 固定参数
                        'kernel_size': kernel,
                        'lr': lr,
                        'batch_size': batch,
                        'loss': round(np.mean(data_dict["test"]["loss"]), 4),
                        'early_stop': round(np.mean(data_dict["early_stats"]), 4),

                        # Pearson R结果
                        'pearson_r_mean': round(np.mean(data_dict["test"]["pearson_r"]), 4),
                        'pearson_r_sd': round(np.std(data_dict["test"]["pearson_r"]), 4),
                        **{f'pearson_r_{i + 1}': data_dict["test"]["pearson_r"][i] for i in range(0, 5)},

                        # R²结果
                        'r2_mean': round(np.mean(data_dict["test"]["R2"]), 4),
                        'r2_sd': round(np.std(data_dict["test"]["R2"]), 4),
                        **{f'r2_{i + 1}': data_dict["test"]["R2"][i] for i in range(0, 5)},

                        # Spearman_r结果
                        'spearman_r_mean': round(np.mean(data_dict["test"]["spearman_r"]), 4),
                        'spearman_r_sd': round(np.std(data_dict["test"]["spearman_r"]), 4),
                        **{f'spearman_r_{i + 1}': data_dict["test"]["spearman_r"][i] for i in range(0, 5)},
                    }

        # 排序并打印训练的关键信息
        print("train")
        train_df = rank_print_metrics(train_df)
        print("test")
        test_df = rank_print_metrics(test_df)

        train_df.to_csv(f"{args.log_dir_base}/{os.path.basename(args.net_file)}_train.csv", index=False)
        test_df.to_csv(f"{args.log_dir_base}/{os.path.basename(args.net_file)}_test.csv", index=False)


# pearson_r_mean
def rank_print_metrics(df):
    df.insert(df.columns.get_loc('pearson_r_mean') + 1, 'pearson_r_rank',
              df['pearson_r_mean'].rank(ascending=False, method='min').astype(int))
    df.insert(df.columns.get_loc('r2_mean') + 1, 'r2_rank',
              df['r2_mean'].rank(ascending=False, method='min').astype(int))
    df.insert(df.columns.get_loc('spearman_r_mean') + 1, 'spearman_r_rank',
              df['spearman_r_mean'].rank(ascending=False, method='min').astype(int))

    pearson_r_best = df[df['pearson_r_rank'] == 1]
    r2_best = df[df['r2_rank'] == 1]
    spearman_r_best = df[df['spearman_r_rank'] == 1]

    print(
        f"graph: {pearson_r_best.iloc[0, 0]}, \tlr: {pearson_r_best['lr'].values}, "
        f"\tbatch_size: {pearson_r_best['batch_size'].values}, \tbest_pearson_r: {pearson_r_best['pearson_r_mean'].values}\n"
        f"graph: {r2_best.iloc[0, 0]}, \tlr: {r2_best['lr'].values}, "
        f"\tbatch_size: {r2_best['batch_size'].values}, \tbest_r2: {r2_best['r2_mean'].values}\n"
        f"graph: {spearman_r_best.iloc[0, 0]}, \tlr: {spearman_r_best['lr'].values},"
        f"\tbatch_size: {spearman_r_best['batch_size'].values}, \tspearman_r: {spearman_r_best['spearman_r_mean'].values}"
    )
    return df


def eval_gc(model, train_loader, test_loader, device, criterion):
    model.eval()  # 用于关闭 Dropout 层和 Batchnorm 层，使其在验证中保持训练时的权重进行训练
    total_train_loss = 0
    total_loss = 0
    all_train_labels = []
    all_train_preds = []
    all_labels = []
    all_preds = []

    with torch.no_grad():  # 用于禁用下面验证中的自动求导操作，减少训练成本
        for batch_train_features, batch_train_vk, batch_train_labels in train_loader:
            batch_train_features = batch_train_features.to(device)  # (batch_size, 490, 40)
            batch_train_labels = batch_train_labels.to(device)  # (batch_size,)
            batch_train_vk = batch_train_vk.to(device)

            pred_train = model(batch_train_features, batch_train_vk)
            # 计算 loss
            loss_train = criterion(pred_train.flatten(), batch_train_labels)
            total_train_loss += loss_train.item()  # item() 得到 tensor 中变量的值，而不是 tensor 类型的数值
            # 将 GPU 中的变量输出到 CPU 中收集预测标签和真实标签
            all_train_preds.append(pred_train.detach().cpu().numpy().flatten())
            all_train_labels.append(batch_train_labels.detach().cpu().numpy())

        for batch_features, batch_vk, batch_labels in test_loader:
            batch_features = batch_features.to(device)  # (batch_size, 490, 40)
            batch_labels = batch_labels.to(device)  # (batch_size,)
            batch_vk = batch_vk.to(device)

            pred = model(batch_features, batch_vk)
            # 计算 loss
            loss = criterion(pred.flatten(), batch_labels)
            total_loss += loss.item()  # item() 得到 tensor 中变量的值，而不是 tensor 类型的数值
            # 将 GPU 中的变量输出到 CPU 中收集预测标签和真实标签
            all_preds.append(pred.detach().cpu().numpy().flatten())
            all_labels.append(batch_labels.detach().cpu().numpy())

    all_train_preds = np.concatenate(all_train_preds)
    all_train_labels = np.concatenate(all_train_labels)
    pearson_r_train, pearson_p_train = pearsonr(all_train_labels, all_train_preds)
    r2_train = r2_score(all_train_labels, all_train_preds)
    spearman_r_train, spearman_p_train = spearmanr(all_train_labels, all_train_preds)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    pearson_r_test, pearson_p_test = pearsonr(all_labels, all_preds)
    r2_test = r2_score(all_labels, all_preds)
    spearman_r_test, spearman_p_test = spearmanr(all_labels, all_preds)

    return [total_train_loss / len(train_loader), total_loss / len(test_loader),
            pearson_r_train, r2_train, spearman_r_train, pearson_r_test, r2_test, spearman_r_test]


def train_gc(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_features, batch_vk, batch_labels in loader:
        batch_features = batch_features.to(device)  # (batch_size, 490, 40)
        batch_labels = batch_labels.to(device)  # (batch_size,)
        batch_vk = batch_vk.to(device)
        optimizer.zero_grad()  # 优化器清零

        pred = model(batch_features, batch_vk)
        loss = criterion(pred.flatten(), batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def train_cn(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device)  # (batch_size, 490, 40)
        batch_labels = batch_labels.to(device)  # (batch_size,)
        optimizer.zero_grad()  # 优化器清零

        pred = model(batch_features)
        loss = criterion(pred.flatten(), batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def eval_cn(model, train_loader, test_loader, device, criterion):
    model.eval()  # 用于关闭 Dropout 层和 Batchnorm 层，使其在验证中保持训练时的权重进行训练
    total_train_loss = 0
    total_loss = 0
    all_train_labels = []
    all_train_preds = []
    all_labels = []
    all_preds = []

    with torch.no_grad():  # 用于禁用下面验证中的自动求导操作，减少训练成本
        for batch_train_features, batch_train_labels in train_loader:
            batch_train_features = batch_train_features.to(device)  # (batch_size, 490, 40)
            batch_train_labels = batch_train_labels.to(device)  # (batch_size,)

            pred_train = model(batch_train_features)
            # 计算 loss
            loss_train = criterion(pred_train.flatten(), batch_train_labels)
            total_train_loss += loss_train.item()  # item() 得到 tensor 中变量的值，而不是 tensor 类型的数值
            # 将 GPU 中的变量输出到 CPU 中收集预测标签和真实标签
            all_train_preds.append(pred_train.detach().cpu().numpy().flatten())
            all_train_labels.append(batch_train_labels.detach().cpu().numpy())

        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)  # (batch_size, 490, 40)
            batch_labels = batch_labels.to(device)  # (batch_size,)

            pred = model(batch_features)
            # 计算 loss
            loss = criterion(pred.flatten(), batch_labels)
            total_loss += loss.item()  # item() 得到 tensor 中变量的值，而不是 tensor 类型的数值
            # 将 GPU 中的变量输出到 CPU 中收集预测标签和真实标签
            all_preds.append(pred.detach().cpu().numpy().flatten())
            all_labels.append(batch_labels.detach().cpu().numpy())

    all_train_preds = np.concatenate(all_train_preds)
    all_train_labels = np.concatenate(all_train_labels)
    pearson_r_train, pearson_p_train = pearsonr(all_train_labels, all_train_preds)
    r2_train = r2_score(all_train_labels, all_train_preds)
    spearman_r_train, spearman_p_train = spearmanr(all_train_labels, all_train_preds)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    pearson_r_test, pearson_p_test = pearsonr(all_labels, all_preds)
    r2_test = r2_score(all_labels, all_preds)
    spearman_r_test, spearman_p_test = spearmanr(all_labels, all_preds)

    return [total_train_loss / len(train_loader), total_loss / len(test_loader),
            pearson_r_train, r2_train, spearman_r_train, pearson_r_test, r2_test, spearman_r_test]


# 早停机制
class EarlyStopping:
    def __init__(self, patience=10, delta=0.000001):
        self.patience = patience
        self.delta = delta
        self.val_metrics = None
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model, train_loader, val_loader, device, criterion):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return self.val_metrics

        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return self.val_metrics

        else:
            self.best_score = score
            self.val_metrics = eval_gc(model, train_loader, val_loader, device, criterion)
            self.counter = 0
            return self.val_metrics


class EarlyStoppingCnn:
    def __init__(self, patience=10, delta=0.000001):
        self.patience = patience
        self.delta = delta
        self.val_metrics = None
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model, train_loader, val_loader, device, criterion):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return self.val_metrics

        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return self.val_metrics

        else:
            self.best_score = score
            self.val_metrics = eval_cn(model, train_loader, val_loader, device, criterion)
            self.counter = 0
            return self.val_metrics


def run_training(data, args):  # 控制单个参数从而训练模型
    # 可以实现：保存训练数据的索引（训练集与验证集）、模型的权重、顺带画图

    logger.info("setting random seeds py={}, np={}, tf={}".format(args.py_rseed, args.np_rseed, args.np_rseed))
    random.seed(args.py_rseed)
    np.random.seed(args.np_rseed)
    torch.manual_seed(args.np_rseed)
    torch.cuda.manual_seed_all(args.np_rseed)  # 设置随机种子

    if not args.graph_fn == "":
        g = gsg.load_graph(args.graph_fn)  # 加载在 geg 代码中生成的图
        g = gsg.expand_adj(g)

        adj_mtx = gsg.ordered_adjacency_matrix(g)  # 得到邻接热图
        adj = torch.tensor(adj_mtx, dtype=torch.float32)  # 这个才是邻接矩阵
    batch = 32
    lr = 0.001
    kernel = 1
    model_name = 'gat_2X64'
    epochs = 200

    # results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=args.np_rseed)

    columns = [
        'model', 'kernel_size', 'lr', 'batch_size', 'loss', 'early_stop',
        'pearson_r', 'r2', 'spearman_r'
    ]
    train_df = pd.DataFrame(columns=columns)
    test_df = pd.DataFrame(columns=columns)

    for fold, (train_idx, val_idx) in enumerate(kf.split(data["encoded_data"]["encoded_data"])):
        # 0. 1. 3. 4. 5. 6. 7. 10. 11. 13. 14. 15. 17.
        # 保存索引
        if not (os.path.exists(f"../data/cyp2c9/train_idx.npy")):
            np.save(f"../data/cyp2c9/train_idx.npy", train_idx)
            np.save(f"../data/cyp2c9/test_idx.npy", val_idx)

        if not args.graph_fn == "":
            # --graph_fn
            # ../ data / cyp2c9 / graphs / dist_thresh_9.graph
            # 早停
            early_stopping = EarlyStopping()

            # set the encoded data to its own var to make things cleaner
            train_loader = utils.create_dataloader(
                torch.tensor(data["encoded_data"]["encoded_data"][train_idx]),
                torch.tensor(data["scores"][train_idx]),
                vk=torch.tensor(data["encoded_data"]["vk_coding"][train_idx]),
                batch_size=batch)
            val_loader = utils.create_dataloader(
                torch.tensor(data["encoded_data"]["encoded_data"][val_idx]),
                torch.tensor(data["scores"][val_idx]),
                vk=torch.tensor(data["encoded_data"]["vk_coding"][val_idx]),
                batch_size=batch, shuffle=False)

            # 将数据加载到GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # model = bm.TFStyleGCN(num_features=40, adj_matrix=adj.to(device)).to(device)

            model = bm.GCNdp3(num_features=data["encoded_data"]["encoded_data"].shape[3],
                              vk=data["encoded_data"]["vk_coding"].shape[1],
                              adj_matrix=adj.to(device)).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.MSELoss()

            # 训练循环
            i = 1
            for epoch in range(epochs):
                val_loss = train_gc(model, train_loader, device, optimizer, criterion)
                val_metrics = early_stopping(val_loss, model, train_loader, val_loader, device, criterion)
                print(i)
                i = i + 1
                if early_stopping.early_stop:
                    break
            torch.save(model, f"./output/{model_name}_full_model.pth")

            # 画图
            draw_resource_data = draw_graph(model_name, model, val_loader, val_idx, device)
        else:
            # 早停
            early_stopping = EarlyStoppingCnn()

            # set the encoded data to its own var to make things cleaner
            train_loader = utils.create_dataloader(
                torch.tensor(data["encoded_data"]["encoded_data"][train_idx]),
                torch.tensor(data["scores"][train_idx]), batch_size=batch)
            val_loader = utils.create_dataloader(
                torch.tensor(data["encoded_data"]["encoded_data"][val_idx]),
                torch.tensor(data["scores"][val_idx]), batch_size=batch, shuffle=False)

            # 将数据加载到GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # model = bm.GCNdp6(num_features=data["encoded_data"]["encoded_data"].shape[3],
            #                   kernel_size=kernel).to(device)  # 序列卷积使用这条命令
            model = bm.GCNdp7().to(device)  # 全连接与线性回归使用这条命令

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.MSELoss()

            # 训练循环
            i = 1
            for epoch in range(epochs):
                val_loss = train_cn(model, train_loader, device, optimizer, criterion)
                val_metrics = early_stopping(val_loss, model, train_loader, val_loader, device, criterion)

                print(i)
                i = i + 1
                if early_stopping.early_stop:
                    break

            torch.save(model, f"./output/{model_name}_full_model.pth")

            # 画图
            draw_resource_data = draw_graph_cn(model_name, model, val_loader, val_idx, device)
        break

    # 将相关参数加载到数据框中
    train_df.loc[len(train_df)] = {
        # 固定参数
        'model': model_name,
        'kernel_size': kernel,
        'lr': lr,
        'batch_size': batch,
        'loss': val_metrics[0],
        'early_stop': epoch,

        # Pearson R结果
        'pearson_r': val_metrics[2],
        'r2': val_metrics[3],
        # Spearman_r结果
        'spearman_r': val_metrics[4]
    }
    test_df.loc[len(test_df)] = {
        # 固定参数
        'model': model_name,
        'kernel_size': kernel,
        'lr': lr,
        'batch_size': batch,
        'loss': val_metrics[1],
        'early_stop': epoch,

        # Pearson R结果
        'pearson_r': draw_resource_data[0],
        # R²结果
        'r2': draw_resource_data[1],
        # Spearman_r结果
        'spearman_r': draw_resource_data[2]
    }

    train_df.to_csv(f"{args.log_dir_base}/final_train.csv", mode="a",
                    header=not os.path.exists(f"{args.log_dir_base}/final_train.csv"),
                    index=False)
    test_df.to_csv(f"{args.log_dir_base}/final_test.csv", mode="a",
                   header=not os.path.exists(f"{args.log_dir_base}/final_test.csv"),
                   index=False)


def draw_graph(model_name, model, val_loader, val_idx, device):
    model.eval()  # 用于关闭 Dropout 层和 Batchnorm 层，使其在验证中保持训练时的权重进行训练
    all_labels = []
    all_preds = []

    with torch.no_grad():  # 用于禁用下面验证中的自动求导操作，减少训练成本
        for batch_features, batch_vk, batch_labels in val_loader:
            batch_features = batch_features.to(device)  # (batch_size, 490, 40)
            batch_labels = batch_labels.to(device)  # (batch_size,)
            batch_vk = batch_vk.to(device)

            pred = model(batch_features, batch_vk)

            all_preds.append(pred.detach().cpu().numpy().flatten())
            all_labels.append(batch_labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 将结果转为DataFrame
    results = pd.DataFrame({
        'test_idx': val_idx,
        'true_label': all_labels,
        'predicted_value': all_preds
    })  # 4.3904715\3.2740993\2.260158

    results.to_csv(f"./output/{model_name}_pred_labels.csv", index=False)

    # 画图
    plt.figure(figsize=(6, 6))

    # 散点图（真实值 vs 预测值）
    sns.scatterplot(x=all_labels, y=all_preds, s=30, linewidth=0)
    plt.xlim(0, 7.5)  # min_val和max_val是你要的范围
    plt.ylim(0, 7.5)
    # 添加理想线（y=x，预测完全正确时应重合）
    min_val = 0
    max_val = 7.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=1.25)

    # 计算并显示R²和Pearson相关系数

    pearson_r, pearson_p = pearsonr(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    spearman_r, spearman_p = spearmanr(all_labels, all_preds)

    plt.title(f"R²={r2:.4f}", fontsize=12)
    # plt.xlabel("True Labels")
    # plt.ylabel("Predictions")
    plt.legend()
    plt.grid(False)  # 图像中的网格线

    # 获取当前坐标轴对象
    ax = plt.gca()

    # 设置x轴和y轴标签的样式（字号+加粗）
    ax.tick_params(
        axis='both',  # 同时修改x轴和y轴
        which='major',  # 主刻度标签
        labelsize=12,  # 字号14
        width=1.5  # 刻度线粗细
    )

    # 加粗坐标轴边框（上下左右四边）
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 边框粗细2pt
    # 自动调整布局
    plt.tight_layout()
    # 保存图片（可选）
    plt.savefig(f"./output/{model_name}_visual.png", dpi=300, bbox_inches='tight')
    # plt.show()
    # print(1)
    return [pearson_r, r2, spearman_r]


def draw_graph_cn(model_name, model, val_loader, val_idx, device):
    model.eval()  # 用于关闭 Dropout 层和 Batchnorm 层，使其在验证中保持训练时的权重进行训练
    all_labels = []
    all_preds = []

    with torch.no_grad():  # 用于禁用下面验证中的自动求导操作，减少训练成本
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)  # (batch_size, 490, 40)
            batch_labels = batch_labels.to(device)  # (batch_size,)

            pred = model(batch_features)

            all_preds.append(pred.detach().cpu().numpy().flatten())
            all_labels.append(batch_labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 将结果转为DataFrame
    results = pd.DataFrame({
        'test_idx': val_idx,
        'true_label': all_labels,
        'predicted_value': all_preds
    })  # 4.3904715\3.2740993\2.260158

    results.to_csv(f"./output/{model_name}_pred_labels.csv", index=False)

    # 画图
    plt.figure(figsize=(7, 7))

    # 散点图（真实值 vs 预测值）
    sns.scatterplot(x=all_labels, y=all_preds, s=30, linewidth=0)
    plt.xlim(0, 7.5)  # min_val和max_val是你要的范围
    plt.ylim(0, 7.5)
    # 添加理想线（y=x，预测完全正确时应重合）
    min_val = 0
    max_val = 7.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2)

    # 计算并显示R²和Pearson相关系数

    pearson_r, pearson_p = pearsonr(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    spearman_r, spearman_p = spearmanr(all_labels, all_preds)

    plt.title(f"R²={r2:.4f}", fontsize=12)

    plt.xlabel('')  # 移除x轴标题
    plt.ylabel('')  # 移除y轴标题
    plt.legend()
    plt.grid(False)  # 图像中的网格线

    # 获取当前坐标轴对象
    ax = plt.gca()
    ax.set_yticks([0, 2, 4, 6])  # 这将只显示0,2,4,6这几个刻度
    ax.set_xticks([0, 2, 4, 6])  # 这将只显示0,2,4,6这几个刻度
    # 设置x轴和y轴标签的样式（字号+加粗）
    ax.tick_params(
        axis='both',  # 同时修改x轴和y轴
        which='major',  # 主刻度标签
        labelsize=36,  # 字号14
        length=10,  # 刻度线长度（新增，默认是5-8左右）
        pad=10,  # 标签位移距离
        width=3  # 刻度线粗细
    )

    # 加粗坐标轴边框（上下左右四边）
    for spine in ax.spines.values():
        spine.set_linewidth(3)  # 边框粗细2pt
    # 自动调整布局
    plt.tight_layout()
    # 保存图片（可选）
    plt.savefig(f"./output/{model_name}_visual.png", dpi=300, bbox_inches='tight')
    # plt.show()
    # print(1)
    return [pearson_r, r2, spearman_r]


def train_gc_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    for batch_features, batch_vk, batch_labels in loader:
        batch_features = batch_features.to(device)  # (batch_size, 490, 40)
        batch_labels = batch_labels.to(device)  # (batch_size,)
        batch_vk = batch_vk.to(device)
        optimizer.zero_grad()  # 优化器清零

        pred = model(batch_features, batch_vk)
        loss = criterion(pred.flatten(), batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_preds.append(pred.detach().cpu().numpy().flatten())
        all_labels.append(batch_labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    pearson_r = pearsonr(all_labels, all_preds)[0]

    return {
        'train_mse': total_loss / len(loader),
        'train_pearson_r': pearson_r
    }


def evaluate_gc(model, loader, device, criterion):
    model.eval()  # 用于关闭 Dropout 层和 Batchnorm 层，使其在验证中保持训练时的权重进行训练
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():  # 用于禁用下面验证中的自动求导操作，减少训练成本
        for batch_features, batch_vk, batch_labels in loader:
            batch_features = batch_features.to(device)  # (batch_size, 490, 40)
            batch_labels = batch_labels.to(device)  # (batch_size,)
            batch_vk = batch_vk.to(device)

            pred = model(batch_features, batch_vk)
            # 计算 loss
            loss = criterion(pred.flatten(), batch_labels)
            total_loss += loss.item()  # item() 得到 tensor 中变量的值，而不是 tensor 类型的数值
            # 将 GPU 中的变量输出到 CPU 中收集预测标签和真实标签
            all_preds.append(pred.detach().cpu().numpy().flatten())
            all_labels.append(batch_labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    pearson_r = pearsonr(all_labels, all_preds)

    return {
        'mse': total_loss / len(loader),
        'pearson_r': pearson_r
    }


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device)  # (batch_size, 490, 40)
        batch_labels = batch_labels.to(device)  # (batch_size,)

        optimizer.zero_grad()  # 优化器清零

        pred = model(batch_features)
        loss = criterion(pred.flatten(), batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_preds.append(pred.detach().cpu().numpy().flatten())
        all_labels.append(batch_labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    pearson_r = pearsonr(all_labels, all_preds)[0]

    return {
        'train_mse': total_loss / len(loader),
        'train_pearson_r': pearson_r
    }


def evaluate(model, loader, device, criterion):
    model.eval()  # 用于关闭 Dropout 层和 Batchnorm 层，使其在验证中保持训练时的权重进行训练
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():  # 用于禁用下面验证中的自动求导操作，减少训练成本
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)  # (batch_size, 490, 40)
            batch_labels = batch_labels.to(device)  # (batch_size,)

            # x = batch_features.view(-1, 40)
            # batch_indices = torch.arange(batch_features.size(0)).repeat_interleave(490).to(device)
            # # 验证
            # pred = model(x, edge_index.to(device), batch_indices)

            pred = model(batch_features)
            # 计算 loss
            loss = criterion(pred.flatten(), batch_labels)
            total_loss += loss.item()  # item() 得到 tensor 中变量的值，而不是 tensor 类型的数值
            # 将 GPU 中的变量输出到 CPU 中收集预测标签和真实标签
            all_preds.append(pred.detach().cpu().numpy().flatten())
            all_labels.append(batch_labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    pearson_r = pearsonr(all_labels, all_preds)

    return {
        'mse': total_loss / len(loader),
        'pearson_r': pearson_r
    }


def log_dir_name(args):  # 输出 log 路径 原文件
    # log directory captures the cluster & process (if running on HTCondor), the dataset name, the
    # network specification file basename, the learning rate, the batch size, and the date and time
    log_dir_str = "log_{}_{}_{}_{}_{}_lr{}_bs{}_{}"

    # just use the net file basename
    net_arg = basename(args.net_file)[:-4]  # basename 函数用于将路径中的文件名提取出来，并去掉后缀

    # dataset file basename if no dataset_name is specified
    if args.dataset_name != "":
        ds_arg = args.dataset_name
    else:
        ds_arg = basename(args.dataset_file)[:-4]

    format_args = [args.cluster, args.process, time.strftime("%Y-%m-%d_%H-%M-%S"),
                   ds_arg, net_arg, args.learning_rate, args.batch_size, shortuuid.encode(uuid.uuid4())[:8]]

    log_dir = join(args.log_dir_base, log_dir_str.format(*format_args))  # 合并路径

    # log directory already exists. so just append a number to it.
    # should only happen if you run the script within the same second with the same args.
    # extra note: now that the log dir also includes a UUID, this *really* shouldn't happen
    if isdir(log_dir):  # 判断路径是否存在
        log_dir = log_dir + "_2"
    while isdir(log_dir):
        log_dir = "_".join(log_dir.split("_")[:-1] + [str(int(log_dir.split("_")[-1]) + 1)])
        if not isdir(log_dir):
            break
    return log_dir


def main(args):  # 与mai相似，但用于使用固定参数训练模型
    logger.info("software version {}".format(utils.__version__))

    if args.dataset_name != "":
        dataset_file = constants.DATASETS[args.dataset_name]["ds_fn"]  # constants 含有不同数据的字典，
    else:
        dataset_file = args.dataset_file
    logger.info("loading dataset from {}".format(dataset_file))
    ds = utils.load_dataset(ds_fn=dataset_file)  # 以 pandas 格式加载数据集

    # figure out the wt_aa and wt_offset for encoding data
    if args.dataset_name != "":  # 加载训练数据集
        wt_aa = constants.DATASETS[args.dataset_name]["wt_aa"]  # 野生氨基酸序列
        wt_ofs = constants.DATASETS[args.dataset_name]["wt_ofs"]
    else:
        wt_aa = args.wt_aa
        wt_ofs = args.wt_ofs

    # create the dataset dictionary, containing encoded data, scores, etc, based on the splits
    data = {}
    vk = ['score_GG', 'score_GA', 'score_AA']
    data["scores"] = []

    data["ds"] = ds
    data["variants"] = ds['variant'].tolist()
    for i in vk:
        data["scores"].append(ds[f"{i}"].to_numpy(dtype='float32'))
    data["scores"] = np.array(np.array(data["scores"]).flatten().tolist(), dtype=np.float32)
    data["encoded_data"] = enc.encode(encoding=args.encoding, graph_fn=args.graph_fn,
                                      variants=data["variants"],
                                      wt_aa=wt_aa, wt_offset=wt_ofs)

    run_training(data, args)


# 改进版本
def mai(args):  # 确定超参数
    logger.info("software version {}".format(utils.__version__))

    if args.dataset_name != "":
        dataset_file = constants.DATASETS[args.dataset_name]["ds_fn"]  # constants 含有不同数据的字典，
    else:
        dataset_file = args.dataset_file
    logger.info("loading dataset from {}".format(dataset_file))
    ds = utils.load_dataset(ds_fn=dataset_file)  # 以 pandas 格式加载数据集

    # figure out the wt_aa and wt_offset for encoding data
    if args.dataset_name != "":  # 加载训练数据集
        wt_aa = constants.DATASETS[args.dataset_name]["wt_aa"]  # 野生氨基酸序列
        wt_ofs = constants.DATASETS[args.dataset_name]["wt_ofs"]
    else:
        wt_aa = args.wt_aa
        wt_ofs = args.wt_ofs

    # create the dataset dictionary, containing encoded data, scores, etc, based on the splits
    data = {}
    vk = ['score_GG', 'score_GA', 'score_AA']
    data["scores"] = []

    data["ds"] = ds
    data["variants"] = ds['variant'].tolist()
    for i in vk:
        data["scores"].append(ds[f"{i}"].to_numpy(dtype='float32'))
    data["scores"] = np.array(np.array(data["scores"]).flatten().tolist(), dtype=np.float32)
    data["encoded_data"] = enc.encode(encoding=args.encoding, graph_fn=args.graph_fn,
                                      variants=data["variants"],
                                      wt_aa=wt_aa, wt_offset=wt_ofs)

    run_trainin(data, args)


if __name__ == "__main__":
    """
    训练循环所需要的形参
--dataset_name
cyp2c9
--net_file 定义了训练结果所需要保存的目录
network_specs/gcn_5X64.yml
--kernel_size （训练序列卷积时需要）
3,7,13,17
--graph_fn 图所需要的不同图连接图（训练图卷积的时候需要）
../data/cyp2c9/graphs/dist_thresh_7.graph,../data/cyp2c9/graphs/dist_thresh_9.graph,../data/cyp2c9/graphs/dist_thresh_11.graph,../data/cyp2c9/graphs/dist_thresh_13.graph,../data/cyp2c9/graphs/dist_thresh_15.graph,../data/cyp2c9/graphs/dist_thresh_17.graph
--encoding 编码方式
one_hot,aa_index
--batch_size
8,16,32
--learning_rate
0.001,0.0001,0.00001
--log_dir_base
output/training_logs
--epochs
200
--early_stopping
--train_size
0.6
--tune_size
0.2
--test_size
0.2
--split_rseed
9
    """

    parser = get_parser()
    parsed_args = parser.parse_args()
    if parsed_args.dataset_name == "":
        if parsed_args.dataset_file == "" or parsed_args.wt_coaa == "" or parsed_args.wt_ofs == "":
            parser.error("you must specify either a dataset_name (for a dataset defined in constants.py) or "
                         "all three of the dataset_file, the wt_aa, and the wt_ofs. if you specify the dataset_name,"
                         "it takes priority over all the other args.")
    mai(parsed_args)
