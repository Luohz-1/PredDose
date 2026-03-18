import torch
import torch.nn as nn
import math
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
import torch.nn.functional as F


class NodeAverageGC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Wc = nn.Linear(in_channels, out_channels)
        self.Wn = nn.Linear(in_channels, out_channels)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # He初始化
        nn.init.kaiming_normal_(self.Wc.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.Wn.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x: (batch_size, num_nodes, in_channels)
        adj: (num_nodes, num_nodes), 非归一化的邻接矩阵
        """
        # 中心信号
        h_center = self.Wc(x)  # [B, N, C_out]

        # 邻居信号
        h_neighbor = self.Wn(x)  # [B, N, C_out]
        aggregated = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), h_neighbor)  # 将 adj 变量扩展到与 batch 相同的维度

        # 度数归一化
        degree = adj.sum(1).clamp(min=1).view(1, -1, 1)
        h_neighbor = aggregated / degree

        # 合并信号
        out = h_center + h_neighbor + self.bias
        return F.leaky_relu(out, 0.2)




"""
以下为可训练双倍型的模型结构
"""

class GCNdp1(nn.Module): # GCN
    def __init__(self, num_features, vk, adj_matrix):
        super().__init__()
        self.adj = adj_matrix
        self.W1 = nn.Linear(vk, num_features)
        self.W2 = nn.Linear(vk, num_features)
        self.gc1_1 = NodeAverageGC(num_features, 64)
        self.gc1_2 = NodeAverageGC(64, 64)
        self.gc1_3 = NodeAverageGC(64, 64)
        self.gc1_4 = NodeAverageGC(64, 64)
        self.gc1_5 = NodeAverageGC(64, 64)

        self.gc2_1 = NodeAverageGC(num_features, 64)
        self.gc2_2 = NodeAverageGC(64, 64)
        self.gc2_3 = NodeAverageGC(64, 64)
        self.gc2_4 = NodeAverageGC(64, 64)
        self.gc2_5 = NodeAverageGC(64, 64)


        self.fc = None

    def forward(self, x, vk_data):
        # 提取每个等位基因的数据
        x_dp1 = x[:, 0, :, :]  # (batch, 490, 40)
        x_dp2 = x[:, 1, :, :]  # (batch, 490, 40)

        # 处理第一个等位基因
        x_vk_1 = self.W1(vk_data)
        x_vk_1 = x_vk_1.unsqueeze(1)    # (batch, 1, 40)
        x_dp1 = torch.cat((x_dp1, x_vk_1), dim=1)  # (batch, 491, num_features)
        x_dp1 = self.gc1_1(x_dp1, self.adj)
        x_dp1 = self.gc1_2(x_dp1, self.adj) # (batch, 491, 64)
        x_dp1 = self.gc1_3(x_dp1, self.adj) # (batch, 491, 64)
        x_dp1 = self.gc1_4(x_dp1, self.adj) # (batch, 491, 64)
        x_dp1 = self.gc1_5(x_dp1, self.adj) # (batch, 491, 64)

        # 处理第二个等位基因
        x_vk_2 = self.W2(vk_data)
        x_vk_2 = x_vk_2.unsqueeze(1)
        x_dp2 = torch.cat((x_dp2, x_vk_2), dim=1)  # (batch, 491, num_features)
        x_dp2 = self.gc2_1(x_dp2, self.adj)
        x_dp2 = self.gc2_2(x_dp2, self.adj) # (batch, 491, 64)
        x_dp2 = self.gc2_3(x_dp2, self.adj) # (batch, 491, 64)
        x_dp2 = self.gc2_4(x_dp2, self.adj) # (batch, 491, 64)
        x_dp2 = self.gc2_5(x_dp2, self.adj) # (batch, 491, 64)


        # 合并两个等位基因的特征
        x = torch.cat((x_dp1, x_dp2), dim=1)  # (batch, 982, 64)

        # 全连接层进行预测
        if self.fc is None:
            device = next(self.parameters()).device
            self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(x.shape[1] * x.shape[2], 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(100, 1)
        ).to(device)
        x = self.fc(x)
        return x

class GCNdp2(nn.Module):  # GCN + Attention 0.55
    def __init__(self, num_features, vk, adj_matrix):
        super().__init__()
        self.adj = adj_matrix
        self.W1 = nn.Linear(vk, num_features)
        self.W2 = nn.Linear(vk, num_features)
        self.gc1_1 = NodeAverageGC(num_features, 64)
        self.gc1_2 = NodeAverageGC(64, 32)
        self.gc2_1 = NodeAverageGC(num_features, 64)
        self.gc2_2 = NodeAverageGC(64, 32)

        # 新增GlobalAttention层
        self.att1 = CustomAttentionalAggregation(nn.Sequential(
        nn.Linear(32, 32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 1)
    ))  # 用于等位基因1
        self.att2 = CustomAttentionalAggregation(nn.Sequential(
        nn.Linear(32, 32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 1)
    ))  # 用于等位基因2

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 32 * adj_matrix.shape[0], 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x, vk_data):
        # 原始输入x形状: (batch, 2, 490, 40)
        x_dp1 = x[:, 0, :, :]  # (batch, 490, 40)
        x_dp2 = x[:, 1, :, :]  # (batch, 490, 40)

        # 处理第一个等位基因
        x_vk_1 = self.W1(vk_data).unsqueeze(1)  # (batch, 1, 40)
        x_dp1 = torch.cat((x_dp1, x_vk_1), dim=1)  # (batch, 491, 40)
        x_dp1 = self.gc1_1(x_dp1, self.adj)  # (batch, 491, 64)
        x_dp1 = self.gc1_2(x_dp1, self.adj)  # (batch, 491, 32)

        # 新增：全局注意力聚合
        x_dp1 = self.att1(x_dp1).squeeze(1)   # 形状从 (batch,491,32) -> (batch,32)

        # 处理第二个等位基因
        x_vk_2 = self.W2(vk_data).unsqueeze(1)
        x_dp2 = torch.cat((x_dp2, x_vk_2), dim=1)  # (batch, 491, 40)
        x_dp2 = self.gc2_1(x_dp2, self.adj)  # (batch, 491, 64)
        x_dp2 = self.gc2_2(x_dp2, self.adj)  # (batch, 491, 32)

        # 新增：全局注意力聚合
        x_dp2 = self.att2(x_dp2).squeeze(1)   # 形状从 (batch,491,32) -> (batch,32)

        # 合并两个等位基因的特征
        x = torch.cat((x_dp1, x_dp2), dim=1)  # (batch, 64) = (batch, 32*2)

        # 全连接层进行预测
        x = self.fc(x)  # (batch, 1)
        return x

class CustomAttentionalAggregation(nn.Module):
    def __init__(self, gate_nn):
        super(CustomAttentionalAggregation, self).__init__()
        self.gate_nn = gate_nn  # 门控网络

    def forward(self, x):
        """
        x: (batch_size, num_nodes, in_channels)
        """
        # 计算门控分数
        gate_scores = self.gate_nn(x).squeeze(-1)  # (batch_size, num_nodes)

        # 归一化门控分数
        attention_weights = F.softmax(gate_scores, dim=1)  # (batch_size, num_nodes)

        # 应用注意力权重
        attention_weights = attention_weights.unsqueeze(-1)  # (batch_size, num_nodes, 1)
        weighted_features = x * attention_weights  # (batch_size, num_nodes, in_channels)

        return weighted_features




class GCNdp3(nn.Module):    # GAT 0.65
    def __init__(self, num_features, vk, adj_matrix):
        super().__init__()
        # 将邻接矩阵转换为 PyG 需要的 edge_index 格式
        self.edge_index = adj_matrix.nonzero().t().contiguous() # (2, num_edges)

        # 线性变换层（处理 vk_data）
        self.W1 = nn.Linear(vk, num_features)
        self.W2 = nn.Linear(vk, num_features)

        # 使用 PyG 的 GATConv 替换 GCN 层
        self.gat1_1 = GATConv(num_features, 64, heads=1)  # 单头注意力
        self.gat1_2 = GATConv(64, 64, heads=1)

        self.gat2_1 = GATConv(num_features, 64, heads=1)
        self.gat2_2 = GATConv(64, 64, heads=1)


        # 全连接层（保持不变）
        self.fc = None

    def forward(self, x, vk_data):
        """
        x: (batch_size, 2, num_nodes, num_features)
        vk_data: (batch_size, vk)
        adj_matrix: (num_nodes, num_nodes)
        """
        # 提取两个等位基因的数据
        x_dp1 = x[:, 0, :, :]  # (batch, 490, num_features)
        x_dp2 = x[:, 1, :, :]  # (batch, 490, num_features)

        # 处理 vk_data 并拼接
        x_vk_1 = self.W1(vk_data).unsqueeze(1)  # (batch, 1, num_features)
        x_vk_2 = self.W2(vk_data).unsqueeze(1)
        x_dp1 = torch.cat([x_dp1, x_vk_1], dim=1)  # (batch, 491, num_features)
        x_dp2 = torch.cat([x_dp2, x_vk_2], dim=1)

        batch_size, num_nodes, num_features = x_dp1.shape

        # 1. 展平节点特征: (batch, 3, 4) -> (batch*3, 4)
        x_dp1 = x_dp1.view(-1, num_features)
        x_dp2 = x_dp2.view(-1, num_features)
        # 2. 正确扩展 edge_index
        offsets = torch.arange(0, batch_size * num_nodes, num_nodes, device=x.device)
        expanded_edge_index = torch.cat([
            self.edge_index + offset for offset in offsets], dim=1)

        # 处理第一个等位基因（使用 GAT）
        x_dp1 = self.gat1_1(x_dp1, expanded_edge_index)  # (491, batch, 64)
        x_dp1 = F.leaky_relu(x_dp1, 0.2)
        x_dp1 = self.gat1_2(x_dp1, expanded_edge_index)  # (491, batch, 32)
        x_dp1 = F.leaky_relu(x_dp1, 0.2)

        x_dp1 = x_dp1.view(batch_size, num_nodes, -1)

        # 处理第二个等位基因
        x_dp2 = self.gat2_1(x_dp2,expanded_edge_index)
        x_dp2 = F.leaky_relu(x_dp2, 0.2)
        x_dp2 = self.gat2_2(x_dp2, expanded_edge_index)
        x_dp2 = F.leaky_relu(x_dp2, 0.2)

        x_dp2 = x_dp2.view(batch_size, num_nodes, -1)

        # 合并两个等位基因的特征
        x = torch.cat([x_dp1, x_dp2], dim=1)  # (batch, 982, 32)

        # 全连接层进行预测
        if self.fc is None:
            device = next(self.parameters()).device
            self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(x.shape[1] * x.shape[2], 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(100, 1)
        ).to(device)
        x = self.fc(x)
        return x


class GCNdp4(nn.Module):    # Graph Transformer 0.88
    def __init__(self, num_features, vk, adj):
        super().__init__()
        # Store the adjacency matrix for edge_index
        self.edge_index = adj.nonzero().t().contiguous()  # (2, num_edges)

        # Linear transformation for vk_data (unchanged)
        self.W1 = nn.Linear(vk, num_features)
        self.W2 = nn.Linear(vk, num_features)

        # Replace GAT layers with Graph Transformer layers
        # Using TransformerConv from PyG which implements graph attention with positional encodings
        self.transformer1_1 = TransformerConv(num_features, 64, heads=1)
        self.transformer1_2 = TransformerConv(64, 32, heads=1)
        self.transformer2_1 = TransformerConv(num_features, 64, heads=1)
        self.transformer2_2 = TransformerConv(64, 32, heads=1)

        self.pos_enc = nn.Parameter(torch.randn(1, num_features))   # 创造一个张量，他可以被模型标记为可训练的参数

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 32 * 491, 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(100, 1)
        )

    def forward(self, x, vk_data):
        # Extract two alleles' data
        x_dp1 = x[:, 0, :, :]  # (batch, 490, num_features)
        x_dp2 = x[:, 1, :, :]  # (batch, 490, num_features)

        # Process vk_data and concatenate
        x_vk_1 = self.W1(vk_data).unsqueeze(1)  # (batch, 1, num_features)
        x_vk_2 = self.W2(vk_data).unsqueeze(1)
        x_dp1 = torch.cat([x_dp1, x_vk_1], dim=1)  # (batch, 491, num_features)
        x_dp2 = torch.cat([x_dp2, x_vk_2], dim=1)

        batch_size, num_nodes, num_features = x_dp1.shape

        # Add positional encoding
        x_dp1 = x_dp1 + self.pos_enc
        x_dp2 = x_dp2 + self.pos_enc

        # Reshape for graph processing
        x_dp1 = x_dp1.view(-1, num_features)  # (batch*491, num_features)
        x_dp2 = x_dp2.view(-1, num_features)

        # Expand edge_index for batch processing
        offsets = torch.arange(0, batch_size * num_nodes, num_nodes, device=x.device)
        expanded_edge_index = torch.cat([self.edge_index + offset for offset in offsets], dim=1)

        # Process first allele with Transformer
        x_dp1 = self.transformer1_1(x_dp1, expanded_edge_index)
        x_dp1 = F.leaky_relu(x_dp1, 0.2)
        x_dp1 = self.transformer1_2(x_dp1, expanded_edge_index)
        x_dp1 = x_dp1.view(batch_size, num_nodes, -1)

        # Process second allele with Transformer
        x_dp2 = self.transformer2_1(x_dp2, expanded_edge_index)
        x_dp2 = F.leaky_relu(x_dp2, 0.2)
        x_dp2 = self.transformer2_2(x_dp2, expanded_edge_index)
        x_dp2 = x_dp2.view(batch_size, num_nodes, -1)

        # Combine features from both alleles
        x = torch.cat([x_dp1, x_dp2], dim=1)  # (batch, 982, 32)

        # Final prediction
        x = self.fc(x)
        return x


class GCNdp5(torch.nn.Module):
    def __init__(self, num_features, vk, adj_matrix):
        super().__init__()
        # Store the adjacency matrix for edge_index
        self.edge_index = adj_matrix.nonzero().t().contiguous()  # (2, num_edges)

        # Linear transformation for vk_data (unchanged)
        self.W1 = nn.Linear(vk, num_features)
        self.W2 = nn.Linear(vk, num_features)

        # 增加更多的GCN层和更宽的维度
        self.conv1_1 = GCNConv(num_features, 64)
        self.conv1_2 = GCNConv(64, 64)
        self.conv1_3 = GCNConv(64, 64)
        self.conv1_4 = GCNConv(64, 64)
        self.conv1_5 = GCNConv(64, 64)

        self.conv2_1 = GCNConv(num_features, 64)
        self.conv2_2 = GCNConv(64, 64)
        self.conv2_3 = GCNConv(64, 64)
        self.conv2_4 = GCNConv(64, 64)
        self.conv2_5 = GCNConv(64, 64)

        self.fc = None

    def forward(self, x, vk_data):
        # Extract two alleles' data
        x_dp1 = x[:, 0, :, :]  # (batch, 490, num_features)
        x_dp2 = x[:, 1, :, :]  # (batch, 490, num_features)

        # Process vk_data and concatenate
        x_vk_1 = self.W1(vk_data).unsqueeze(1)  # (batch, 1, num_features)
        x_vk_2 = self.W2(vk_data).unsqueeze(1)
        x_dp1 = torch.cat([x_dp1, x_vk_1], dim=1)  # (batch, 491, num_features)
        x_dp2 = torch.cat([x_dp2, x_vk_2], dim=1)

        batch_size, num_nodes, num_features = x_dp1.shape

        # Reshape for graph processing
        x_dp1 = x_dp1.view(-1, num_features)  # (batch*491, num_features)
        x_dp2 = x_dp2.view(-1, num_features)

        # Expand edge_index for batch processing
        offsets = torch.arange(0, batch_size * num_nodes, num_nodes, device=x.device)
        expanded_edge_index = torch.cat([self.edge_index + offset for offset in offsets], dim=1)

        # Process first allele with Transformer
        x_dp1 = self.conv1_1(x_dp1, expanded_edge_index)
        x_dp1 = F.leaky_relu(x_dp1, 0.2)
        x_dp1 = self.conv1_2(x_dp1, expanded_edge_index)
        x_dp1 = F.leaky_relu(x_dp1, 0.2)
        x_dp1 = self.conv1_3(x_dp1, expanded_edge_index)
        x_dp1 = F.leaky_relu(x_dp1, 0.2)
        x_dp1 = self.conv1_4(x_dp1, expanded_edge_index)
        x_dp1 = F.leaky_relu(x_dp1, 0.2)
        x_dp1 = self.conv1_5(x_dp1, expanded_edge_index)
        x_dp1 = F.leaky_relu(x_dp1, 0.2)


        x_dp1 = x_dp1.view(batch_size, num_nodes, -1)

        # Process second allele with Transformer
        x_dp2 = self.conv2_1(x_dp2, expanded_edge_index)
        x_dp2 = F.leaky_relu(x_dp2, 0.2)
        x_dp2 = self.conv2_2(x_dp2, expanded_edge_index)
        x_dp2 = F.leaky_relu(x_dp2, 0.2)
        x_dp2 = self.conv2_3(x_dp2, expanded_edge_index)
        x_dp2 = F.leaky_relu(x_dp2, 0.2)
        x_dp2 = self.conv2_4(x_dp2, expanded_edge_index)
        x_dp2 = F.leaky_relu(x_dp2, 0.2)
        x_dp2 = self.conv2_5(x_dp2, expanded_edge_index)
        x_dp2 = F.leaky_relu(x_dp2, 0.2)

        x_dp2 = x_dp2.view(batch_size, num_nodes, -1)

        # Combine features from both alleles
        x = torch.cat([x_dp1, x_dp2], dim=1)  # (batch, 982, 32)

        # 全连接层进行预测
        if self.fc is None:
            device = next(self.parameters()).device
            self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(x.shape[1] * x.shape[2], 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(100, 1)
        ).to(device)
        x = self.fc(x)
        return x


class GCNdp6(nn.Module):    # 序列卷积 0.91
    def __init__(self, num_features=40, kernel_size=3):
        super().__init__()

        # 卷积层
        self.conv1_1 = nn.Conv1d(in_channels=num_features, out_channels=128, kernel_size=kernel_size, padding=1)
        self.conv1_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=1)
        self.conv1_3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=1)

        self.conv2_1 = nn.Conv1d(in_channels=num_features, out_channels=128, kernel_size=kernel_size, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=1)
        self.conv2_3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=1)

        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2)

        # 全连接层1

        # self.fc = None
        self.fc1 = None
        self.fc2 = None
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.1)

        self.bn1_1 = nn.BatchNorm1d(128)
        self.bn1_2 = nn.BatchNorm1d(128)
        self.bn1_3 = nn.BatchNorm1d(128)
        # self.bn1_4 = nn.BatchNorm1d(64)
        self.bn2_1 = nn.BatchNorm1d(128)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.bn2_3 = nn.BatchNorm1d(128)
        # self.bn2_4 = nn.BatchNorm1d(64)

    def forward(self, x, return_features=False):
        # Extract two alleles' data
        x_dp1 = x[:, 0, :, :]  # (batch, 490, num_features)
        x_dp2 = x[:, 1, :, :]  # (batch, 490, num_features)

        x_dp1 = x_dp1.permute(0, 2, 1)  # torch 卷积的形式是 （batch， features， sequences
        x_dp2 = x_dp2.permute(0, 2, 1)

        # Process first allele
        x_dp1_1 = self.bn1_1(self.conv1_1(x_dp1))
        x_dp1_rule1 = F.leaky_relu(x_dp1_1, 0.2)
        x_dp1_rule1 = self.pool(x_dp1_rule1)

        x_dp1_2 = self.bn1_2(self.conv1_2(x_dp1_rule1))
        x_dp1_rule2 = F.leaky_relu(x_dp1_2, 0.2)
        x_dp1_rule2 = self.pool(x_dp1_rule2)


        x_dp1_3 = self.bn1_3(self.conv1_3(x_dp1_rule2))
        x_dp1_rule3 = F.leaky_relu(x_dp1_3, 0.2)
        x_dp1_rule3 = self.pool(x_dp1_rule3)

        # Process second allele
        x_dp2_1 = self.bn2_1(self.conv2_1(x_dp2))
        x_dp2_rule1 = F.leaky_relu(x_dp2_1, 0.2)
        x_dp2_rule1 = self.pool(x_dp2_rule1)

        x_dp2_2 = self.bn2_2(self.conv2_2(x_dp2_rule1))
        x_dp2_rule2 = F.leaky_relu(x_dp2_2, 0.2)
        x_dp2_rule2 = self.pool(x_dp2_rule2)


        x_dp2_3 = self.bn2_3(self.conv2_3(x_dp2_rule2))
        x_dp2_rule3 = F.leaky_relu(x_dp2_3, 0.2)
        x_dp2_rule3 = self.pool(x_dp2_rule3)
        # Combine features from both alleles
        x_final = torch.cat([x_dp1_rule3, x_dp2_rule3], dim=1)

        # Final prediction
        if self.fc1 is None:
            device = next(self.parameters()).device
            self.fc1 = nn.Linear(x_final.shape[1] * x_final.shape[2], 100).to(device)
            self.fc2 = nn.Linear(100, 1).to(device)


        x_final = self.flatten(x_final)
        x_100 = self.fc1(x_final)
        x_final = F.leaky_relu(x_100, 0.2)
        x_final = self.dropout(x_final)
        x_final = self.fc2(x_final)


        if return_features:
            # Return features from three different layers (after pooling operations)
            features = {
                'layer1': x_dp1_1,
                'layer2': x_dp1_2,
                'layer3': x_dp1_3,
                'layer_cat': x_100
            }
            return x_final, features

        return x_final


class GCNdp7(nn.Module):    # 线性回归 0.87
    def __init__(self, num_features=40, kernel_size=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(980 * 43, 1),
        )

    def forward(self, x):
        # x: (batch_size, 490, 40) -> (batch_size, 40, 490) (调整维度以适应Conv1d)
        # Extract two alleles' data
        x_dp1 = x[:, 0, :, :]  # (batch, 490, num_features)
        x_dp2 = x[:, 1, :, :]  # (batch, 490, num_features)

        # Combine features from both alleles
        x = torch.cat([x_dp1, x_dp2], dim=1)

        # Final prediction
        x = self.fc(x)

        return x



# 序列卷积
class GCNdp8(nn.Module):    # 全连接网络 0.81
    def __init__(self, num_features=40, kernel_size=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(980 * 43
                      , 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(100, 1)
        )   # 一层


    def forward(self, x):
        # x: (batch_size, 490, 40) -> (batch_size, 40, 490) (调整维度以适应Conv1d)
        # Extract two alleles' data
        x_dp1 = x[:, 0, :, :]  # (batch, 490, num_features)
        x_dp2 = x[:, 1, :, :]  # (batch, 490, num_features)

        # Combine features from both alleles
        x = torch.cat([x_dp1, x_dp2], dim=1)

        # Final prediction
        x = self.fc(x)

        return x



"""
专利模型
"""


class ConvBlock1D(nn.Module):
    """基础卷积块"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(ConvBlock1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ReductionBlock(nn.Module):
    """降维块，用于减少序列长度"""

    def __init__(self, in_channels, out_channels):
        super(ReductionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            ConvBlock1D(in_channels, out_channels // 2, kernel_size=1),
            ConvBlock1D(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1)
        )

        self.branch2 = nn.Sequential(
            ConvBlock1D(in_channels, out_channels // 2, kernel_size=1),
            ConvBlock1D(out_channels // 2, out_channels, kernel_size=5, stride=2, padding=2)
        )

        self.branch3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.branch3_conv = ConvBlock1D(in_channels, out_channels, kernel_size=1)

        self.bn = nn.BatchNorm1d(out_channels * 3)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3_conv(self.branch3(x))

        outputs = torch.cat([branch1, branch2, branch3], dim=1)
        outputs = self.bn(outputs)
        return F.relu(outputs)


class InceptionResNetBlockA(nn.Module):
    """Inception-ResNet-A块"""

    def __init__(self, in_channels, scale=0.1): # scale 缩放因子
        super(InceptionResNetBlockA, self).__init__()
        self.scale = scale

        # 分支1: 1x1卷积
        self.branch1 = ConvBlock1D(in_channels, 32, kernel_size=1)

        # 分支2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            ConvBlock1D(in_channels, 32, kernel_size=1),
            ConvBlock1D(32, 48, kernel_size=3, padding=1)
        )

        # 分支3: 1x1 -> 3x3 -> 3x3
        self.branch3 = nn.Sequential(
            ConvBlock1D(in_channels, 32, kernel_size=1),
            ConvBlock1D(32, 48, kernel_size=3, padding=1),
            ConvBlock1D(48, 64, kernel_size=3, padding=1)
        )

        # 合并后的1x1卷积
        self.conv = nn.Conv1d(144, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        identity = x

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        # 拼接分支输出
        outputs = torch.cat([branch1, branch2, branch3], dim=1)

        # 1x1卷积调整维度
        outputs = self.conv(outputs)
        outputs = self.bn(outputs)

        # 添加缩放残差连接
        outputs = outputs * self.scale
        outputs = outputs + identity

        return F.relu(outputs)


class InceptionResNetBlockB(nn.Module):
    """Inception-ResNet-B块"""

    def __init__(self, in_channels, out_channels, scale=0.1):
        super(InceptionResNetBlockB, self).__init__()
        self.scale = scale

        # 分支1: 1x1卷积
        self.branch1 = ConvBlock1D(in_channels, out_channels // 4, kernel_size=1)

        # 分支2: 1x1 -> 1x7 -> 7x1 (近似7x7)
        self.branch2 = nn.Sequential(
            ConvBlock1D(in_channels, out_channels // 4, kernel_size=1),
            ConvBlock1D(out_channels // 4, out_channels // 4, kernel_size=7, padding=3)
        )

        # 合并后的1x1卷积
        self.conv = nn.Conv1d(out_channels // 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)

        # 如果输入输出维度不同，需要投影残差连接
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity(),
            nn.BatchNorm1d(out_channels)
        )



    def forward(self, x):
        identity = self.shortcut(x)

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)

        # 拼接分支输出
        outputs = torch.cat([branch1, branch2], dim=1)

        # 1x1卷积调整维度
        outputs = self.conv(outputs)
        outputs = self.bn(outputs)

        # 添加缩放残差连接
        outputs = outputs * self.scale
        outputs = outputs + identity
        outputs = F.relu(outputs)
        return outputs


class InceptionResNetBlockC(nn.Module):
    """Inception-ResNet-C块"""

    def __init__(self, in_channels, scale=0.1):
        super(InceptionResNetBlockC, self).__init__()
        self.scale = scale

        # 分支1: 1x1卷积
        self.branch1 = ConvBlock1D(in_channels, 576, kernel_size=1)

        # 分支2: 1x1 -> 1x3 -> 3x1 (近似3x3)
        self.branch2 = nn.Sequential(
            ConvBlock1D(in_channels, 864, kernel_size=1),
            ConvBlock1D(864, 576, kernel_size=3, padding=1)
        )

        # 合并后的1x1卷积
        self.conv = nn.Conv1d(1152, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        identity = x

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)

        # 拼接分支输出
        outputs = torch.cat([branch1, branch2], dim=1)

        # 1x1卷积调整维度
        outputs = self.conv(outputs)
        outputs = self.bn(outputs)

        # 添加缩放残差连接
        outputs = outputs * self.scale
        outputs = outputs + identity

        return F.relu(outputs)


class SqueezeExcitation(nn.Module):
    """压缩与激励模块"""

    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class DualBranchInceptionResNet(nn.Module):
    """双分支Inception-ResNet回归网络"""

    def __init__(self, input_channels=43, num_blocks_A=5, num_blocks_B=10, num_blocks_C=5):
        super(DualBranchInceptionResNet, self).__init__()

        # 茎网络（stem）
        self.stem = nn.Sequential(
            ConvBlock1D(input_channels, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock1D(32, 32, kernel_size=3, padding=1),
            ConvBlock1D(32, 64, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        )

        # 分支1
        self.branch1_blockA = self._make_layer(InceptionResNetBlockA, 64, num_blocks_A)
        self.branch1_reduction = ReductionBlock(64, 128)
        self.branch1_blockB = self._make_layer(lambda in_c: InceptionResNetBlockB(in_c, 384), 384, num_blocks_B)
        self.branch1_reduction2 = ReductionBlock(384, 384)
        self.branch1_blockC = self._make_layer(InceptionResNetBlockC, 1152, num_blocks_C)
        self.branch1_se = SqueezeExcitation(1152)

        # 分支2（与分支1结构相同，但参数独立）
        self.branch2_blockA = self._make_layer(InceptionResNetBlockA, 64, num_blocks_A)
        self.branch2_reduction = ReductionBlock(64, 128)
        self.branch2_blockB = self._make_layer(lambda in_c: InceptionResNetBlockB(in_c, 384), 384, num_blocks_B)
        self.branch2_reduction2 = ReductionBlock(384, 384)
        self.branch2_blockC = self._make_layer(InceptionResNetBlockC, 1152, num_blocks_C)
        self.branch2_se = SqueezeExcitation(1152)

        # 特征融合模块
        self.fusion = FeatureFusionModule(1152 * 2, 256)

        # 回归头
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._initialize_weights()

    def _make_layer(self, block, in_channels, num_blocks):
        """创建连续的块"""
        layers = []
        for i in range(num_blocks):
            layers.append(block(in_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播
        x: 输入张量，形状为 (batch_size, 2, 490, 43)
        """
        batch_size = x.size(0)

        # 分离两个分支的输入
        x1 = x[:, 0, :, :].permute(0, 2, 1)  # (batch, 43, 490)
        x2 = x[:, 1, :, :].permute(0, 2, 1)  # (batch, 43, 490)

        # 分别通过茎网络
        x1 = self.stem(x1)
        x2 = self.stem(x2)

        # 分支1处理
        x1 = self.branch1_blockA(x1)  # (batch, 64, 122)
        x1 = self.branch1_reduction(x1)  # (batch, 384, 61)
        x1 = self.branch1_blockB(x1)  # (batch, 384, 61)
        x1 = self.branch1_reduction2(x1)  # (batch, 1152, 31)
        x1 = self.branch1_blockC(x1)  # (batch, 1152, 31)
        x1 = self.branch1_se(x1)  # (batch, 1152, 31)

        # 分支2处理
        x2 = self.branch2_blockA(x2)
        x2 = self.branch2_reduction(x2)
        x2 = self.branch2_blockB(x2)
        x2 = self.branch2_reduction2(x2)
        x2 = self.branch2_blockC(x2)
        x2 = self.branch2_se(x2)

        # 特征融合
        combined = torch.cat([x1, x2], dim=1)  # (batch, 1152, 31)
        fused = self.fusion(combined)  # (batch, 100, 31)

        # 回归预测
        output = self.regression_head(fused)

        return output


class FeatureFusionModule(nn.Module):
    """特征融合模块"""

    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()

        self.conv1 = ConvBlock1D(in_channels, out_channels, kernel_size=1)
        self.conv2 = ConvBlock1D(out_channels, out_channels, kernel_size=3, padding='same')

        self.se = SqueezeExcitation(out_channels)
        self.conv_final = ConvBlock1D(out_channels, out_channels, kernel_size=1)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.conv_final(out)

        return F.relu(out + identity)



# 模型工厂函数
def create_inception_resnet(model_type='standard', **kwargs):

    models = {
        'standard': DualBranchInceptionResNet,
    }

    return models[model_type](**kwargs)



if __name__ == "__main__":
    pass