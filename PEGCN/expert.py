import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from attention import *

import torch.nn.functional as F


# 定义欧式距离专家
class EuclideanExpert1(nn.Module):

    def __init__(self, hidden_dim, Laplace):
        super(EuclideanExpert1, self).__init__()

        # self.final_projection = nn.Linear(num_hidden, num_hidden)
        self.Laplace = Laplace

        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, encoder1_input, query1):
        # if self.Laplace:
        euclidean_distance = torch.norm(encoder1_input - query1, p=2, dim=-1)
        inverse_distances = 1.0 / (euclidean_distance + 1e-8)
        rep = inverse_distances.unsqueeze(-1) * query1
        rep = self.final_projection(rep)  # (bs,nt,hidden_number)
        return rep


# 定义相似度专家
class SimilarityExpert1(nn.Module):
    def __init__(self, hidden_dim, Similarity1):
        super(SimilarityExpert1, self).__init__()
        self.Similarity1 = Similarity1
        # self.final_projection = nn.Linear(128, 128)

        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, encoder1_input, query1):
        cosine_similarity = F.cosine_similarity(encoder1_input, query1, dim=-1)
        rep = cosine_similarity.unsqueeze(-1) * query1
        rep = self.final_projection(rep)
        return rep


# 定义门控机制
class GatingNetwork1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatingNetwork1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 两个专家
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        gate_weights = torch.softmax(self.fc2(x), dim=1)  # 计算每个专家的权重
        return gate_weights


# 定义混合专家模型
class MixtureOfExpertsModel1(nn.Module):
    def __init__(self, input_dim, hidden_dim, laplace, Similarity1):
        super(MixtureOfExpertsModel1, self).__init__()
        self.laplace = laplace
        self.Similarity1 = Similarity1
        self.euclidean_expert = EuclideanExpert1(hidden_dim, self.laplace)
        self.similarity_expert = SimilarityExpert1(hidden_dim,self.Similarity1)
        self.gating_network = GatingNetwork1(input_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, 1)

        self.input_projection1 = nn.Linear(input_dim + 1, hidden_dim)
        self.target_projection1 = nn.Linear(input_dim, hidden_dim)

        # torch.nn.init.xavier_uniform_(self.input_projection1.weight)

    def forward(self, context_x, context_y, target_x):
        bs, nt, _ = target_x.size()
        encoder_input = torch.cat([context_x, context_y], dim=-1)  # concat context location (x), context value (y)
        query = self.target_projection1(target_x)  # (bs,nt,12)--> (bs,nt,num_hidden)
        # if torch.isnan(context_x).any() or torch.isinf(context_x).any():
        #     print("Input contains NaN or Inf values!")
        encoder1_input = self.input_projection1(encoder_input)  # (bs,nc,13)--> (bs,nc,num_hidden)
        encoder1_input = torch.mean(encoder1_input, dim=1).repeat(1, nt, 1)

        euclidean_output = self.euclidean_expert(encoder1_input, query)
        similarity_output = self.similarity_expert(encoder1_input, query)

        # 计算门控权重
        # xy = torch.cat([context_x, context_y], dim=-1)
        gate_weights = self.gating_network(target_x)
        # tt=gate_weights[..., 0].unsqueeze(-1) * euclidean_output
        # ttt=gate_weights[..., 1].unsqueeze(-1) * similarity_output
        # 加权输出
        expert_output = gate_weights[..., 0].unsqueeze(-1) * euclidean_output \
                        + gate_weights[..., 1].unsqueeze(-1) * similarity_output

        # 最终通过全连接层输出
        # output = self.fc_output(expert_output)
        # return expert_output
        return expert_output


class EuclideanExpert2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EuclideanExpert2, self).__init__()
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x = self.relu(self.fc1(x))
        self.fc2(self.dropout(self.relu(x)))
        return x


# 定义相似度专家
class SimilarityExpert2(nn.Module):
    def __init__(self, input_dim, num_hidden):
        super(SimilarityExpert2, self).__init__()
        # self.fc1 = nn.Linear(input_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x = self.relu(self.fc1(x))
        self.fc2(self.dropout(self.relu(x)))
        return x


# 定义门控机制
class GatingNetwork2(nn.Module):
    def __init__(self, input_dim):
        super(GatingNetwork2, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # 两个专家

    def forward(self, x):
        gate_weights = torch.softmax(self.fc(x), dim=1)  # 计算每个专家的权重
        return gate_weights


class MixtureOfExpertsModel2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MixtureOfExpertsModel2, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.euclidean_expert2 = EuclideanExpert2(hidden_dim, hidden_dim)
        self.similarity_expert2 = SimilarityExpert2(hidden_dim, hidden_dim)
        self.gating_network2 = GatingNetwork2(hidden_dim)
        # self.fc_output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.linear(x)
        # 计算两个专家的输出
        euclidean_output = self.euclidean_expert2(x)
        similarity_output = self.similarity_expert2(x)
        # 计算门控权重
        gate_weights = self.gating_network2(x)
        # 加权输出
        expert_output = gate_weights[..., 0].unsqueeze(1) * euclidean_output + \
                        gate_weights[..., 1].unsqueeze(1) * similarity_output
        # euclidean_output = gate_weights[..., 0].unsqueeze(1) * euclidean_output
        # similarity_output = gate_weights[..., 1].unsqueeze(1) * similarity_output
        # 最终通过全连接层输出
        # output = self.fc_output(expert_output)
        return expert_output
