import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from attention import *

import torch.nn.functional as F


# 定义欧式距离专家
class EuclideanExpert1(nn.Module):

    def __init__(self, hidden_dim, distance):
        super(EuclideanExpert1, self).__init__()

        # self.final_projection = nn.Linear(num_hidden, num_hidden)
        self.distance = distance

        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, encoder1_input, query1):
        if self.distance:
            euclidean_distance = torch.norm(encoder1_input - query1, p=2, dim=-1)
            inverse_distances = 1.0 / (euclidean_distance + 1e-8)
            rep = inverse_distances.unsqueeze(-1) * query1
            rep = self.final_projection(rep)  # (bs,nt,hidden_number)
        else:
            rep = self.final_projection(encoder1_input)  # (bs,nt,hidden_number)
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
        if self.Similarity1:
            cosine_similarity = F.cosine_similarity(encoder1_input, query1, dim=-1)
            rep = cosine_similarity.unsqueeze(-1) * query1
            rep = self.final_projection(rep)
        else:
            rep=self.final_projection(encoder1_input)
        return rep


# 定义门控机制
class GatingNetwork1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatingNetwork1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 两个专家
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        gate_weights = torch.softmax(self.fc2(x), dim=1)  # 计算每个专家的权重
        return gate_weights


# 定义混合专家模型
class MixtureOfExpertsModel1(nn.Module):
    def __init__(self, input_dim, hidden_dim, distance, Similarity1):
        super(MixtureOfExpertsModel1, self).__init__()
        self.distance = distance
        self.Similarity1 = Similarity1
        self.euclidean_expert = EuclideanExpert1(hidden_dim, self.distance)
        self.similarity_expert = SimilarityExpert1(hidden_dim, self.Similarity1)
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


class GaussianComponent(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp_mu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )
        self.mlp_sigma = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 保证σ为正
        )

    def forward(self, x):
        mu = self.mlp_mu(x)
        sigma = self.mlp_sigma(x) + 1e-4
        return mu, sigma


class MixtureDensityNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_components):
        super().__init__()
        self.num_components = num_components
        # 多个 MLP 学习每个分量的 μ 和 σ
        self.components = nn.ModuleList([
            GaussianComponent(input_dim, hidden_dim) for _ in range(num_components)
        ])

        # 门控网络输出 α（softmax 保证为概率）
        self.gating = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_components),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 获取每个分量的 μ 和 σ
        mus = []
        sigmas = []
        for comp in self.components:
            mu, sigma = comp(x)
            mus.append(mu)
            sigmas.append(sigma)

        mu = torch.cat(mus, dim=1)  # (B, K)
        sigma = torch.cat(sigmas, dim=1)  # (B, K)

        # 获取 α
        alpha = self.gating(x)  # (B, K)

        return mu, sigma, alpha


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily


def build_mixture_distribution(mu, sigma, alpha):
    # mu, sigma, alpha: shape = (B, K)
    mixture = Categorical(alpha)
    components = Normal(mu, sigma)
    gmm = MixtureSameFamily(mixture_distribution=mixture, component_distribution=components)
    return gmm


# 定义门控机制
class GatingNetwork2(nn.Module):
    def __init__(self, input_dim):
        super(GatingNetwork2, self).__init__()
        self.fc = nn.Linear(input_dim, 3)  # 两个专家

    def forward(self, x):
        gate_weights = torch.softmax(self.fc(x), dim=1)  # 计算每个专家的权重
        return gate_weights


class MixtureOfExpertsModel2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_components):
        super(MixtureOfExpertsModel2, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.num_components = num_components
        self.MixtureDensity = MixtureDensityNetwork(hidden_dim, hidden_dim, num_components=self.num_components)
        self.gating_network2 = GatingNetwork2(hidden_dim)
        # self.fc_output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.linear(x)
        #  # 推理
        mu, sigma, alpha = self.MixtureDensity(x)  # 每个为 (1, 3)

        # gmm = build_mixture_distribution(mu, sigma, alpha)
        #
        # # # # 采样一个 y*
        # # # y_sample = gmm.sample()  # shape: (1,)
        # #
        # # # 或者获取期望值作为预测
        # expected_y = (alpha * mu).sum(dim=1, keepdim=True)

        return mu, sigma, alpha
