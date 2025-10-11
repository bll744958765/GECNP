import torch
import torch.nn as nn


# 优化后的代码
class MoranIndexOptimized(nn.Module):
    def __init__(self, k, decay_rate):
        super(MoranIndexOptimized, self).__init__()
        self.k = k
        self.decay_rate = decay_rate

    def forward(self, target_x, target_y, mu):

        target_x = target_x.squeeze(0)
        target_y = target_y.squeeze(0).squeeze(-1)
        coordinates = target_x[..., 0:2]
        mu = mu.squeeze(0).squeeze(-1)
        mean_y = torch.mean(target_y)
        mean_mu = torch.mean(mu)
        # 计算距离矩阵
        dist_matrix = torch.cdist(coordinates, coordinates, p=2)

        # 获取每个点的k个最近邻的索引
        knn_indices = torch.argsort(dist_matrix, dim=1)[:, :self.k]

        # 根据最近邻索引获取相应的距离值
        knn_distances = torch.gather(dist_matrix, 1, knn_indices)

        # 计算权重矩阵
        weight_matrix = torch.exp(-self.decay_rate * knn_distances)

        # 计算莫兰指数的分子
        deviation1 = target_y - mean_y
        numerator1 = torch.sum(weight_matrix * deviation1.unsqueeze(1) * deviation1[knn_indices])  # 计算莫兰指数的分母
        denominator1 = torch.sum((target_y - mean_y) ** 2)
        moran_y = target_y.shape[0] / torch.sum(weight_matrix) * numerator1 / denominator1  # 与原始公式一致

        deviation2 = mu - mean_mu
        numerator2 = torch.sum(weight_matrix * deviation2.unsqueeze(1) * deviation2[knn_indices])  # 计算莫兰指数的分母
        denominator2 = torch.sum((mu - mean_mu) ** 2)
        moran_mu = mu.shape[0] / torch.sum(weight_matrix) * numerator2 / denominator2  # 与原始公式一致
        return moran_y, moran_mu

# # 使用较小的数据集进行测试
# torch.manual_seed(42)
# target_x = torch.randn(1, 10, 3)  # 10个点的坐标 (1, 10, 3)
# target_y = torch.randn(1, 10, 1)  # 对应的y值 (1, 10, 1)
# mu = torch.randn(1, 10, 1)  # 对应的y值 (1, 10, 1)
# # 定义两个模型
# # original_model = MoranIndexOriginal(k=3, decay_rate=0.1)
# optimized_model = MoranIndexOptimized(k=3, decay_rate=0.1)
#
# # 计算莫兰指数
# # moran_original = original_model(target_x, target_y)
# moran_optimized,moran_optimized_mu = optimized_model(target_x, target_y,mu)
#
# # print(f"Original Moran Index: {moran_original}")
# print(f"Optimized Moran Index: {moran_optimized}")
# print(f"Optimized Moran Index: {moran_optimized_mu}")
