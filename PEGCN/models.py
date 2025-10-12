# -*- coding : utf-8 -*-

"""
实现spatial Neural Process
"""

import torch
import torch.nn as nn
from expert import MixtureOfExpertsModel1
from expert import MixtureOfExpertsModel2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from mm import MoranIndexOptimized


class DeterministicEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """

    def __init__(self, input_dim, num_hidden, encMoe, distance, Similarity1):
        super(DeterministicEncoder, self).__init__()

        self.input_projection3 = nn.Linear(input_dim + 1, num_hidden)
        self.encMoe = encMoe
        self.distance = distance
        self.Similarity1 = Similarity1
        self.expert = MixtureOfExpertsModel1(input_dim, num_hidden, self.distance, self.Similarity1)
        self.final_projection = nn.Sequential(
            nn.Linear(num_hidden, int(num_hidden / 2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(int(num_hidden / 2), num_hidden),
        )
        self.final_projection_shared = nn.Sequential(
            nn.Linear(2 * num_hidden, int(num_hidden / 2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(int(num_hidden / 2), num_hidden),
        )
        self.linear = nn.Linear(num_hidden, num_hidden)

    def forward(self, context_x, context_y, target_x):

        bs, nt, _ = target_x.size()
        if self.encMoe:
            NO_shared_input = self.expert(context_x, context_y, target_x)
            shared_input = torch.cat([context_x, context_y], dim=-1)
            shared_input = self.input_projection3(shared_input)
            shared_input = torch.mean(shared_input, dim=1).repeat(1, nt, 1)
            shared_input = self.linear(shared_input)
            encoder_input = torch.cat([shared_input, NO_shared_input], dim=-1)
            expert_out = self.final_projection_shared(encoder_input)
        else:
            encoder3_input = torch.cat([context_x, context_y], dim=-1)
            encoder3_input = self.input_projection3(encoder3_input)
            encoder_input = torch.mean(encoder3_input, dim=1).repeat(1, nt, 1)
            encoder_input = torch.mean(encoder_input)
            encoder_input = self.linear(encoder_input)
            expert_out = self.final_projection(encoder_input)

        return expert_out


class Decoder(nn.Module):
    """
    Dencoder
    """

    def __init__(self, x_size, y_size, num_hidden, decMoe,num_components):
        super(Decoder, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.num_components=num_components
        self.num_hidden = num_hidden
        self.attribute = nn.Linear(self.x_size, int(self.num_hidden / 2))
        self.decoder1 = nn.Sequential(
            nn.Linear(self.num_hidden + int(self.num_hidden / 2), num_hidden),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(num_hidden, self.num_hidden),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.num_hidden, 2),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.num_hidden + int(self.num_hidden / 2), num_hidden),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(num_hidden, int(num_hidden / 2)),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(int(num_hidden / 2), 2),
        )
        self.expert = MixtureOfExpertsModel2(self.num_hidden + int(self.num_hidden / 2), self.num_hidden + int(self.num_hidden / 2),self.num_components)
        self.softplus = nn.Softplus()
        self.decMoe = decMoe

    def forward(self, target_x, r):
        """ context_x : (batch_size, n_context, x_size)
            context_y : (batch_size, n_context, y_size)
            target_x : (batch_size, n_target, x_size)
        """
        bs, nt, x_size = target_x.shape  # (bs,nt, x_size)
        t_x = self.attribute(target_x)
        z_tx = torch.cat([t_x, r], dim=-1)
        z1_tx = z_tx.view((bs * nt, 1 * self.num_hidden + int(self.num_hidden / 2)))
        if self.decMoe:
            mu, sigma, alpha = self.expert(z1_tx)
            # # decoder1 = self.decoder1(expert_output)  # (bs * nt, 2)
            # decoder1 = expert_output.view((bs, nt, 2))  # (bs, nt, 2)
            # mu = decoder1[:, :, 0]
            # log_sigma = decoder1[:, :, 1]
            # sigma = 0.1 + 0.9 * self.softplus(log_sigma)
        else:
            decoder2 = self.decoder1(z1_tx)  # (bs * nt, 2)
            decoder2 = decoder2.view((bs, nt, 2))  # (bs, nt, 2)
            mu = decoder2[:, :, 0]
            log_sigma = decoder2[:, :, 1]
            sigma = 0.1 + 0.9 * self.softplus(log_sigma)  # variance  sigma=0.1+0.9*log(1+exp(log_sigma))
        return mu, sigma, alpha


class SpatialNeuralProcess(nn.Module):
    def __init__(self, x_size, y_size, num_hidden, encMoe, decMoe, distance, Similarity1, num_components):
        super(SpatialNeuralProcess, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.num_hidden = num_hidden
        self.num_components=num_components
        self.encMoe = encMoe
        self.decMoe = decMoe
        self.distance = distance
        self.Similarity1 = Similarity1
        self.determine = DeterministicEncoder(self.x_size, self.num_hidden, self.encMoe, self.distance, self.Similarity1)
        self.decoder = Decoder(self.x_size, self.y_size, self.num_hidden, self.decMoe,self.num_components)

        # self.moran_index = MoranIndexOptimized(k=5, decay_rate=0.1)

    def forward(self, context_x, context_y, target_x, target_y):
        """ context_x : (batch_size, n_context, x_size)
            context_y : (batch_size, n_context, y_size)
            target_x : (batch_size, n_target, x_size)
        """
        r = self.determine(context_x, context_y, target_x)  # attribute_cross_attention(context_x,ccontext_y,target_x)
        mu, sigma, alpha = self.decoder(target_x, r)  # decoder cat(z,target_x, r, w)  (bs,3*num_hidden+tx)--> (bs  nt, 2)
        # moran_y, moran_mu = self.moran_index(target_x, target_y, mu)
        return mu, sigma, alpha

from torch.distributions import Normal

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

        self.kl_div = nn.KLDivLoss()
        self.mse = nn.MSELoss()

    # def forward(self, prior_mu, prior_sigma, mu, sigma, target_y):
    def forward(self, mu, sigma, target_y, alpha):
        """ mu : (bs, n_target)
            sigma : (bs, n_target)
            target_y : (bs, n_target)
        """
        if target_y.shape[0] == 1:
            target_y = target_y.T  # (287, 1)

        m = Normal(mu, sigma)
        probs = m.log_prob(target_y)  # (B, K)
        log_probs = torch.log(alpha + 1e-8) + probs  # (B, K)
        log_sum = torch.logsumexp(log_probs, dim=1)  # (B,)
        return -log_sum.mean()
        # else:
        #     log_p = None
        #     loss = None

        #loss = loss / len(mu) + self.lam * self.mse(moran_y, moran_mu)
        # loss = loss / len(mu)
        # return loss
