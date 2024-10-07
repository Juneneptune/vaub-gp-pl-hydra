import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.mixture_same_family import MixtureSameFamily
import torch.distributions as DIST
from torch.utils.data import DataLoader
import torch.nn as nn


class MoGNN(nn.Module):

    def __init__(self, n_components, input_dim, loc_init=None, scale_init=None, weight_init=None):

        super(MoGNN, self).__init__()

        if loc_init is None:
            self.loc = nn.Parameter((torch.rand(n_components, input_dim)*2-1))
        else:
            self.loc = nn.Parameter(loc_init)

        if scale_init is None:
            self.log_scale = nn.Parameter(torch.zeros(n_components, input_dim))
        else:
            self.log_scale = nn.Parameter(torch.log(scale_init))

        if weight_init is None:
            self.raw_weight = nn.Parameter(torch.ones(n_components))
        else:
            self.raw_weight = nn.Parameter(torch.log(weight_init/(1-weight_init)))

    def log_prob(self, Z):

        self.loc = self.loc.to(Z.device)
        self.log_scale = self.log_scale.to(Z.device)
        self.raw_weight = self.raw_weight.to(Z.device)

        # print(torch.sigmoid(self.raw_weight))
        mix = torch.distributions.Categorical(torch.sigmoid(self.raw_weight))
        comp = torch.distributions.Independent(torch.distributions.Normal(self.loc, torch.exp(self.log_scale)), 1)
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)

        return gmm.log_prob(Z)


def kl_divergence_with_pz(prior, z, mu, log_var):
    std = torch.exp(log_var/2)

    # p = torch.distributions.Normal(torch.zeros_like(mu)+self.latent_loc, torch.ones_like(std)*torch.exp(self.latent_log_var/2))
    # p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    log_qzx = q.log_prob(z)
    # log_pz = p.log_prob(z)
    log_pz = prior.log_prob(z)

    kl = (log_qzx.sum(-1) - log_pz)/z.shape[-1]

    return kl.mean()