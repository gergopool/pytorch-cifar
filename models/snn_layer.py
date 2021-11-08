import torch
from torch import nn
from torch.nn.functional import softmax, normalize
import time

class SnnLayer(nn.Module):

    def __init__(self, emb_len, n_classes, n_protos=60, tau=100., smooth=0.1):
        super().__init__()
        self.protos = nn.Parameter(torch.rand(n_protos, emb_len))

        offset = smooth / n_classes
        base_i = torch.arange(n_protos)
        base_votes = torch.zeros(n_protos, n_classes) + offset
        base_votes[base_i, base_i % n_classes] += 1 - offset
        self.labels = nn.Parameter(base_votes * 100)
        self.tau = tau

    def forward(self, x):
        x1 = x.unsqueeze(0)
        x2 = self.protos.unsqueeze(0)
        dist = torch.cdist(x1,x2)[0]
        votes = softmax(-dist / self.tau, dim=1)
        return votes @ self.labels