# HyperAMP model
# MIC regression via hypergraph neural network
# Ruihan Dong, 23-04-20

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg.structure.hypergraphs import Hypergraph


class HGNNConv(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False):

        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, L_HGNN: torch.Tensor) -> torch.Tensor:
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        # X = hg.smoothing_with_HGNN(X) # i.e. torch.matmul(hg.L_HGNN, X) (no dropout)
        X = torch.matmul(L_HGNN.to_dense(), X)
        if not self.is_last:
            X = self.drop(self.act(X))
        return X

class HGNN(nn.Module):
    def __init__(self,
        in_channels: int = 768,   # for Ankh
        hid_channels: int = 640,
        emb_channels: int = 320,
        use_bn: bool = False,
        drop_rate: float = 0.2,) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(HGNNConv(in_channels, hid_channels, use_bn=use_bn))
        self.layers.append(HGNNConv(hid_channels, emb_channels, use_bn=use_bn, is_last=True))

        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop_rate)

        self.fc = nn.Linear(emb_channels, 16)
        # combined layers
        self.fc1 = nn.Linear(960 * 3, hid_channels)
        self.fc2 = nn.Linear(hid_channels, hid_channels)
        self.out = nn.Linear(hid_channels, 1) # 1 for regression

    def forward(self, X: torch.Tensor, L_HGNN: torch.Tensor) -> torch.Tensor:
        def single_hgnn(i, X):
            X1 = X
            for layer in self.layers:
                X1 = layer(X1, L_HGNN[i])
            X2 = self.fc(X1)
            X2 = torch.flatten(X2, 1, 2)
            Xmean = torch.mean(X1, dim=1)
            X3 = torch.cat((Xmean, X2), 1)
            return X3

        # fully connected layers
        X_k2 = single_hgnn(0, X)
        X_k3 = single_hgnn(1, X)
        X_k4 = single_hgnn(2, X)
        Xcat = torch.cat((X_k2, X_k3, X_k4), axis=1)

        xc = self.fc1(Xcat)
        xc = self.fc2(self.drop(self.act(xc)))
        out = self.out(self.drop(self.act(xc))).squeeze(-1)
        return out
