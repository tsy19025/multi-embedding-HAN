import torch
import torch.nn as nn

class HomoAttention(nn.Module):
    def __init__(self, features, neigh_features):
        super(HomoAttention, self).__init__()
        self.features = features
        self.neigh_features = features

    def forward(self):


class HeteAttention(nn.Module):
    def __init__(self, features, metapath_features):
        super(HeteAttention, self).__init__()
        self.features = features
        self.metapath_features = metapath_features

    def forward(self):
