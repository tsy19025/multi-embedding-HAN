import torch
import torch.nn as nn

class HomoAttention(nn.Module):
    def __init__(self, features, neigh_features):
        super(HomoAttention, self).__init__()
        self.features = features
        self.neigh_features = features

    def forward(self):

        
