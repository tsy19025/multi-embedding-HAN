import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, input_feature_size, output_feature_size, dropout, alpha, concat = True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.input_feature_size = input_feature_size
        self.out_feature_size = out_feature_size
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size = (input_feature_size, output_feature_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(output_feature_size * 2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, input, adj):
        hidden = torch.mm(input, self.W)
        N = hidden.size()[0]
        a_input = torch.cat([hidden.repeat(1, N).view(N * N, -1), hidden.repeat(N, 1)], dim=1).view(N, -1, 2 * self.output_feature_size)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        
        if self.concat: return F.elu(h_prime)
        else: return h_prime
