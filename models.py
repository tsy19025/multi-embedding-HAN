import torch
import torch.nn as nn
from layers import GraphAttentionLayer, SparseGraphAttentionLayer

class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_features, hiddens, outputs, nheads, alpha):
        super(GraphAttentionNetwork, self).__init__()
        self.input_features = input_features
        self.outputs = outputs

        self.layer = nn.Linear(input_features, input_features)
        self.attentions = [GraphAttentionLayer(input_features, hiddens, alpha, end = False) for i in range(nheads)]
        self.attentionend = GraphAttentionLayer(hiddens * nheads, outputs, alpha, end = True)
    def forward(self, input, adj):
        input = self.layer(input)
        x = torch.cat([attention(input, adj) for attention in self.attentions], dim = 1)
        print(x.shape)
        x = self.attentionend(x, adj)
        x = torch.softmax(x, dim = 1)
        return x

class SparseGraphAttentionNetwork(nn.Module):
    def __init__(self, input_features, hiddens, outputs, nheads, alpha):
        super(SparseGraphAttentionNetwork, self).__init__()
        self.input_features = input_features
        self.outputs = outputs

        self.layer = nn.Linear(input_features, input_features)
        self.attentions = [SparseGraphAttentionLayer(input_features, hiddens, alpha, end = False) for i in range(nheads)]
        self.attentionend = SparseGraphAttentionLayer(hiddens * nheads, outputs, alpha, end = True)
    def forward(self, input, edges):
        input = self.layer(input)
        x = torch.cat([attention(input, edges) for attention in self.attentions], dim = 1)
        x = self.attentionend(x, edges)
        x = torch.softmax(x, dim = 1)
        return x
