import torch
import torch.nn as nn
from layers import GraphAttentionLayer

class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_features, hiddens, outputs, nheads, alpha):
        super(GraphAttentionNetwork, self).__init__()
        self.input_features = input_features
        self.outputs = outputs

        self.layer = nn.Linear(input_features, input_features)
        self.attentions = [GraphAttentionLayer(input_features, hiddens, alpha) for i in range(nheads)]
        self.attentionend = GraphAttentionLayer(hiddens * nheads, outputs, alpha)
    def forward(self, input, adj):
        x = self.layer(input)
        x = torch.cat([attention(x, adj) for attention in self.attentions], dim = 1)
        x = self.attentionend(x, adj)
        x = torch.softmax(x, dim = 1)
        return x
