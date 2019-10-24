import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer

class GAT(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, dropout, alpha, head_size):
        super(GAT, self).__init__()
        
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(feature_size, hidden_size, dropout, alpha) for i in range(head_size)]

        self.out_att = GraphAttentionLayer(hidden_size * head_size, output_size, dropout, alpha, False)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training = self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim = 1)
        x = F.dropout(x, self.dropout, training = self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim = 1)

class HeteGAT_multi(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, ):
