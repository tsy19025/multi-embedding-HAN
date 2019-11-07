import torch
import torch.nn as nn
from layers import GraphAttentionLayer, SparseGraphAttentionLayer
from utils import Metapath

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
    
class OneMetapathGAT(nn.Module):
    def __init__(self, feature_size, the_metapath, nheads, alpha):
        super(OneMetapathGAT, self).__init__()
        self.feature_size = feature_size
        self.the_metapath = the_metapath
        self.SGAT = SparseGraphAttentionNetwork(feature_size, feature_size, feature_size, nheads, alpha)
        self.pathtype = the_metapath.pathtype
        
    def forward(self, node_features):
        # node_features: N_node * 3 * feature_size
        input = node_features[:,self.pathtype]
        x = self.SGAT(input, self.the_metapath.adj)
        return x
        
if __name__ == '__main__':
    n = 3
    features = 20
    hiddens = 20
    outputs = 15
    adj = [[1, 0, 1], [0, 1, 1], [0, 1, 1]]
    adj = torch.tensor(adj)
    edge = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 2, 1, 2, 1, 2]])
    
    node_features = torch.rand([n, features])
    model = GraphAttentionNetwork(features, hiddens, outputs, nheads = 2, alpha = 0.2)
    output = model(node_features, adj)
    print("output of GAT: ", output)

    print("-------------------------------------------------")

    model = SparseGraphAttentionNetwork(features, hiddens, outputs, nheads = 2, alpha = 0.2)
    output = model(node_features, edge)
    print("output of SparseGAT: ", output)
