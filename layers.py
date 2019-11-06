import torch
import torch.nn as nn
from scipy import sparse

# train weight-matrix W and attention function
class GraphAttentionLayer(nn.Module):
    def __init__(self, input_features, outputs, alpha):
        super(GraphAttentionLayer, self).__init__()

        self.input_features = input_features
        self.outputs = self.outputs

        self.W = nn.Parameter(torch.zero([input_features, outputs]))
        self.a = nn.Parameter(torch.zero([2 * input_features, 1]))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, adj, input):
        n = input.size()[0]
        h = torch.mm(input, self.W)

        a_input = torch.cat([h.repeat(1, n).view(n * n, -1)], dim = 1).view(n, -1, 2 * self.outputs)
        e = self.leakyrelu(torch.matmul(a_input, self.a))

        zeros_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zeros_vec)
        attention = torch.softmax(attention, dim = 1)
        h = torch.matmul(attention, h)
        return h

