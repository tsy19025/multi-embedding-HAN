import torch
import torch.nn as nn
from scipy import sparse
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, input_features, outputs, alpha, end):
        super(GraphAttentionLayer, self).__init__()

        self.input_features = input_features
        self.outputs = outputs
        self.end = end

        self.W = nn.Parameter(torch.zeros([input_features, outputs]))
        self.a = nn.Parameter(torch.zeros([2 * outputs, 1]))
        self.leakyrelu = nn.LeakyReLU(alpha)
    def forward(self, input, adj):
        n = input.size()[0]
        h = torch.mm(input, self.W)

        a_input = torch.cat([h.repeat(1, n).view(n * n, -1), h.repeat(n, 1)], dim = 1).view(n, -1, 2 * self.outputs)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zeros_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zeros_vec)
        attention = torch.softmax(attention, dim = 1)
        h = torch.matmul(attention, h)
        if self.end: return h
        return F.elu(h)

class SparsemmFunction(torch.autograd.Function):
    def forward(ctx, indices, values, shape, b):
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.n = shape[0]
        return torch.spmm(a, b)
    def backward(ctx, grad_output):
        a, b = ctx.save_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = torch.matmul(grad_output, b.t())
            edge_idx = a._indices()[0, :] * ctx.n + a._indices()[1, :]
            grad_values = grad_a_dense_view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = torch.matmul(a.t(), grad_output)
        return None, grad_values, None, grad_b

class Sparsemm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SparsemmFunction.apply(indices, values, shape, b)

class SparseGraphAttentionLayer(nn.Module):
    def __init__(self, input_features, outputs, alpha, end):
        super(SparseGraphAttentionLayer, self).__init__()
        self.sparsemm = Sparsemm()
        self.end = end
        self.W = nn.Parameter(torch.zeros([input_features, outputs]))
        self.a = nn.Parameter(torch.zeros([1, 2 * outputs]))
        self.leakyrelu = nn.LeakyReLU(alpha)
    def forward(self, input, edge):
        n = input.size()[0]
        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()

        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim = 1).t()
        tmp = torch.matmul(self.a, edge_h).squeeze()
        edge_e = torch.exp(-self.leakyrelu(tmp))
        assert not torch.isnan(edge_e).any()

        e_rowsum = self.sparsemm(edge, edge_e, torch.Size([n, n]), torch.ones([n, 1]))
        hh = self.sparsemm(edge, edge_e, torch.Size([n, n]), h)
        assert not torch.isnan(hh).any()

        hh = hh.div(e_rowsum)
        assert not torch.isnan(hh).any()
        if self.end: return hh
        return F.elu(hh)
