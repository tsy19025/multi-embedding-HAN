import torch
import torch.nn as nn
import torch.nn.functional as fn

class HomoAttention(nn.Module):
    def __init__(self, features, neigh_features):
        super(HomoAttention, self).__init__()
        self.features = features
        self.neigh_features = features

    def forward(self):


class HeteAttention(nn.Module):
    def __init__(self, emb_dim, n_facet, niter):
        super(HeteAttention, self).__init__()
        self.emb_dim = emb_dim
        self.n_facet = n_facet
        self.n_iter = niter
        # self.features = features
        # self.metapath_features = metapath_features
    def forward(self, features, metapath_featuers):
        batch_size = features.shape[0]
        neighbor_size = metapath_featuers.shape[1]
        x = features.view(batch_size, self.n_facet, self.emb_dim)
        z = metapath_featuers.view(batch_size, neighbor_size, self.n_facet, self.emb_dim)
        x = fn.normalize(x, dim=2)
        z = fn.normalize(z, dim=3)
        u = x
        for clus_iter in range(self.n_iter):
            p = torch.sum(z*u.view(batch_size, 1, self.n_facet, self.emb_dim), dim=3)
            p= fn.softmax(p, dim=2)
            u = torch.sum(z*p.view(batch_size, neighbor_size, self.n_facet, 1), dim=1)
            u += x
            if clus_iter < self.n_iter - 1:
                u = fn.normalize(u, dim=2)
        return u.view(batch_size, self.n_facet*self.emb_dim)



