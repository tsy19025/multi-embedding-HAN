import torch
import torch.nn as nn
import torch.nn.functional as fn

class HomoAttention(nn.Module):
    def __init__(self, emb_dim, n_facet):
        super(HomoAttention, self).__init__()
        self.emb_dim = emb_dim
        self.n_facet = n_facet

    def forward(self, features, neigh_features):
        batch_size = features.shape[0]
        neighbor_size = neigh_features.shape[1]
        x = features.view(batch_size, self.n_facet, self.emb_dim)
        z = neigh_features.view(batch_size, neighbor_size, self.n_facet, self.emb_dim)
        x = fn.normalize(x, dim=2)
        z = fn.normalize(z, dim=3)
        a = torch.sum(z*x.view(batch_size, 1, self.n_facet, self.emb_dim), dim=3)
        a = fn.softmax(a, dim=1)
        u = torch.sum(z*a.view(batch_size, neighbor_size, self.n_facet, 1), dim=1)
        return u.view(batch_size, self.n_facet*self.emb_dim)

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

class UserItemAttention():
    def __init__(self, emb_dim, n_facet, num_values, has_residual):
        super(UserItemAttention, self).__init__()
        self.has_residual = has_residual
        self.num_values = num_values
        self.emb_dim = emb_dim
        self.n_facet = n_facet
        self.combined_layer = torch.nn.Linear(2 * emb_dim, emb_dim)
        self.fc = torch.nn.Linear(emb_dim, 1)

    def forward(self, user_emb, item_emb):
        batch_size = user_emb.shape[0]
        ###user self attention
        u_emb = user_emb.view(batch_size, self.n_facet, self.emb_dim)
        u_p = torch.matmul(u_emb, torch.transpose(u_emb, 1, 2))
        u_p = fn.softmax(u_p, dim=2)
        u_emb_combined = torch.matmul(u_p, u_emb)
        ###item self attention
        i_emb = item_emb.view(batch_size, self.n_facet, self.emb_dim)
        i_p = torch.matmul(i_emb, torch.transpose(i_emb, 1, 2))
        i_p = fn.softmax(i_p, dim=2)
        i_emb_combined = torch.matmul(i_p, i_emb)
        ###user item attention
        u_i_p = torch.matmul(u_emb_combined, torch.transpose(i_emb_combined, 1, 2))
        u_i_p = u_i_p.view(batch_size, self.n_facet*self.n_facet, 1)
        u_i_p = fn.softmax(u_i_p, dim=1)
        u_emb_combined = u_emb_combined.view(batch_size, self.n_facet, 1, self.emb_dim).expand(-1, -1, self.n_facet, -1)
        i_emb_combined = i_emb_combined.view(batch_size, 1, self.n_facet, self.emb_dim).expand(-1, self.n_facet, -1, -1)
        u_i_emb_combined = torch.cat([u_emb_combined, i_emb_combined], dim=3).view(batch_size, self.n_facet*self.n_facet, self.emb_dim)
        final_states = torch.tanh(self.combined_layer(u_i_emb_combined))
        final_states = final_states*u_i_p
        predic = torch.sigmoid(self.fc(final_states))
        return predic















