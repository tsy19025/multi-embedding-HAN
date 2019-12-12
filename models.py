import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from h_atten import HomoAttention, HeteAttention, UserItemAttention

class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = nn.Parameter(torch.zeros(inp_dim, out_dim))
        # weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        # weight = nn.Parameter(torch.from_numpy(weight))
        # bias = np.zeros(out_dim, dtype=np.float32)
        # bias = nn.Parameter(torch.from_numpy(weight))
        bias = nn.Parameter(torch.zeros(out_dim))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias

class multi_HAN(nn.Module):
    def __init__(self, n_nodes_list, args):
        super(multi_HAN, self).__init__()
        user_cuda = torch.cuda.is_available() and args.cuda
        dev = torch.device('cuda' if user_cuda else 'cpu')
        self.dataset = args.dataset
        self.n_facet = args.n_facet
        self.emb_dim = args.emb_dim
        self.niter = args.iter
        if self.dataset == 'yelp':
            n_users, n_businesses, n_cities, n_categories = n_nodes_list
            cur_dim = self.n_facet * self.emb_dim
            self.user_embed_init = SparseInputLinear(n_users, cur_dim)
            self.business_embed_init = SparseInputLinear(n_businesses, cur_dim)
            self.city_embed_init = SparseInputLinear(n_cities, cur_dim)
            self.category_embed_init = SparseInputLinear(n_categories, cur_dim)
        else:
            print('dataset wrong!')

    def forward(self, users, businesses, user_neigh_list_lists, business_neigh_list_lists):
        if self.dataset == 'yelp':
            user_embed = self.user_embed_init(users)
            business_embed = self.business_embed_init(businesses)
            neigh_emb_list = [self.user_embed_init, self.business_embed_init, self.city_embed_init, self.category_embed_init]
            #user embedding propagate
            user_homo_encoder_list = []
            for list_index in range(len(user_neigh_list_lists)):
                for neigh in user_neigh_list_lists[list_index]:
                    user_neigh_embed = neigh_emb_list[list_index](neigh)
                    user_homo_encoder = HomoAttention(self.emb_dim, self.n_facet)
                    user_homo_encoder_list.append(user_homo_encoder(user_embed, user_neigh_embed))
            user_hete_encoder = HeteAttention(self.emb_dim, self.n_facet, self.niter)
            updated_user_embed = user_hete_encoder(user_embed, torch.stack(user_homo_encoder_list, dim=1))
            #business embedding propagete
            business_homo_encoder_list = []
            for list_index in range(len(business_neigh_list_lists)):
                for neigh in business_neigh_list_lists[list_index]:
                    business_neigh_embed = neigh_emb_list[list_index](neigh)
                    business_homo_encoder = HomoAttention(self.emb_dim, self.n_facet)
                    business_homo_encoder_list.append(business_homo_encoder(business_embed, business_neigh_embed))
            business_hete_encoder = HeteAttention(self.emb_dim, self.n_facet, self.niter)
            updated_business_embed = business_hete_encoder(user_embed, torch.stack(business_homo_encoder_list, dim=1))
        logit = self.autocross(updated_user_embed, updated_business_embed)
        return fn.softmax(logit, dim=1)
    def autocross(self, user_emb, business_emb):
        user_item_fusion_layer = UserItemAttention(self.emb_dim, self.n_facet)
        return user_item_fusion_layer(user_emb, business_emb)
