import numpy as np
import torch
import torch.nn as nn
from h_atten import HomoAttention, HeteAttention, UserItemAttention
import time

class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

class multi_HAN(nn.Module):
    def __init__(self, n_nodes_list, args):
        super(multi_HAN, self).__init__()
        user_cuda = torch.cuda.is_available() and args.cuda
        self.device = torch.device('cuda' if user_cuda else 'cpu')
        self.dataset = args.dataset
        self.n_facet = args.n_facet
        self.emb_dim = args.emb_dim
        self.n_iter = args.n_iter
        self.n_neg = args.n_neg
        if self.dataset == 'yelp':
            n_users, n_businesses, n_cities, n_categories = n_nodes_list
            cur_dim = self.n_facet * self.emb_dim
            self.user_emb_init = SparseInputLinear(n_users, cur_dim).to(self.device)
            self.business_emb_init = SparseInputLinear(n_businesses, cur_dim).to(self.device)
            self.city_emb_init = SparseInputLinear(n_cities, cur_dim).to(self.device)
            self.category_emb_init = SparseInputLinear(n_categories, cur_dim).to(self.device)
            # self.user_emb_init = nn.Embedding(n_users, cur_dim)
            # self.business_emb_init = nn.Embedding(n_businesses, cur_dim)
            # self.city_emb_init = nn.Embedding(n_cities, cur_dim)
            # self.category_emb_init = nn.Embedding(n_categories, cur_dim)
        else:
            print('dataset wrong!')

    # def forward(self, users, pos_businesses, neg_businesses, user_neigh_list_lists, pos_business_neigh_list_lists, neg_business_neigh_list_lists):
    def forward(self, users, businesses, user_neigh_list_lists, business_neigh_list_lists):
        if self.dataset == 'yelp':
            user_emb = self.user_emb_init(users)
            neigh_emb_list = [self.user_emb_init, self.business_emb_init, self.city_emb_init, self.category_emb_init]
            homo_encoder = HomoAttention(self.emb_dim, self.n_facet).to(self.device)
            hete_encoder = HeteAttention(self.emb_dim, self.n_facet, self.n_iter).to(self.device)
            #user embedding propagate
            user_homo_encoder_list = []
            for list_index in range(len(user_neigh_list_lists)):
                for neigh in user_neigh_list_lists[list_index]:
                    user_neigh_emb = neigh_emb_list[list_index](neigh)
                    user_homo_encoder_list.append(homo_encoder(user_emb, user_neigh_emb))
            updated_user_emb = hete_encoder(user_emb, torch.stack(user_homo_encoder_list, dim=1))
            #business embedding propagate
            business_emb = self.business_emb_init(businesses)
            business_homo_encoder_list = []
            for list_index in range(len(business_neigh_list_lists)):
                for neigh in business_neigh_list_lists[list_index]:
                    business_neigh_emb = neigh_emb_list[list_index](neigh)
                    business_homo_encoder_list.append(homo_encoder(business_emb, business_neigh_emb))
            updated_business_emb = hete_encoder(business_emb, torch.stack(business_homo_encoder_list, dim=1))
            # pos_business_emb = self.business_emb_init(pos_businesses)
            # pos_business_homo_encoder_list = []
            # for list_index in range(len(pos_business_neigh_list_lists)):
            #     for neigh in pos_business_neigh_list_lists[list_index]:
            #         pos_business_neigh_emb = neigh_emb_list[list_index](neigh)
            #         pos_business_homo_encoder_list.append(homo_encoder(pos_business_emb, pos_business_neigh_emb))
            # updated_pos_business_emb = hete_encoder(pos_business_emb, torch.stack(pos_business_homo_encoder_list, dim=1))
            #neg business embedding propagate
            # updated_neg_business_embs = []
            # neg_businesses_embs = []
            # for neg in range(self.n_neg):
            #     neg_business_emb = self.business_emb_init(neg_businesses[neg])
                # neg_businesses_embs.append(neg_business_emb)
                # neg_business_homo_encoder_list = []
                # for list_index in range(len(neg_business_neigh_list_lists[neg])):
                #     for neigh in neg_business_neigh_list_lists[neg][list_index]:
                #         neg_business_neigh_emb = neigh_emb_list[list_index](neigh)
                #         neg_business_homo_encoder_list.append(homo_encoder(neg_business_emb, neg_business_neigh_emb))
                # updated_neg_business_embs.append(hete_encoder(neg_business_emb, torch.stack(neg_business_homo_encoder_list, dim=1)))
        # logit = self.autocross(updated_user_embed, updated_pos_business_emb, updated_neg_business_embs)
        # logit = self.autocross(user_emb, pos_business_emb, neg_businesses_embs)
        logit = self.autocross(updated_user_emb, updated_business_emb)
        return logit
    # def autocross(self, user_emb, pos_business_emb, neg_business_embs):

    def autocross(self, user_emb, business_emb):
        # logit = []
        user_item_fusion_layer = UserItemAttention(self.emb_dim, self.n_facet).to(self.device)
        logit = user_item_fusion_layer(user_emb, business_emb)
        return logit
        # pos_logit = user_item_fusion_layer(user_emb, pos_business_emb)
        # pos_logit = torch.sum(torch.mul(user_emb, pos_business_emb),1)
        # logit.append(pos_logit)
        # for neg_business_emb in neg_business_embs:
        #     logit.append(user_item_fusion_layer(user_emb, neg_business_emb))
            # logit.append(torch.sum(torch.mul(user_emb, neg_business_emb),1))
        # return torch.squeeze(torch.stack(logit, dim=1))

