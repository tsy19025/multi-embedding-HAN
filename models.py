import numpy as np
import torch
import torch.nn as nn
# from layers import GraphAttentionLayer, SparseGraphAttentionLayer
# from utils import Metapath

class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias

class multi_HAN(nn.Module):
    def __init__(self, n_nodes_list, emb_dim, n_facet, dataset, args):
        super(multi_HAN, self).__init__()
        user_cuda = torch.cuda.is_available() and args.cuda
        dev = torch.device('cuda' if user_cuda else 'cpu')
        self.dataset = dataset
        if self.dataset == 'yelp':
            n_users, n_businesses, n_cities, n_categories = n_nodes_list
            cur_dim = n_facet * emb_dim
            self.user_embed_init = SparseInputLinear(n_users, cur_dim)
            self.business_embed_init = SparseInputLinear(n_businesses, cur_dim)
            self.city_embed_init = SparseInputLinear(n_cities, cur_dim)
            self.category_embed_init = SparseInputLinear(n_categories, cur_dim)
        else:
            print('dataset wrong!')

    def forward(self, users, businesses, user_neigh_list_lists, business_neigh_list_lists):
        if self.dataset == 'yelp':
            user_user_neigh_list, user_business_neigh_list, user_city_neigh_list, user_category_neigh_list = user_neigh_list_lists
            business_business_neigh_list, business_user_neigh_list, business_city_neigh_list, business_category_neigh_list = business_neigh_list_lists
            user_embed = self.user_embed_init(users)
            business_embed = self.business_embed_init(businesses)
            #user embedding propagate
            homo_encoder_list = []
            for user_neigh in user_user_neigh_list:
                neigh_embed = self.user_embed_init(user_neigh)
                homo_encoder = HomoAttention(features=user_embed, neigh_features=neigh_embed)
                homo_encoder_list.append(homo_encoder())
            for business_neigh in user_business_neigh_list:
                neigh_embed = self.business_embed_init(business_neigh)
                homo_encoder = HomoAttention(features=user_embed, neigh_features=neigh_embed)
                homo_encoder_list.append(homo_encoder())
            for city_neigh in user_city_neigh_list:
                neigh_embed = self.city_embed_init(city_neigh)
                homo_encoder = HomoAttention(features=user_embed, neigh_features=neigh_embed)
                homo_encoder_list.append(homo_encoder())
            for category_neigh in user_category_neigh_list:
                neigh_embed = self.category_embed_init(category_neigh)
                homo_encoder = HomoAttention(features=user_embed, neigh_features=neigh_embed)
                homo_encoder_list.append(homo_encoder())
            hete_encoder = HeteAttention(features=user_embed, meta_feature_list=homo_encoder_list)

            #business embedding update






    def loss(self):
        return loss


# class GraphAttentionNetwork(nn.Module):
#     def __init__(self, input_features, hiddens, outputs, nheads, alpha):
#         super(GraphAttentionNetwork, self).__init__()
#         self.input_features = input_features
#         self.outputs = outputs
#
#         self.layer = nn.Linear(input_features, input_features)
#         self.attentions = [GraphAttentionLayer(input_features, hiddens, alpha, end = False) for i in range(nheads)]
#         self.attentionend = GraphAttentionLayer(hiddens * nheads, outputs, alpha, end = True)
#     def forward(self, input, adj):
#         input = self.layer(input)
#         x = torch.cat([attention(input, adj) for attention in self.attentions], dim = 1)
#         print(x.shape)
#         x = self.attentionend(x, adj)
#         x = torch.softmax(x, dim = 1)
#         return x
#
# class SparseGraphAttentionNetwork(nn.Module):
#     def __init__(self, input_features, hiddens, outputs, nheads, alpha):
#         super(SparseGraphAttentionNetwork, self).__init__()
#         self.input_features = input_features
#         self.outputs = outputs
#
#         self.layer = nn.Linear(input_features, input_features)
#         self.attentions = [SparseGraphAttentionLayer(input_features, hiddens, alpha, end = False) for i in range(nheads)]
#         self.attentionend = SparseGraphAttentionLayer(hiddens * nheads, outputs, alpha, end = True)
#     def forward(self, input, edges):
#         input = self.layer(input)
#         x = torch.cat([attention(input, edges) for attention in self.attentions], dim = 1)
#         x = self.attentionend(x, edges)
#         x = torch.softmax(x, dim = 1)
#         return x
#
# class OneMetapathGAT(nn.Module):
#     def __init__(self, feature_size, the_metapath, nheads, alpha):
#         super(OneMetapathGAT, self).__init__()
#         self.feature_size = feature_size
#         self.the_metapath = the_metapath
#         self.SGAT = SparseGraphAttentionNetwork(feature_size, feature_size, feature_size, nheads, alpha)
#         self.pathtype = the_metapath.pathtype
#
#     def forward(self, node_features):
#         # node_features: N_node * 3 * feature_size
#         input = node_features[:,self.pathtype]
#         x = self.SGAT(input, self.the_metapath.adj)
#         return x
#
# if __name__ == '__main__':
#     n = 3
#     features = 20
#     hiddens = 20
#     outputs = 15
#     adj = [[1, 0, 1], [0, 1, 1], [0, 1, 1]]
#     adj = torch.tensor(adj)
#     edge = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 2, 1, 2, 1, 2]])
#
#     node_features = torch.rand([n, features])
#     model = GraphAttentionNetwork(features, hiddens, outputs, nheads = 2, alpha = 0.2)
#     output = model(node_features, adj)
#     print("output of GAT: ", output)
#
#     print("-------------------------------------------------")
#
#     model = SparseGraphAttentionNetwork(features, hiddens, outputs, nheads = 2, alpha = 0.2)
#     output = model(node_features, edge)
#     print("output of SparseGAT: ", output)
