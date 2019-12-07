import json
import numpy as np
# from numpy import array
# from scipy import sparse
import torch
from torch.utils.data import Dataset
import pickle
# from yelp_dataset.data_sparse import load_jsondata_from_file, get_id_to_num

# def load_jsondata_from_file(path):
#     data = []
#     with open(path, 'r') as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data

# def matrix_factorization(adj_user_review, adj_user_business, adj_review_business, k, lr = 0.001):
#     tot_user, tot_review = adj_user_review.shape
#     _, tot_business = adj_user_business.shape
#
#     W_user = Variable(torch.randn(tot_user, k), requires_grad = True)
#     W_review = Variable(torch.randn(tot_review, k), requires_grad = True)
#     W_business = Variable(torch.randn(tot_business, k), requires_grad = True)
#
#     old_loss = 0
#     while (True):
#         user_review = W_user.mm(W_review.t())
#         user_business = W_user.mm(W_business.t())
#         review_business = W_review.mm(W_business.t())
#         loss = (user_review - adj_user_review).pow(2).sum() + (user_business - adj_user_business).pow(2).sum() + (review_business - adj_review_business).pow(2).sum()
#
#         # print("loss:{:.4f}".format(loss.data))
#         if abs(loss.data - old_loss) < 1e-5: break
#         old_loss = loss.data
#         loss.backward()
#
#         W_user.data -= lr * W_user.grad.data
#         W_review.data -= lr * W_review.grad.data
#         W_business.data -= lr * W_business.grad.data
#
#         W_user.grad.data.zero_()
#         W_review.grad.data.zero_()
#         W_business.grad.data.zero_()
#
#         # print(W_user)
#         # print(W_review)
#         # print(W_business)
#     def normalize(W):
#         z = np.zeros(W.shape[0])
#         for vector in W:
#             z = z + vector
#         return np.exp(z)/sum(np.exp(z))
#     return normalize(W_user), normalize(W_review), normalize(W_business)
#
# def read_data(path):
#     user_json = load_jsondata_from_file(path + '/')
    

class YelpDataset(Dataset):
    def __init__(self, data_path, adj_paths, neighbor_size):
        # xy = load_jsondata_from_file(review_path)
        self.neighbor_size = neighbor_size
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        adjs = [] # adj_UU, adj_UB, adj_BCa, adj_BCi, adj_UUB, adj_UBU, adj_UBUB, adj_UBCa, adj_UBCi, adj_BCaB, adj_BCiB
        for adj_path in adj_paths:
            with open(adj_path, 'rb') as f:
                adjs.append(pickle.load(f))
        print(adjs[0].shape)
        adjs.append(adjs[1].T)
        self.user_adjs = [torch.LongTensor(adjs[i]) for i in [0, 1, 4, 5, 6, 7, 8]]
        self.business_adjs = [torch.LongTensor(adjs[i]) for i in [1, 2, 3, 9, 10, 11]]
        # self.x_user = torch.LongTensor([user_num for user_num in range(len(num_to_userid))], requires_grad=False)
        # self.x_business = torch.LongTensor([business_num for business_num in range(len(num_to_businessid))], requires_grad=False)
        # self.x_user = torch.LongTensor([userid_to_num[r['user_id']] for r in xy], requires_grad=False)
        # self.x_business = torch.LongTensor([userid_to_num[r['business_id']] for r in xy], requires_grad=False)
        # self.y = torch.FloatTensor([float(r["stars"]) for r in xy], requires_grad=False)
        # self.len = xy.shape[0]

    def __getitem__(self, index):
        user = self.data[index]["user"]
        business = self.data[index]["business"]
        label = self.data[index]["rate"]
        user_neighbors_list = []
        business_neighbors_list = []
        for adj in self.user_adjs:
            neighbors_index = np.nonzero(adj[index])[0]
            if len(neighbors_index) < self.neighbor_size:
                neighbors = np.random.choice(neighbors_index, size=self.neighbor_size, replace=True)
            else:
                neighbors = np.random.choice(neighbors_index, size=self.neighbor_size, replace=False)
            user_neighbors_list.append(neighbors)
        for adj in self.business_adjs:
            neighbors_index = np.nonzero(adj[index])[0]
            if len(neighbors_index) < self.neighbor_size:
                neighbors = np.random.choice(neighbors_index, size=self.neighbor_size, replace=True)
            else:
                neighbors = np.random.choice(neighbors_index, size=self.neighbor_size, replace=False)
            business_neighbors_list.append(neighbors)
        return user, business, label, user_neighbors_list, business_neighbors_list

    def __len__(self):
        return len(self.data)
        # return self.x_user[index], self.x_business[index], self.y[index]

# class Metapath():
#     def __init__(self, name, pathtype, adj):
#         # pathtype: 0.user 1.review 2.business
#         self.name = name
#         self.pathtype = pathtype
#         ix = torch.LongTensor(adj.row)
#         iy = torch.LongTensor(adj.col)
#         idata = torch.FloatTensor(adj.data)
#         ixy = torch.cat([ix.unsqueeze(0), iy.unsqueeze(0)], dim = 0)
#         self.adj = torch.sparse.FloatTensor(ixy, idata, adj.shape)
        
