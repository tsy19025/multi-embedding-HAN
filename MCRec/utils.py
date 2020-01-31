import torch
from torch.utils.data import Dataset
import numpy as np
import random
import sys
import time

class YelpDataset(Dataset):
    def __init__(self, data, path_num, timestamps, adj_BCa, adj_BCi, adj_UB, adj_UU, negatives, sample, mode):
        self.n_user, self.n_item = adj_UB.shape
        self.data = data
        self.path_num = path_num
        self.timestamps = timestamps
        self.n_negative = negatives
        self.adj_BCa = adj_BCa
        self.adj_BCi = adj_BCi
        _, self.n_ca = adj_BCa.shape
        _, self.n_ci = adj_BCi.shape
        self.adj_UU = adj_UU
        self.adj_UB = adj_UB
        self.sample = sample
        self.mode = mode
        
        adj_BU = adj_UB.T
        self.item_bu = {}
        self.item_bca = {}
        self.item_bci = {}

        for i in range(self.n_item):
            bu = np.nonzero(adj_BU[i])[0]
            bca = np.nonzero(adj_BCa[i])[0]
            bci = np.nonzero(adj_BCi[i])[0]

            if len(bu) > 50: bu = np.random.choice(bu, size = 50, replace = False)
            self.item_bu[i] = set(bu)
            self.item_bca[i] = set(bca)
            self.item_bci[i] = set(bci)

        self.user_uu = {}
        self.user_ub = {}
        for u in range(self.n_user):
            uu = np.nonzero(adj_UU[u])[0]
            ub = np.nonzero(adj_UB[u])[0]

            if len(uu) > 50: uu = np.random.choice(uu, size = 50, replace = False)
            if len(ub) > 50: ub = np.random.choice(ub, size = 50, replace = False)
            self.user_uu[u] = uu
            self.user_ub[u] = ub


    def sample_negative_item_for_user(self, user, negative):
        items = []
        for t in range(negative):
            item = np.random.randint(0, self.n_item)
            while self.adj_UB[user][item] == 1:
                item = np.random.randint(0, self.n_item)
            items.append(item)
        # print("user: ", user, item)
        return items

    def get_path(self, u, i, idx):
        path = []
        if idx == 0:
            if self.adj_UB[u][i] == 1: return [[u, i]]
            return [[self.n_user, self.n_item]]
        if idx == 1:
            path = [user for user in self.user_uu[u] if user in self.item_bu[i]]
            n = len(path)
            # print("idx: ", idx, " n: ", n)
            if n == 0: return [[self.n_user, self.n_user, self.n_item]] * self.sample
            if n > self.sample: path = np.random.choice(path, size = self.sample, replace = False)
            else: path = np.random.choice(path, size = self.sample)
            return [[u, user, i] for user in path]
        if idx == 2:
            for item in self.user_ub[u]:
                for user in self.item_bu[i]:
                    if self.adj_UB[user][item] == 1:
                        path.append([u, item, user, i])
            n = len(path)
            # print("idx: ", idx, " n: ", n)
            if n == 0: return [[self.n_user, self.n_item, self.n_user, self.n_item]] * self.sample
            if n > self.sample: tmp = np.random.choice(range(n), size = self.sample, replace = False)
            else: tmp = np.random.choice(range(n), size = self.sample)
            return [path[t] for t in tmp]
        if idx == 3:
            for item in self.user_ub[u]:
                for ca in self.item_bca[i]:
                    if self.adj_BCa[item][ca] == 1:
                        path.append([u, item, ca, i])
            n = len(path)
            # print("idx: ", idx, " n: ", n)
            if n == 0: return [[self.n_user, self.n_item, self.n_ca, self.n_item]] * self.sample
            if n > self.sample: tmp = np.random.choice(range(n), size = self.sample, replace = False)
            else: tmp = np.random.choice(range(n), size = self.sample)
            return [path[t] for t in tmp]
        if idx == 4:
            for item in self.user_ub[u]:
                for ci in self.item_bci[i]:
                    if self.adj_BCi[item][ci] == 1:
                        path.append([u, item, ci, i])
            n = len(path)
            # print("idx:", idx, " n: ", n)
            if n == 0: return [[self.n_user, self.n_item, self.n_ci, self.n_item]] * self.sample
            if n > self.sample: tmp = np.random.choice(range(n), size = self.sample, replace = False)
            else: tmp = np.random.choice(range(n), size = self.sample)
            return [path[t] for t in tmp]
    def __getitem__(self, index):
        n_path = len(self.path_num)
        if self.mode == 'train':
            user = self.data[index]['user_id']
            items = [self.data[index]['business_id']] + self.sample_negative_item_for_user(user, self.n_negative)
            
            path_inputs = []
            for i in range(n_path):
                path_input = []
                for item in items:
                    feature_path = []
                    paths = self.get_path(user, item, i)
                    for path in paths:
                        feature_path.append(list(val for val in path))
                    path_input.append(feature_path)
                path_inputs.append(torch.tensor(path_input, dtype = torch.int64))
            return [user] * (self.n_negative + 1), items, [1.0] + [0.0] * self.n_negative, path_inputs
        else:
            user = self.data[index]['user_id']
            pos_n = len(self.data[index]['pos_business_id'])
            neg_n = len(self.data[index]['neg_business_id'])
            items = self.data[index]['pos_business_id'] + self.data[index]['neg_business_id']

            path_inputs = []
            for i in range(n_path):
                path_input = []
                for item in items:
                    feature_path = []
                    paths = self.get_path(user, item, i)
                    for path in paths:
                        feature_path.append(list([val] for val in path))
                    path_input.append(feature_path)
                path_inputs.append(torch.tensor(path_input, dtype = torch.int64))
            return [user] * (pos_n + neg_n), items, path_inputs, [1.0] * pos_n + [0.0] * neg_n, pos_n, neg_n
    # path_inputs[0], path_inputs[1], path_inputs[2], path_inputs[3] , path_inputs[4]
    def __len__(self):
        return len(self.data)
