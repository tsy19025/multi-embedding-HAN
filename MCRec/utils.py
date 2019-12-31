import torch
from torch.utils.data import Dataset
import numpy as np
import random
import sys
import time

'''
def get_path(adj_BCa, adj_BCi, adj_UB, adj_UU):
    adj_BU = adj_UB.T
    adj_CaB = adj_BCa.T
    adj_CiB = adj_BCi.T

    ub_path = []
    uub_path = []
    ubub_path = []
    ubcab_path = []
    ubcib_path = []

    users, items = adj_UB.shape
    Cas = adj_BCa.shape[1]
    Cis = adj_BCi.shape[1]
    print("n_user: ", users)
    print("n_item: ", items)
    print("n_ca: ", Cas)
    print("n_ci: ", Cis)
    print("Generate paths")
    max_path = 0
    for i in range(users):
        items = np.nonzero(adj_UB[i])[0]
        ub = []
        ubcab = []
        ubcib = []
        for item in items:
            ub.append(str(i)+'-'+str(item))
            cas = np.randome_np.nonzero(adj_BCa[item])[0]
            print(cas)
            cis = np.nonzero(adj_BCi[item])[0]
            print(cis)
            for ca in cas:
                tmp_items = np.nonzero(adj_CaB[ca])[0]
                print(len(tmp_items))
                for tmp_item in tmp_items:
                    ubcab.append(str(i) + '-' + str(item) + '-' + str(ca) + '-' + str(tmp_item))
            for ci in cis:
                tmp_items = np.nonzero(adj_CiB[ci])[0]
                for tmp_item in tmp_items:
                    ubcib.append(str(i) + '-' + str(item) + '-' + str(ci) + '-' + str(tmp_item))
        ub_path.append(ub)
        ubcab_path.append(ubcab)
        ubcib_path.append(ubcib)
        print(len(ubcab))
        print(len(ubcib))
        sys.exit(0)
        max_path = max(max_path, len(ubcab))
        max_path = max(max_path, len(ubcib))
    print("Step1 end.")
    print(max_path)
    sys.exit(0)

    for i in range(users):
        ubub = []
        for j in range(users):
            for pathi in ub_path[i]:
                for pathj in ub_path[j]:
                    ubub.append(pathi+'-'+pathj)
        ubub_path.append(ubub)
    print("Step2 end.")

    for i in range(users):
        users = np.nonzero(adj_UU[i])[0]
        uub = []
        for j in users:
            for pathj in ub_path[j]:
                uub.append(str(i) + '-' + pathj)
        uub_path.append(uub)
    print("Step3 end.")
    return ub_path, uub_path, ubub_path, ubcab_path, ubcib_path
'''

def get_path(adj_BCa, adj_BCi, adj_UB, adj_UU, sample, sample_a_size = 50, sample_b_size = 50):
    paths = [{}, {}, {}, {}, {}]
    adj_BU = adj_UB.T
    users, items = adj_UB.shape
    print(users, items)
    # sample 50
    item_bu = []
    item_bca = []
    item_bci = []
    for i in range(items):
        bu = np.nonzero(adj_BU[i])[0]
        bca = np.nonzero(adj_BCa[i])[0]
        bci = np.nonzero(adj_BCi[i])[0]

        if len(bu) > sample_b_size: bu = np.random.choice(bu, size = sample_b_size, replace = False)
        if len(bca) > sample_b_size: bca = np.random.choice(bca, size = sample_b_size, replace = False)
        if len(bci) > sample_b_size: bci = np.random.choice(bci, size = sample_b_size, replace = False)

        item_bu.append(bu)
        item_bca.append(bca)
        item_bci.append(bci)

    begin_ticks = time.time()
    for u in range(users):
        print(u)
        end_ticks = time.time()
        print(end_ticks - begin_ticks)
        user_uu = np.nonzero(adj_UU[u])[0]
        user_ub = np.nonzero(adj_UB[u])[0]

        if len(user_ub) > sample_a_size: user_ub = np.random.choice(user_ub, size = sample_a_size, replace = False)
        for i in range(items):
            # print("i:", i)
            # ub:
            if adj_UB[u][i] == 1: paths[0][(u, i)] = [[u, i]]
            else: paths[0][(u, i)] = [[0, 0]]

            # uub:
            path = [user for user in user_uu if user in item_bu[i]]
            n = len(path)
            # print(n)
            # print(i, " : ", n)
            if n > sample: path = np.random.choice(path, size = sample, replace = False)
            elif n > 0: path = np.random.choice(path, size = sample, replace = True)
            paths[1][(u, i)] = np.array(list([u, user, i] for user in path))
            if n <= 0: paths[1][(u, i)] = np.zeros([sample, 3])

            # ubub:
            # print("ubub")
            path = []
            for item in user_ub:
                for user in item_bu[i]:
                    if adj_UB[user][item] == 1: path.append([u, item, user, i])
            n = len(path)
            # print(n)
            if n > sample: tmp = np.random.choice(range(n), size = sample, replace = False)
            elif n > 0: tmp = np.random.choice(range(n), size = sample, replace = True)
            else:
                path = [[0, 0, 0, 0]] * sample
                tmp = [0] * sample
            paths[2][(u, i)] = np.array(list(path[t] for t in tmp))

            # ubcab
            # print("ubcab")
            path = []
            for item in user_ub:
                for ca in item_bca[i]:
                    if adj_BCa[item][ca] == 1: path.append([u, item, ca, i])
            n = len(path)
            # print(n)
            if n > sample: tmp = np.random.choice(range(n), size = sample, replace = False)
            elif n > 0: tmp = np.random.choice(range(n), size = sample, replace = True)
            else:
                path = [[0, 0, 0, 0]] * sample
                tmp = [0] * sample
            paths[3][(u, i)] = np.array(list(path[t] for t in tmp))

            # ubcib
            # print("ubcib")
            path = []
            for item in user_ub:
                for ci in item_bci[i]:
                    if adj_BCi[item][ci] == 1: path.append([u, item, ci, i])
            n = len(path)
            # print(n)
            if n > sample: tmp = np.random.choice(range(n), size = sample, replace = False)
            elif n > 0: tmp = np.random.choice(range(n), size = sample, replace = True)
            else:
                path = [[0, 0, 0, 0]] * sample
                tmp = range(sample)
            paths[4][(u, i)] = np.array(list(path[t] for t in tmp))

            # print(paths[0][(u, i)], paths[1][(u, i)], paths[2][(u, i)], paths[3][(u, i)], paths[4][(u, i)])
    return paths

class YelpDataset(Dataset):
    def __init__(self, n_type, data, paths, path_num, timestamps, adj_UB, negetives, mode):
        self.n_user = n_type[0]
        self.n_item = n_type[1]
        self.data = data
        self.path_num = path_num
        self.timestamps = timestamps
        self.n_negetive = negetives
        self.adj_UB = adj_UB
        self.mode = mode
        self.paths = paths

    def sample_negetive_item_for_user(self, user, negetive):
        items = []
        for t in range(negetive):
            item = np.random.randint(0, self.n_item)
            while self.adj_UB[user][item] == 1:
                item = np.random.randint(0, self.n_item)
            items.append(item)
        # print("user: ", user, item)
        return items

    def __getitem__(self, index):
        if self.mode == 'train':
            user = self.data[index]['user_id']
            items = [self.data[index]['business_id']] + self.sample_negetive_item_for_user(user, self.n_negetive)
            
            path_inputs = []
            for i in range(len(self.paths)):
                path_input = []
                for item in items:
                    feature_path = []
                    for path in self.paths[i][(user, item)]:
                        feature_path.append(list(val for val in path))
                    path_input.append(feature_path)
                path_inputs.append(torch.tensor(path_input, dtype = torch.int64))
            return [user] * (self.n_negetive + 1), items, [1.0] + [0.0] * self.n_negetive, path_inputs
        else:
            user = self.data[index]['user_id']
            pos_n = len(self.data[index]['pos_business_id'])
            neg_n = len(self.data[index]['neg_business_id'])
            items = self.data[index]['pos_business_id'] + self.data[index]['neg_business_id']

            path_inputs = []
            for i in range(len(self.paths)):
                path_input = []
                for item in items:
                    feature_path = []
                    for path in self.paths[i][(user, item)]:
                        feature_path.append(list([val] for val in path))
                    path_input.append(feature_path)
                path_inputs.append(torch.tensor(path_input, dtype = torch.int64))
            return [user] * (pos_n + neg_n), items, [1.0] * pos_n + [0.0] * neg_n, path_inputs, pos_n, neg_n
    # path_inputs[0], path_inputs[1], path_inputs[2], path_inputs[3] , path_inputs[4]
    def __len__(self):
        return len(self.data)
