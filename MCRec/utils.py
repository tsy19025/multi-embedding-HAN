import torch
from torch.utils.data import Dataset
import numpy as np
import random
import sys

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

def get_path(adj_BCa, adj_BCi, adj_UB, adj_UU, sample = 5):
    paths = [{}, {}, {}, {}, {}]
    adj_BU = adj_UB.T
    users, items = adj_UB.shape
    # sample 50
    print("n_user: ", users)
    print("n_item: ", items)
    for u in range(users):
        for i in range(items):
            # ub:
            if adj_UB[u][i] == 1: paths[0][(u, i)] = [u, i]
            else: paths[0][(u, i)] = []

            # uub:
            a = np.nonzero(adj_UU[u])[0]
            b = np.nonzero(adj_BU[i])[0]
            path = [user for user in a if user in b]
            n = len(path)
            if n > sample: path = np.random.choice(path, size = sample, replace = False)
            elif n > 0: path = np.random.choice(path, size = sample, replace = True)
            paths[1][(u, i)] = list([[u, user, i] for user in path])

            # ubub:
            path = []
            a = np.nonzero(adj_UB[u])[0]
            for item in a:
                for user in b:
                    if adj_UB[user][item] == 1: path.append([u, item, user, i])
            n = len(path)
            if n > sample: tmp = np.random.choice(range(n), size = sample, replace = False)
            elif n > 0: tmp = np.random.choice(range(n), size = sample, replace = True)
            else: tmp = []
            paths[2][(u, i)] = list(path[t] for t in tmp)

            # ubcab
            path = []
            b = np.nonzero(adj_BCa[i])[0]
            for item in a:
                for ca in b:
                    if adj_BCa[item][ca] == 1: path.append([u, item, ca, i])
            n = len(path)
            if n > sample: tmp = np.random.choice(range(n), size = sample, replace = False)
            elif n > 0: tmp = np.random.choice(range(n), size = sample, replace = True)
            else: tmp = []
            paths[3][(u, i)] = list(path[t] for t in tmp)

            # ubcib
            path = []
            b = np.nonzero(adj_BCi[i])[0]
            for item in a:
                for ci in b:
                    if adj_BCi[item][ci] == 1: path.append([u, item, ci, i])
            n = len(path)
            if n > sample: tmp = np.random.choice(range(n), size = sample, replace = False)
            elif n > 0: tmp = np.random.choice(range(n), size = sample, replace = True)
            else: tmp = []
            paths[4][(u, i)] = list(path[t] for t in tmp)

            # print(paths[0][(u, i)], paths[1][(u, i)], paths[2][(u, i)], paths[3][(u, i)], paths[4][(u, i)])

    return paths

def YelpDataset(Dataset):
    def __init__(self, users, items, data_path, paths, path_num, timestamps, negetive):
        self.n_user = users
        self.n_item = items
        self.path_num = path_num
        self.timestamps = timestamps
        self.n_negetive = negetives

        with open(data_path, 'rb') as f:
            self.data = (pickle.load(f))
        self.paths = paths
    def sample_negetive_item_for_user(self, user, negetive):
        items = np.nonzero(self.adj_UB[user])[0]
        if len(items) > negetive: items = np.random_choice(items, size = negetive, replace = False)
        else: items = np.random.choice(items, size = negetive, replace = True)
        return items

    def __getitem__(self, index):
        user = self.data[index]['user_id']
        items = [self.data[index]['business_id']] + sample_negetive_item_for_user(user, self.n_negetive)
        
        path_inputs = []
        for item in items:
            path_inputs.append([path[(user, item)] for path in self.paths])
        '''
            for i in range(len(self.paths)):
                path = self.paths[i]
                path_input = 
                if item in path[user]:
                    for pp in path[user][item]:
                        tmp_path = pp.split('-')
                        tmp_path = [int(tmp) for tmp in tmp_path]
                        if self.timestamps[i] > len(tmp_path): tmp_path = tmp_path + [0] * (self.timestamps[i] - len(tmp_path))
                        path_input.append(tmp_path)
                if self.path_num[i] > len(path_input):
                    nn = self.path_num[i] - len(path_input)
                    for j in range(nn):
                        path_input.append([0] * self.timestamps[i])
                paths.append(path_input)
            path_inputs.append(paths)
        '''
        return user, items, [1] + [0] * self.n_negetive, torch.tensor(path_inputs)
    def __len__(self):
        return len(self.data)
