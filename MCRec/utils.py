import torch
from torch.utils.data import Dataset
import numpy as np

def get_path(adj_BCa, adj_BCi, adj_UB, adj_UU):
    adj_BU = adj_UB.T
    adj_CaB = adj_BCa.T
    adj_CiB = adj_BCi.T

    ub_path = []
    uub_path = []
    ubub_path = []
    ubcab_path = []
    ubcib_path = []

    users = adj_UU.shape[0]
    print("n_user: ", users)
    print("Generate paths")
    for i in range(users):
        items = np.nonzero(adj_UB[i])[0]
        ub = []
        ubcab = []
        ubcib = []
        for item in items:
            ub.append(str(i)+'-'+str(item))
            cas = np.nonzero(adj_BCa[item])[0]
            cis = np.nonzero(adj_BCi[item])[0]
            for ca in cas:
                tmp_items = np.nonzero(adj_CaB[ca])[0]
                for tmp_item in tmp_items:
                    ubcab.append(str(i) + '-' + str(item) + '-' + str(ca) + '-' + str(tmp_item))
            for ci in cis:
                tmp_items = np.nonzero(adj_CiB[ci])[0]
                for tmp_item in tmp_items:
                    ubcib.append(str(i) + '-' + str(item) + '-' + str(ci) + '-' + str(tmp_item))
        ub_path.append(ub)
        ubcab_path.append(ubcab)
        ubcib_path.append(ubcib)
    print("Step1 end.")

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
            uub.append(str(i)+'-'+pathj for pathj in ub_path[j])
        uub_path.append(uub)
    print("Step3 end.")
    return ub_path, uub_path, ubub_path, ubcab_path, ubcib_path

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

    def __getitem__(self, index):
        user = self.data[index]['user_id']
        items = [self.data[index]['business_id']] + sample_negetive_item_for_user(user, self.n_negetive)
        
        path_inputs = []
        for item in items:
            paths = []
            for i in range(len(self.paths)):
                path = self.paths[i]
                path_input = []
                if item in path[user]:
                    for tmp_path in path[user][item]:
                        tmp_path = tmp_path.split('-')
                        tmp_path = [int(tmp) for tmp in tmp_path]
                        if self.timestamps[i] > len(tmp_path): tmp_path = tmp_path + [0] * (self.timestamps[i] - len(tmp_path))
                        path_input.append(tmp_path)
                if self.path_num[i] > len(path_input):
                    nn = self.path_num[i] - len(path_input)
                    for j in range(nn):
                        path_input.append([0] * self.timestamps[i])
                paths.append(path_input)
            path_inputs.append(paths)
        return user, items, [1] + [0] * self.n_negetive, torch.tensor(path_inputs)
    def __len__(self):
        return len(self.data)
