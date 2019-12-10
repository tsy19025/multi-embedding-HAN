import numpy as np
from torch.utils.data import Dataset
import pickle

class YelpDataset(Dataset):
    def __init__(self, data_path, adj_paths, neighbor_size):
        self.neighbor_size = neighbor_size
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.adjs = [] # adj_UU, adj_UB, adj_BCa, adj_BCi, adj_UUB, adj_UBU, adj_UBUB, adj_UBCa, adj_UBCi, adj_BCaB, adj_BCiB
        for adj_path in adj_paths:
            with open(adj_path, 'rb') as f:
                self.adjs.append(pickle.load(f))

    def __getitem__(self, index):
        user = self.data[index]["user"]
        business = self.data[index]["business"]
        label = self.data[index]["rate"]
        user_neigh_list_lists = []
        business_neigh_list_lists = []
        user_user_adjs = [self.adjs[0], self.adjs[5]]
        user_business_adjs = [self.ajds[1], self.adjs[4], self.ajds[6]]
        user_city_adjs = [self.adjs[8]]
        user_category_adjs = [self.adjs[7]]
        user_neigh_adjs = [user_user_adjs, user_business_adjs, user_city_adjs, user_category_adjs]
        for adjs in user_neigh_adjs:
            user_neigh_list = []
            for adj in adjs:
                neighbors_index = np.nonzero(adj[user])[0]
                if len(neighbors_index) < self.neighbor_size:
                    neighbors = np.random.choice(neighbors_index, size=self.neighbor_size, replace=True)
                else:
                    neighbors = np.random.choice(neighbors_index, size=self.neighbor_size, replace=False)
                user_neigh_list.append(neighbors)
            user_neigh_list_lists.append(user_neigh_list)
        business_user_adjs = [self.adjs[1].T, self.adjs[4].T, self.adjs[6].T]
        business_business_adjs = [self.adjs[9], self.adjs[10]]
        business_city_adjs = [self.adjs[3]]
        business_category_adjs = [self.adjs[2]]
        business_neigh_adjs = [business_user_adjs, business_business_adjs, business_city_adjs, business_category_adjs]
        for adjs in business_neigh_adjs:
            business_neigh_list = []
            for adj in adjs:
                neighbors_index = np.nonzero(adj[user])[0]
                if len(neighbors_index) < self.neighbor_size:
                    neighbors = np.random.choice(neighbors_index, size=self.neighbor_size, replace=True)
                else:
                    neighbors = np.random.choice(neighbors_index, size=self.neighbor_size, replace=False)
                business_neigh_list.append(neighbors)
            business_neigh_list_lists.append(business_neigh_list)
        return user, business, label, user_neigh_list_lists, business_neigh_list_lists

    def __len__(self):
        return len(self.data)
