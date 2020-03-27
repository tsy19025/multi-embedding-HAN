import torch
from torch.utils.data import Dataset
import numpy as np

class YelpDataset(Dataset):
    def __init__(self, data, negatives, adj_UB, mode, u_embedding, i_embedding):
        self.n_user, self.n_item = adj_UB.shape
        self.data = data
        self.n_negative = negatives
        self.adj_UB = adj_UB
        self.mode = mode
        self.u_embedding = u_embedding
        self.i_embedding = i_embedding
    
    def get_userfeature(self, user):
        feature = []
        for embedding in self.u_embedding:
            feature.append(embedding[user])
        return feature
    def get_itemfeature(self, item):
        feature = []
        for embedding in self.i_embedding:
            feature.append(embedding[item])
        return feature

    def sample_negative_item_for_user(self, user, negative):
        items = []
        for t in range(negative):
            item = np.random.randint(0, self.n_item)
            while self.adj_UB[user][item] == 1:
                item = np.random.randint(0, self.n_item)
            items.append(self.get_itemfeature(item))
        return items

    def __getitem__(self, index):
        if self.mode == 'train':
            user = self.data[index]['user_id']
            items = [self.get_itemfeature(self.data[index]['business_id'])] + self.sample_negative_item_for_user(user, self.n_negative)
            return np.array([self.get_userfeature(user)] * (self.n_negative + 1), dtype=np.float32), np.array(items, dtype=np.float32), np.array([1.0] + [0.0] * self.n_negative, dtype=np.float32)
        else:
            user = self.get_userfeature(self.data[index]['user_id'])
            pos_n = len(self.data[index]['pos_business_id'])
            neg_n = len(self.data[index]['neg_business_id'])
            items = list(self.get_itemfeature(item) for item in self.data[index]['pos_business_id']) + list(self.get_itemfeature(item) for item in self.data[index]['neg_business_id'])
            return np.array([user] * (pos_n + neg_n), dtype=np.float32), np.array(items, dtype=np.float32), np.array([1.0] * pos_n + [0.0] * neg_n, dtype=np.float32)
    def __len__(self):
        return len(self.data)
