import numpy as np 
import pandas as pd 
import pickle
import scipy.sparse as sp
import random

import torch.utils.data as data

import config

def read_pickle(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def write_pickle(path, data):
    with open(path, 'wb') as fw:
        pickle.dump(data, fw);

def load_all():
    """ We load all the three files here to save time in each epoch. """
    train_data = read_pickle(config.train_data)
    users = read_pickle(config.user_data)
    items = read_pickle(config.item_data)

    user_num = len(users)
    item_num = len(items)

    train_data = [[data['user_id'], data['business_id']] for data in train_data]

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = read_pickle(config.test_negative)

    gt_items = {entry['user_id']:entry['pos_business_id'] for entry in test_data}

    return train_data, test_data, train_mat, user_num, item_num


class BPRData(data.Dataset):
    def __init__(self, features, num_user,
                num_item, train_mat=None, num_ng=0, is_training=None):
        super(BPRData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.

            features are different in training and test
        """
        self.features = features
        self.num_user = num_user
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        # self.user_neg_dict = {u:set(range(num_item)) for u in range(num_user)}
        # user_pos_dict = {u:set() for u in range(num_user)}
        # if self.is_training:
        #     for x in features:
        #         user_pos_dict[x[0]].add(x[1])
        #     for u in user_pos_dict.keys():
        #         self.user_neg_dict[u] = list(self.user_neg_dict[u] - user_pos_dict[u])

        if not self.is_training:
            self.data = []
            for input in features:
                pos = [i for i in input['pos_business_id']]
                neg = [i for i in input['neg_business_id']]
                items = pos + neg
                user = [input['user_id']] * len(items)
                labels = [1] * len(pos) + [0] * len(neg)
                self.data.append([np.asarray(user), np.asarray(items), np.asarray(labels)])

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([u, i, j])

    def __len__(self):
        return self.num_ng * len(self.features) if \
                self.is_training else len(self.features)

    def __getitem__(self, idx):
        if self.is_training:
            features = self.features_fill
            user = features[idx][0]
            item_i = features[idx][1]
            item_j = features[idx][2]    
            return user, item_i, item_j

        else:
            user_ids = self.data[idx][0]
            item_ids = self.data[idx][1]
            labels   = self.data[idx][2]
            return user_ids, item_ids, labels        