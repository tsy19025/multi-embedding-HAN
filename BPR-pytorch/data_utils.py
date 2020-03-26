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

def load_all(test_num=100):
    """ We load all the three files here to save time in each epoch. """
    # train_data = pd.read_csv(
    # 	config.train_rating, 
    # 	sep='\t', header=None, names=['user', 'item'], 
    # 	usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    train_data = read_pickle(config.train_data)
    users = read_pickle(config.user_data)
    items = read_pickle(config.item_data)

    # user_num = train_data['user'].max() + 1
    # item_num = train_data['item'].max() + 1
    user_num = len(users)
    item_num = len(items)

    train_data = [[data['user_id'], data['business_id']] for data in train_data]

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    # with open(config.test_negative, 'r') as fd:
    # 	line = fd.readline()
    # 	while line != None and line != '':
    # 		arr = line.split('\t')
    # 		u = eval(arr[0])[0]
    # 		test_data.append([u, eval(arr[0])[1]])
    # 		for i in arr[1:]:
    # 			test_data.append([u, int(i)])
    # 		line = fd.readline()

    # allocate each user 1 positive item and 50 negative samples
    test_neg = read_pickle(config.test_negative)
    for entry in test_neg:
        user = entry['user_id']
        pos = random.choice(entry['pos_business_id'])
        test_data.append([user, pos])
        for neg in entry['neg_business_id']:
            test_data.append([user, neg])

    return train_data, test_data, user_num, item_num, train_mat


class BPRData(data.Dataset):
    def __init__(self, features, 
                num_item, train_mat=None, num_ng=0, is_training=None):
        super(BPRData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        
        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for t in range(self.num_ng):    # generate `num_ng` neg samples for each pos sample
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([u, i, j])

    def __len__(self):
        return self.num_ng * len(self.features) if \
                self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if \
                self.is_training else self.features

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if \
                self.is_training else features[idx][1]
        # when testing, the negative samples are already in the dataset: 
        # each user has 1 positive sample and 99 negative samples.
        return user, item_i, item_j 
        