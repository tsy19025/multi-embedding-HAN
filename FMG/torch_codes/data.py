import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from utils import *


def load_feature(feature_path, metapaths):
    user_features = [read_pickle(feature_path+metapath+'_user.pickle') for metapath in metapaths]
    item_features = [read_pickle(feature_path+metapath+'_item.pickle') for metapath in metapaths]
    return user_features, item_features

class FMG_YelpDataset(Dataset):
    def __init__(self, interaction_data, user_features, item_features, neg_sample_n, mode, cuda=False):
        r"""
        Parameters
        ----------
        data: list of dict
        
        neg_sample_n: int,
            number of negative samples to be sampled
            
        mode: 'train', 'valid' or 'test',
            among which 'valid' and 'test' have the same functions
            
        Returns
        -------
        If mode == 'train':
        users: list of user ids
        
        items: list of business ids
        
        labels: corresponding labels
        
        If mode == 'valid' or 'test':
        user: list, 
            only one user id, but list size is len(items)
        
        items: list,
            positive samples and negative samples
            
        labels: corresponding labels
        """
        super().__init__()
        self.neg_sample_n = neg_sample_n
        self.mode = mode
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.user_concats = torch.cat(user_features, 1)     # a tensor with size [n_user, n_metapath*n_factor]
        self.item_concats = torch.cat(item_features, 1)
        self.n_user = self.user_concats.shape[0]
        self.n_item = self.item_concats.shape[0]
        self.item_ids = set(i for i in range(self.n_item))

        if self.mode == 'train':
            pos_sampleset_list = [set() for i in range(self.n_user)]
            for y in interaction_data:
                pos_sampleset_list[y['user_id']].add(y['business_id'])            
            self.pos_sampleset_list = pos_sampleset_list    # used for train set
            self.data = torch.tensor(np.asarray([[y['user_id'], y['business_id'], 1] for y in interaction_data]), 
                                     device=self.device)

        elif self.mode == 'valid' or self.mode == 'test':
            self.data = []
            for input in interaction_data:
                pos = [i for i in input['pos_business_id']]
                neg = [i for i in input['neg_business_id']]
                items = pos + neg[0:self.neg_sample_n]
                user = [input['user_id']] * len(items)
                labels = [1] * len(pos) + [0] * self.neg_sample_n
                self.data.append([np.asarray(user), 
                                  np.asarray(items), 
                                  torch.tensor(np.asarray(labels), dtype=torch.long, device=self.device)])
        
    def make_embedding(self, user_ids, item_ids):
        r"""
        embed is too large to store in even memory, 
        thus we can only generate each embedding when loading the data

        Parameters
        ----------
        user_ids: list or 1-D np.ndarray

        item_ids: list or 1-D np.ndarray

        Return
        ------
        embed: torch.tensor with size(\[ batch_size, 2*L*K \])
        """
        embed = []
        for uid, bid in zip(user_ids, item_ids):
            user = self.user_concats[uid]
            item = self.item_concats[bid]
            embed_concat = torch.cat([user,item], 0).unsqueeze(0)
            embed.append(embed_concat)
        embed = torch.cat(embed, 0).to(self.device)
        return embed

    def __getitem__(self, index):
        r"""
        ps: index is a number.
        
        return the uid and all pos bids and neg bids, along with labels.
        
        all in python list
        """
        if self.mode == 'train':
            pos_ind = self.data[index][0:2]
            user = pos_ind[0].item()    # user id
            pos_ind = pos_ind.unsqueeze(0)
            neg_sample_array = np.asarray(list(self.item_ids - self.pos_sampleset_list[user]))
            neg_samples = np.random.choice(neg_sample_array, self.neg_sample_n, replace=False)

            neg_inds = torch.tensor(np.asarray([[user, neg_sample] for neg_sample in neg_samples]), device=self.device)
            indices = torch.cat((pos_ind, neg_inds), 0)

            embed = self.make_embedding(indices[:, 0], indices[:, 1])

            labels = torch.tensor(np.asarray([1] + [0]*self.neg_sample_n), dtype=torch.long, device=self.device)
            return embed, labels

        elif self.mode == 'valid' or self.mode == 'test':
            r"""
            here the data contains 'pos_business_id' and 'neg_business_id' keys.
            """
            user_ids = self.data[index][0]
            item_ids = self.data[index][1]
            labels   = self.data[index][2]

            embed = self.make_embedding(user_ids, item_ids)
            
            return embed, item_ids, labels
        
    def __len__(self):
        return len(self.data)    # available for both list and np.ndarray


if __name__ == "__main__":
    pass
