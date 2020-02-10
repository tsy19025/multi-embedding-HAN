import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

def make_embedding(user_features, item_features, cuda=False):
    r"""
    Return
    ------
    embed: torch.tensor with size(\[ n_user, n_item, 2*L*K \])
    """
    user_concat = torch.cat(user_features, 1)
    item_concat = torch.cat(item_features, 1)
    embed = []
    for user in user_concat:
        tmp = [torch.cat([user,item], 0).unsqueeze(0) for item in item_concat]
        tmp = torch.cat(tmp, 0)
        embed.append(tmp.unsqueeze(0))
    embed = torch.cat(embed, 0)
    device = torch.device('cuda:0' if cuda else 'cpu')
    embed.to(device)
    return embed

class FMG_YelpDataset(Dataset):
    def __init__(self, interaction_data, n_user, n_item, neg_sample_n, mode, cuda=False):
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
        self.n_user = n_user
        self.n_item = n_item
        self.item_ids = set(i for i in range(self.n_item))
        self.device = torch.device('cuda:0' if cuda else 'cpu')

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
                self.data.append([torch.tensor(np.asarray(user), device=self.device), 
                                  torch.tensor(np.asarray(items),device=self.device), 
                                  torch.tensor(np.asarray(labels), dtype=torch.long, device=self.device)])
        
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
            labels = torch.tensor(np.asarray([1] + [0]*self.neg_sample_n), dtype=torch.long, device=self.device)
            return indices, labels

        elif self.mode == 'valid' or self.mode == 'test':
            r"""
            here the data contains 'pos_business_id' and 'neg_business_id' keys.
            """
            return self.data[index]
        
    def __len__(self):
        return len(self.data)    # available for both list and np.ndarray


if __name__ == "__main__":
    pass