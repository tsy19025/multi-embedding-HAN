import os
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset

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

def make_labels(Y, n_user, n_item):
    r"""
    Parameter
    ---------
    Y: list of dict
        saves the interaction information in COO form
    
    Return
    ------
    ret: numpy.ndarray

    pos_sampleset_list: list of set
    """
    if 'business_id' in Y[0].keys():    # this is for training data, need a pos_sampleset_list
        pos_sampleset_list = [set() for i in range(n_user)]
        for y in Y:
            pos_sampleset_list[y['user_id']].add(y['business_id'])
        ret = np.asarray([[y['user_id'], y['business_id'], 1] for y in Y])
        return ret, pos_sampleset_list
        
    elif 'neg_business_id' in Y[0].keys():  # this is for valid data and test data
        pos = [[y['user_id'], pos_id, 1] for y in Y for pos_id in y['pos_business_id']]
        neg = [[y['user_id'], neg_id, 0] for y in Y for neg_id in y['neg_business_id']]
        ret = pos + neg
        ret = np.asarray(ret)   # without copying the data again
        return ret

def make_dataset(data, cuda=False):
    r"""
    Parameters
    ----------
    data: labels with user id and item id, type: list

    Return
    ------
    dataset: torch.utils.data.Dataset
    """
    device = torch.device('cuda:0' if cuda else 'cpu')
    x_tensor = torch.as_tensor(data[:, 0:2], dtype=torch.long, device=device)   # avoid copying memory
    y_tensor = torch.as_tensor(data[:, 2], dtype=torch.long, device=device)
    x_tensor.requires_grad_(False)
    y_tensor.requires_grad_(False)
    dataset = TensorDataset(x_tensor, y_tensor)
    return dataset

if __name__ == "__main__":
    pass