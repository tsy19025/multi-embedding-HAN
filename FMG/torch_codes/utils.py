import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def read_pickle(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def write_pickle(path, data):
    with open(path, 'wb') as fw:
        pickle.dump(data, fw);

def load_feature(feature_path, metapaths):
    user_features = [read_pickle(feature_path+metapath+'_user.pickle') for metapath in metapaths]
    item_features = [read_pickle(feature_path+metapath+'_item.pickle') for metapath in metapaths]
        
    return user_features, item_features

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
    """
    if 'business_id' in Y[0].keys():
        ret = [[y['user_id'], y['business_id'], 1] for y in Y]
    elif 'neg_business_id' in Y[0].keys():
        pos = [[y['user_id'], pos_id, 1] for y in Y for pos_id in y['pos_business_id']]
        neg = [[y['user_id'], neg_id, 0] for y in Y for neg_id in y['neg_business_id']]
        ret = pos + neg
    ret = np.asarray(ret)   # without copying the data again
    return ret

if __name__ == "__main__":
    pass
