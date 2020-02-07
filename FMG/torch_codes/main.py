import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import make_dataset
from loss import MFLoss
from model import FactorizationMachine, MatrixFactorization, MatrixFactorizer
from utils import *

gettime = lambda: time.time()

def train_MF(metapaths, loadpath, savepath, reg_user=5e-2, reg_item=5e-2, lr=1e-2, epoch=5000, cuda=False):
    r"""
    Parameters
    ----------
    metapaths: list
        list of metapaths

    epoch: int
        number of epochs
    """
    i = 0
    for metapath in metapaths:
        # instance the MF trainer
        MFTrainer(metapath, loadpath, savepath, epoch[i], lr=lr[i], reg_user=reg_user[i], reg_item=reg_item[i], cuda=cuda)
        i += 1


def MFTrainer(metapath, loadpath, savepath, epochs=20, n_factor=3, 
            lr=1e-4, reg_user=5e-2, reg_item=5e-2, decay_step=30, decay=0.1, 
            cuda=False):
    device = torch.device('cuda:0' if cuda else 'cpu')

    def _load_data(filepath, metapath, device):
        data = []
        file = filepath + 'adj_' + metapath + '.pickle'
        with open(file, 'rb') as fw:
            adjacency = pickle.load(fw)
            data = torch.tensor(adjacency, dtype=torch.float32, requires_grad=False).to(device)
        n_user, n_item = data.shape
        return n_user, n_item, data

    n_user, n_item, adj_mat = _load_data(loadpath, metapath, device)
    mf = MatrixFactorizer(n_user, n_item, n_factor, cuda).to(device)
    # mf = MatrixFactorization(n_user, n_item, n_factor, cuda).to(device)

    prev_loss = 0
    # set loss function
    criterion = MFLoss(reg_user, reg_item)
    optimizer = torch.optim.Adam([mf.user_factors, mf.item_factors], lr=lr)  # use weight_decay
    # scheduler = StepLR(optimizer, step_size=decay_step, gamma=decay)
    mf.train()
    print("n_user: %d, n_item: %d" % (n_user, n_item))
    for i in range(epochs + 1):
        # scheduler.step()
        optimizer.zero_grad()
        adj_t = mf()
        loss = criterion(mf.user_factors, mf.item_factors, adj_t, adj_mat)   # this line is ugly
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("metapath: %s, epoch %d: loss = %.4f, lr = %.10f, reg_user = %f, reg_item = %f" 
                % (metapath, i, loss, lr, reg_user, reg_item))
            # if abs(int(loss) - prev_loss) < 1e-2:
            #     break
            # prev_loss = int(loss)
            
    mf.export(savepath, metapath)

def train_FM(embed, train_data, valid_data, epochs=500, lr=1e-4, criterion=None, cuda=False):
    r"""
    Parameters
    ----------
    model: nn.Module
        the FM model

    train_data, valid_data: torch.utils.data.Dataset
        each line is [uid, bid, rate]

    criterion: loss function class,
        Default is nn.CrossEntropyLoss
    """
    device = torch.device('cuda:0' if cuda else 'cpu')

    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=10, shuffle=True)

    FM = FactorizationMachine(embed.shape[2], 10, cuda).to(device)
    criterion = nn.CrossEntropyLoss()
    # if criterion is not None:
    #     self.criterion = criterion
    criterion.to(device)

    # set optimizer
    optimizer = torch.optim.Adam([FM.V, FM.W, FM.w0], lr=lr)  # maybe use weight_decay
    FM.train()
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            indices, y = data       # indices: [[i, j], [i, j], ...]
            x = embed[indices[:, 0], indices[:, 1]]
            y_t = FM(x)
            loss = criterion(y_t, y)
            loss.backward()
            optimizer.step()

        # if epoch % 50 == 0:
        print("epoch %d, loss = %f, lr = %f" % (epoch, loss, lr))
        if epoch % 100 == 0:
            # valid evaluate
            pass

    FM.export()
    
if __name__ == "__main__":
    filtered_path = '../yelp_dataset/filtered/'
    adj_path = '../yelp_dataset/adjs/'
    feat_path = '../yelp_dataset/mf_features/'
    rate_path = '../yelp_dataset/rates/'

    # train MF
    metapaths = ['UB', 'UBUB', 'UUB', 'UBCaB', 'UBCiB']
    t0 = gettime()
    # train_MF(metapaths, 
    #          adj_path, 
    #          feat_path, 
    #          epoch=[5000, 5000, 5000, 20000, 5000], 
    #          lr=[2e-3, 5e-2, 5e-3, 5e-3, 5e-3], 
    #          reg_user=[1e-1, 1e-1, 1e-1, 1e-1, 1e-1], 
    #          reg_item=[1e-1, 1e-1, 1e-1, 1e-1, 1e-1], cuda=True)
    t1 = gettime()
    print("time cost: %f" % (t1 - t0))

    # train FM (cross entropy loss)
    t0 = gettime()
    print("loading data...")
    train_data = read_pickle(rate_path+'train_data.pickle')
    # valid_data = read_pickle(rate_path+'valid_data.pickle')
    # do we need negative samples? Yes, we do!
    valid_data = read_pickle(rate_path+'valid_with_neg_sample.pickle')
    print("time cost: %f" % (gettime() - t0))

    t0 = gettime()
    print("loading features and (make) embeddings...")
    # small dataset
    users = read_pickle(filtered_path+'users-small.pickle')
    businesses = read_pickle(filtered_path+'businesses-small.pickle')
    # full dataset
    # users = read_pickle(filtered_path+'users-complete.pickle')
    # businesses = read_pickle(filtered_path+'businesses-complete.pickle')
    if os.path.exists(feat_path+'embed.pickle'):
        embed = read_pickle(feat_path+'embed.pickle')
    else:
        user_features, item_features = load_feature(feat_path, metapaths)
        embed = make_embedding(user_features, item_features, cuda=True)    # in this way, we can use embed[uid][bid] to find the embedding of user-item pair
        write_pickle(feat_path+'embed.pickle', embed)
    print("time cost: %f" % (gettime() - t0))

    t0 = gettime()
    print("making datasets...")
    n_users = len(users)
    n_items = len(businesses)
    print("n_users:", n_users, "n_items:", n_items)

    train_data = make_dataset(make_labels(train_data, n_users, n_items), cuda=True)
    valid_data = make_dataset(make_labels(valid_data, n_users, n_items), cuda=True)
    print("time cost: %f" % (gettime() - t0))

    t0 = gettime()
    print("start training FM...")
    train_FM(embed, train_data, valid_data, 1000, cuda=True)
    print("time cost: %f" % (gettime() - t0))
