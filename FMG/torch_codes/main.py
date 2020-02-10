import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import *
from loss import MFLoss
from metrics import *
from model import FactorizationMachine, MatrixFactorizer
from utils import *

gettime = lambda: time.time()

def eval(embed, valid_data, model, criterion, topK):
    r"""
    Calculate precision, recall and ndcg of the prediction of the model.
    Returns
    -------
    precision, recall, ndcg
    """
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True)
    model.eval()

    loss_list = []
    prec_list = []
    recall_list = []
    ndcg_list = []
    for i, data in enumerate(valid_loader):
        user, items, labels = data
        user = user.view(-1)
        items = items.view(-1)
        labels = labels.view(-1)
        x = embed[user, items]
        out = model(x)
        # Use a Sigmoid to classify
        y_t = torch.sigmoid(out)
        # print(out)
        # y_t = out.clone()
        y_t = y_t.unsqueeze(1).repeat(1, 2)
        y_t[:, 1] = 1 - y_t[:, 0]
        loss = criterion(y_t, labels)
        values, indices = torch.topk(out, topK)
        ranklist = items[indices]
        gtItems = items[torch.nonzero(labels)[:, 0]].tolist()
        loss_list.append(loss.item())
        prec_list.append(getP(ranklist, gtItems))
        recall_list.append(getR(ranklist, gtItems))
        ndcg_list.append(getNDCG(ranklist, gtItems))

    loss = np.mean(np.asarray(loss_list))
    prec = np.mean(np.asarray(prec_list))
    recall = np.mean(np.asarray(recall_list))
    ndcg = np.mean(np.asarray(ndcg_list))

    print("evaluation: loss: %f, precision@%d: %f, recall@%d: %f, NDCG: %f" % (loss, topK, prec, topK, recall, ndcg))

    return loss, prec, recall, ndcg


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

def MFTrainer(metapath, loadpath, savepath, epochs=5000, n_factor=3, 
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

    prev_loss = 0
    # set loss function
    criterion = MFLoss(reg_user, reg_item)
    optimizer = torch.optim.Adam([mf.user_factors, mf.item_factors], lr=lr)  # use weight_decay
    mf.train()
    prev_loss = 0
    print("n_user: %d, n_item: %d" % (n_user, n_item))
    for i in range(epochs + 1):
        optimizer.zero_grad()
        adj_t = mf()
        loss = criterion(mf.user_factors, mf.item_factors, adj_t, adj_mat)   # this line is ugly
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("metapath: %s, epoch %d: loss = %.4f, lr = %.10f, reg_user = %f, reg_item = %f" 
                % (metapath, i, loss, lr, reg_user, reg_item))
            if abs(int(loss) - prev_loss) < 1e-4:
                break
            prev_loss = int(loss)
            
    mf.export(savepath, metapath)

def train_FM(model, embed, train_data, valid_data, epochs=500, lr=1e-4, criterion=None, cuda=False):
    r"""
    Parameters
    ----------
    model: nn.Module
        the FM model

    train_data, valid_data: torch.utils.data.Dataset
        each line is [uid, bid, rate]

    n_neg: number of negative samples for each input in train_data

    criterion: loss function class,
        Default is nn.CrossEntropyLoss
    """
    device = torch.device('cuda:0' if cuda else 'cpu')

    train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
    FM = model

    if not criterion:
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # validation and training interruption
    best_loss = 1e10
    i = 0

    # set optimizer
    optimizer = torch.optim.Adam([FM.V, FM.W, FM.w0], lr=lr)  # maybe use weight_decay
    FM.train()
    for epoch in range(epochs):
        t0 = gettime()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            indices, target = data       # indices: [[i, j], [i, j], ...]
            indices = indices.view(-1, 2)
            target = target.view(-1)
            x = embed[indices[:, 0], indices[:, 1]]
            out = FM(x)
            # Use a Sigmoid to classify
            y_t = torch.sigmoid(out)
            # y_t = out.clone()
            y_t = y_t.unsqueeze(1).repeat(1, 2)
            y_t[:, 1] = 1 - y_t[:, 0]
            loss = criterion(y_t, target)
            loss.backward()
            optimizer.step()
        print("epoch %d, loss = %f, lr = %f, time cost = %f" % (epoch, loss, lr, gettime() - t0))

        # Validation
        if epoch % 10 == 0:
            loss, prec, recall, ndcg = eval(embed, valid_data, FM, criterion, 20)
            if loss > best_loss:
                i += 1
                if i > 5:
                    break
            elif loss < best_loss:
                best_loss = loss
                print("saving current model...")
                # FM.export()

    
if __name__ == "__main__":
    filtered_path = '../yelp_dataset/filtered/'
    adj_path = '../yelp_dataset/adjs/'
    feat_path = '../yelp_dataset/mf_features/'
    rate_path = '../yelp_dataset/rates/'

    # train MF
    # metapaths = ['UB', 'UBUB', 'UUB', 'UBCaB', 'UBCiB']
    metapaths = ['UB', 'UBUB', 'UUB', 'UBCaB', 'UBCiB', 'UCaB', 'UCiB', 'UCaBCiB', 'UCiBCaB']
    t0 = gettime()
    train_MF(metapaths, 
             adj_path, 
             feat_path, 
             epoch=[5000, 5000, 5000, 20000, 5000, 10000, 10000, 10000, 10000], 
             lr=[5e-3, 5e-2, 1e-2, 1e-2, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3], 
             reg_user=[5e-1, 5e-1, 5e-1, 5e-1, 5e-1, 5e-1, 5e-1, 5e-1, 5e-1], 
             reg_item=[5e-1, 5e-1, 5e-1, 5e-1, 5e-1, 5e-1, 5e-1, 5e-1, 5e-1], cuda=True)
    t1 = gettime()
    print("time cost: %f" % (t1 - t0))

    # train FM (cross entropy loss)
    t0 = gettime()
    print("loading data...")
    train_data = read_pickle(rate_path+'train_data.pickle')
    # Do we need negative samples? Yes, we do!
    valid_data = read_pickle(rate_path+'valid_with_neg_sample.pickle')
    test_data = read_pickle(rate_path+'test_with_neg_sample.pickle')
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
    business_ids = set(i for i in range(n_items))
    print("n_users:", n_users, "n_items:", n_items)
    train_dataset = FMG_YelpDataset(train_data, n_users, n_items, neg_sample_n=5, mode='train', cuda=True)
    valid_dataset = FMG_YelpDataset(valid_data, n_users, n_items, neg_sample_n=20, mode='valid', cuda=True)
    test_dataset = FMG_YelpDataset(test_data, n_users, n_items, neg_sample_n=20, mode='test', cuda=True)
    print("time cost: %f" % (gettime() - t0))

    t0 = gettime()
    print("start training FM...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FactorizationMachine(embed.shape[2], 20, cuda=True).to(device)

    train_FM(model, embed, train_dataset, valid_dataset, epochs=100, lr=5e-3, cuda=True)

    print("time cost: %f" % (gettime() - t0))

    # result: loss gets lower as n_neg gets higher
    # Testing
    eval(embed, test_dataset, model, nn.CrossEntropyLoss(), 20)