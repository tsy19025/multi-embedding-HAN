import os
import time
import argparse

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

def parse_args():
    parse = argparse.ArgumentParser(description="Run MCRec.")
    parse.add_argument('--dataset', default='yelp', help='Choose a dataset.')
    parse.add_argument('--epochs', type=int, default=100000)
    parse.add_argument('--data_path', type=str, default='/home1/wyf/Projects/gnn4rec/multi-embedding-HAN/yelp_dataset/')
    parse.add_argument('--negatives', type=int, default=4)
    # parse.add_argument('--batch_size', type=int, default=64)
    # parse.add_argument('--dim', type=int, default=100)
    # parse.add_argument('--sample', type=int, default=64)
    parse.add_argument('--cuda', type=bool, default=True)
    # parse.add_argument('--lr', type=float, default=0.0001)
    # parse.add_argument('--decay_step', type=int, default=5)
    # parse.add_argument('--log_step', type=int, default=1e2)
    # parse.add_argument('--decay', type=float, default=0.95, help='learning rate decay rate')
    # parse.add_argument('--save', type=str, default='model/bigdata_modelpara1_dropout0.5.pth')
    parse.add_argument('--fm-factor', type=int, default=20)
    parse.add_argument('--mode', type=str, default='train')
    # parse.add_argument('--load', type=bool, default=False)
    # parse.add_argument('--patience', type=int, default=10)
    parse.add_argument('--cluster', action='store_true', default=False, help="Run the program on cluster or PC")
    parse.add_argument('--toy', action='store_true', default=False, help="Toy dataset for debugging")
    parse.add_argument('--mf-train', action='store_true', default=False, help="Run Matrix Factorization training")
    parse.add_argument('--mf-factor', type=int, default=20, help="n_factor for MF")

    return parse.parse_args()

def train_MF(metapaths, loadpath, savepath, n_factor=3, reg_user=5e-2, reg_item=5e-2, lr=1e-2, epoch=5000, cuda=False):
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
        MFTrainer(metapath, loadpath, savepath, n_factor, epoch[i], lr=lr[i], reg_user=reg_user[i], reg_item=reg_item[i], cuda=cuda)
        i += 1

def MFTrainer(metapath, loadpath, savepath, n_factor, epochs=5000, 
            lr=1e-4, reg_user=5e-2, reg_item=5e-2, decay_step=30, decay=0.1, 
            cuda=False):
    device = torch.device('cuda:0' if cuda else 'cpu')

    def _load_data(filepath, metapath, device):
        data = []

        if args.cluster:
            file = filepath + 'adj_' + metapath
        else:
            file = filepath + 'adj_' + metapath + '.pickle'

        with open(file, 'rb') as fw:
            adjacency = pickle.load(fw)
            data = torch.tensor(adjacency, dtype=torch.float32, requires_grad=False).to(device)
        n_user, n_item = data.shape
        return n_user, n_item, data

    n_user, n_item, adj_mat = _load_data(loadpath, metapath, device)
    mf = MatrixFactorizer(n_user, n_item, n_factor, cuda).to(device)

    print("--------------------- n_user: {}, n_item: {}---------------------".format(n_user, n_item))

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

def train_FM(model, train_data, valid_data, epochs=500, lr=1e-4, criterion=None, cuda=False):
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
            x, target = data       # indices: [[i, j], [i, j], ...]
            x = x.view(-1, x.shape[2])
            # indices = indices.view(-1, 2)
            target = target.view(-1)
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
        if epoch % 2 == 0:
            loss, prec, recall, ndcg = eval(valid_data, FM, criterion, 20, cuda=cuda)
            if loss > best_loss:
                i += 1
                if i > 2:
                    break
            elif loss < best_loss:
                best_loss = loss
                print("saving current model...")
                # FM.export()

def eval(valid_data, model, criterion, topK, cuda=False):
    r"""
    Calculate precision, recall and ndcg of the prediction of the model.
    Returns
    -------
    precision, recall, ndcg
    """
    device = torch.device('cuda:0' if cuda else 'cpu')

    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True)
    model.eval()

    loss_list = []
    prec_list = []
    recall_list = []
    ndcg_list = []
    for i, data in enumerate(valid_loader):
        x, items, labels = data
        x = x.view(-1, x.shape[2])
        items = items.view(-1)
        labels = labels.view(-1)
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

    
if __name__ == "__main__":
    args = parse_args()

    if args.cluster:
        read_path = '/home1/wyf/Projects/gnn4rec/multi-embedding-HAN/yelp_dataset/'
    else:
        read_path = '../yelp_dataset/'
    write_path = '../yelp_dataset/'
    filtered_path = read_path + 'filtered/'
    adj_path = read_path + 'adjs/'
    feat_path = write_path + 'mf_features/'
    rate_path = read_path + 'rates/'

    # train MF
    metapaths = ['UB', 'UBUB', 'UUB', 'UBCaB', 'UBCiB']
    # metapaths = ['UBUB', 'UUB', 'UBCaB', 'UBCiB']
    # metapaths = ['UB', 'UBUB', 'UUB', 'UBCaB', 'UBCiB', 'UCaB', 'UCiB', 'UCaBCiB', 'UCiBCaB']
    t0 = gettime()
    if args.mf_train:
        train_MF(metapaths, 
                adj_path, 
                feat_path, 
                n_factor=args.mf_factor,
                epoch=[10000, 50000, 30000, 40000, 20000], #, 20000, 10000, 50000, 50000], 
                lr=[5e-3, 3e-3, 3e-3, 3e-3, 5e-3], #, 5e-3, 5e-3, 7e-3, 7e-3], 
                reg_user=[5e-1, 1e-1, 1e-1, 5e-1, 5e-1], #, 5e-1, 5e-1, 5e-1, 5e-1], 
                reg_item=[5e-1, 1e-1, 1e-1, 5e-1, 5e-1], #, 5e-1, 5e-1, 5e-1, 5e-1], 
                cuda=args.cuda)
    t1 = gettime()
    print("time cost: %f" % (t1 - t0))

    # train FM (cross entropy loss)
    t0 = gettime()
    print("loading data...")
    # Do we need negative samples? Yes!
    if args.cluster:
        train_data = read_pickle(rate_path+'rate_train')
        valid_data = read_pickle(rate_path+'valid_with_neg')
        test_data = read_pickle(rate_path+'test_with_neg')
    else:
        train_data = read_pickle(rate_path+'train_data.pickle')
        valid_data = read_pickle(rate_path+'valid_with_neg_sample.pickle')
        test_data = read_pickle(rate_path+'test_with_neg_sample.pickle')
    print("time cost: %f" % (gettime() - t0))

    t0 = gettime()
    print("loading features")

    if args.cluster:
        # temporary solution
        adj_UB = read_pickle(adj_path+'adj_UB')
        n_users = adj_UB.shape[0]
        n_items = adj_UB.shape[1]
        del adj_UB
    else:
        if args.toy:
            # small dataset
            users = read_pickle(filtered_path+'users-small.pickle')
            businesses = read_pickle(filtered_path+'businesses-small.pickle')
        else:
            # full dataset
            users = read_pickle(filtered_path+'users-complete.pickle')
            businesses = read_pickle(filtered_path+'businesses-complete.pickle')
        n_users = len(users)
        n_items = len(businesses)

    print("time cost: %f" % (gettime() - t0))

    t0 = gettime()
    print("making datasets...")
    business_ids = set(i for i in range(n_items))
    print("n_users:", n_users, "n_items:", n_items)
    user_features, item_features = load_feature(feat_path, metapaths)
    train_dataset = FMG_YelpDataset(train_data, user_features, item_features, neg_sample_n=args.negatives, mode='train', cuda=args.cuda)
    valid_dataset = FMG_YelpDataset(valid_data, user_features, item_features, neg_sample_n=20, mode='valid', cuda=args.cuda)
    test_dataset = FMG_YelpDataset(test_data, user_features, item_features, neg_sample_n=20, mode='test', cuda=args.cuda)
    print("time cost: %f" % (gettime() - t0))

    t0 = gettime()
    print("start training FM...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FactorizationMachine(2*len(metapaths)*args.mf_factor, args.fm_factor, cuda=args.cuda).to(device)

    train_FM(model, train_dataset, valid_dataset, epochs=6, lr=5e-3, cuda=args.cuda)

    print("time cost: %f" % (gettime() - t0))

    # result: loss gets lower as n_neg gets higher
    # Testing
    print("------------------test---------------")
    eval(test_dataset, model, nn.CrossEntropyLoss(), 20, cuda=args.cuda)
    print(model.W)
    print(model.V)