import argparse
import time
import math
import json
import pickle
import numpy as np
from numpy import array
# from scipy import sparse
import torch
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
from models import multi_HAN
from torch.optim import lr_scheduler
import torch.nn.functional as F
import os
import operator
from random import sample, random, randint
from functools import reduce
import cProfile
# import time
from utils import YelpDataset

def parse_args():
    parser = argparse.ArgumentParser(description='multi-embedding-HAN')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='dimension of embeddings')
    parser.add_argument('--n_facet', type=int, default=10,
                        help='number of facet for each embedding')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--decay', type=float, default=0.8,
                        help='learning rate decay rate')
    parser.add_argument('--decay_step', type=int, default=1e2,
                        help='learning rate decay step')
    parser.add_argument('--log_step', type=int, default=1e2,
                        help='log print step')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=30,
                        help='upper epoch limit')
    parser.add_argument('--patience', type=int, default=5,
                        help='Extra iterations before early-stopping')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use GPU for training')
    parser.add_argument('--save', type=str, default='model/',
                        help='path to save the final model')
    parser.add_argument('--resume', type=str, default='',
                        help='path of model to resume')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--dataset', default='yelp',
                        help='dataset name')
    parser.add_argument('--iter', type = int, default = 5)
    args = parser.parse_args()
    args.save = args.save + args.dataset
    args.save = args.save + '_batch{}'.format(args.batch_size)
    args.save = args.save + '_lr{}'.format(args.lr)
    args.save = args.save + '_emb{}'.format(args.emb_dim)
    args.save = args.save + '_facet{}'.format(args.n_facet)
    args.save = args.save + '_decay{}'.format(args.decay)
    args.save = args.save + '_decaystep{}'.format(args.decay_step)
    args.save = args.save + '_patience{}.pt'.format(args.patience)
    return args

def train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch, use_cuda):
    epoch_loss = []
    for step, batch_data in enumerate(train_data_loader):
        user, business, label, user_neigh_list_lists, business_neigh_list_lists = batch_data
        if use_cuda:
            user = user.cuda()
            business = business.cuda()
            # user_neigh_list_lists = 
        output = model(user, business, user_neigh_list_lists, business_neigh_list_lists)
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        if (step % args.log_step == 0) and step > 0:
            print('Train epoch: {}[{}/{} ({:.0f}%)]\tLoss: {:.6f}, AvgL: {:.6f}'.format(epoch, step, len(train_data_loader),
                                            100.*step/len(train_data_loader), loss.item(), np.mean(epoch_loss)))
    mean_epoch_loss = np.mean(epoch_loss)
    return mean_epoch_loss

def valid(model, valid_data_loader, loss_fn):
    valid_loss = []
    for step, batch_data in enumerate(valid_data_loader):
        user, business, label, user_neigh_list_lists, business_neigh_list_lists = batch_data
        output = model(user, business, user_neigh_list_lists, business_neigh_list_lists)
        loss = loss_fn(output, label)
        valid_loss.append(loss.item())
    mean_valid_loss = np.mean(valid_loss)
    print('Valid:\tLoss:%f' % (mean_valid_loss))
    return mean_valid_loss

if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'yelp':
        adj_paths = []
        adj_names = ['adj_UU', 'adj_UB', 'adj_BCa', 'adj_BCi', 'adj_UUB', 'adj_UBU', 'adj_UBUB', 'adj_UBCa', 'adj_UBCi', 'adj_BCaB', 'adj_BCiB']
        for name in adj_names:
            adj_paths.append('yelp_dataset/adjs/' + name)
        train_data_path = 'yelp_dataset/rates/rate_train'
        valid_data_path = 'yelp_dataset/rates/rate_valid'
        train_data_loader = DataLoader(dataset = YelpDataset(train_data_path, adj_paths, 50),
                                batch_size = args.batch_size,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = True)
        valid_data_loader = DataLoader(dataset=YelpDataset(valid_data_path, adj_paths, 50),
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True)
        num_to_id_paths = []
        num_to_ids = []
        num_to_id_names = ['num_to_userid', 'num_to_businessid', 'num_to_cityid', 'num_to_categoryid']
        for name in num_to_id_names:
            num_to_id_paths.append('yelp_dataset/adjs/' + name)
        for path in num_to_id_paths:
            with open(path, 'rb') as f:
                num_to_ids.append(pickle.load(f))
        n_users, n_businesses, n_cities, n_categories = [len(num_to_id) for num_to_id in num_to_ids]
        n_nodes_list = [n_users, n_businesses, n_cities, n_categories]
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = multi_HAN(n_nodes_list, args).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay)
    loss_fn = nn.MSELoss(reduce=True, size_average=True).to(device)
    best_loss = 100.0
    best_epoch = -1
    for epoch in range(args.epochs):
        scheduler.step()
        print('Start epoch: ', epoch)
        mean_loss = train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch, use_cuda)
            # total_loss += loss.data[0]
        valid_loss = valid(model, train_data_loader, optimizer, loss_fn)
        if valid_loss < best_loss:
            best_epoch = epoch
            best_loss = valid_loss
            with open(args.save, 'wb') as f:
                torch.save(model, f)
        if epoch-best_epoch >= args.patience:
            print('Stop training after %i epochs without improvement on validation.' % args.patience)
            break




    # Yelp500Dataset = YelpDataset("../yelp/review-500k.json", reviewid_to_num, userid_to_num, businessid_to_num)
    # Yelp500DataLoader = DataLoader(dataset = Yelp500Dataset,
    #                           batch_size = args.batch_size,
    #                           shuffle = True,
    #                           num_workers = 4,
    #                           pin_memory = True,
    #                           drop_last = True)

    # t_names = ('adj_UwR', 'adj_RaB', 'adj_UtB', 'adj_BcB', 'adj_BcateB', 'adj_UfU', 'UrateB', 'UfUwR', 'UfUrB', 'UrBcateB', 'UrBcityB', 'UrateBrateU', 'RaBaR', 'RwUwR')
    # Metapath_list = []
    # for i in range(len(t_names)):
    # adj = pickle.load("../yelp/adjs/"+t_names[i])
    # Metapath_list.append(Metapath(t_names[i], t_types[i], adj))
    print("read data end")
    '''
    n_user = dataset.user_adjs.shape[0]
    n_business = dataset.business_adjs.shape[0]
    n_cities = dataset.adjs[3].shape[1]
    n_categories = dataset.adjs[2].shape[1]
    n_node_list = [n_user, n_business, n_cities, n_categories]
    model = multi_HAN(n_node_list, args)
    for data in data_loader:
        user, business, label, user_neighbor_list， business_neighbor_list = data
        # ans = model(user, business, user_neighbor_list, business_neighbor_list)
    '''
