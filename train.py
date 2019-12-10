import argparse
import time
import math
import json
import pickle
import numpy as np
from numpy import array
# from scipy import sparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import os
import operator
from random import sample, random, randint
from functools import reduce
import cProfile
import time
from utils import YelpDataset

loss_fn = nn.MSELoss(reduction = 'none')

def train_one_epoch(epoch, data_loader):
    total_loss = 0
    start_time = time.time()

    for data in data_loader:
        user, business, label, user_neighbor_list, business_neighbor = data
        output = model(user, business, user_neighbor_list, business_neighbor_list)
        loss = loss_fn(output, label)
        total_loss += loss.data[0]
        loss.backward()

    #    op
    # print("epoch: {0}, loss: {1}, time:{3}".format{epoch, total_loss, time.time() - start_time})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='multi-embedding-HAN')
    parser.add_argument('--embed_dim', type=int, default=64,
                    help='dimension of embeddings')
    parser.add_argument('--facet_num', type=int, default=10,
                    help='number of facet for each embedding')
    parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
    parser.add_argument('--decay', type=float, default=0.8,
                    help='learning rate decay rate')
    parser.add_argument('--decay_step', type=int, default=1e2,
                    help='learning rate decay step')
    parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
    parser.add_argument('--cuda', action='store_true', default=True,
                    help='use GPU for training')
    parser.add_argument('--save', type=str, default='model'+ str(time.time()) + '.pt',
                    help='path to save the final model')
    parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
    parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer to use (sgd, adam)')
    parser.add_argument('--dataset', default='yelp',
                    help='dataset name')
    args = parser.parse_args()

    if args.dataset == 'yelp':
        adj_paths = []
        adj_names = ['adj_UU', 'adj_UB', 'adj_BCa', 'adj_BCi', 'adj_UUB', 'adj_UBU', 'adj_UBUB', 'adj_UBCa', 'adj_UBCi', 'adj_BCaB', 'adj_BCiB']
        for name in adj_names:
            adj_paths.append('yelp_dataset/adjs/' + name)
        data_path = 'yelp_dataset/rates/rate_train'
        data_loader = DataLoader(dataset = YelpDataset(data_path, adj_paths, 50),
                                batch_size = args.batch_size,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = True)
    # user_json = load_jsondata_from_file("../yelp/user-500k.json")
    # business_json = load_jsondata_from_file("../yelp/business-500k.json")
    # review_json = load_jsondata_from_file("../yelp/review-500k.json")
    # reviewnum_to_id, reviewid_to_num = get_id_to_num(review_json, "review_id")
    # businessnum_to_id, businessid_to_num = get_id_to_num(business_json, "business_id")
    # usernum_to_id, userid_to_num = get_id_to_num(user_json, "user_id")
    #
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