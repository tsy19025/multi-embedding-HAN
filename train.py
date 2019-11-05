import argparse
import time
import math
import json
import pickle
import numpy as np
from numpy import array
from scipy import sparse
import torch
from torch import Dataset, DataLoader
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import os
import operator
from random import sample, random, randint
from functools import reduce
import cProfile

from utils import YelpDataset, Metapath
from yelp_dataset.yelp500_gen import load_jsondata_from_file, get_id_to_num

parser = argparse.ArgumentParser(description='multi-embedding-HAN')
parser.add_argument('--embsize', type=int, default=64,
                    help='size of embeddings')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--save', type=str, default='model'+ int(time.time()) + '.pt',
                    help='path to save the final model')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer to use (sgd, adam)')

args = parser.parse_args()

user_json = load_jsondata_from_file("../yelp/user-500k.json")
business_json = load_jsondata_from_file("../yelp/business-500k.json")
review_json = load_jsondata_from_file("../yelp/review-500k.json")
reviewnum_to_id, reviewid_to_num = get_id_to_num(review_json, "review_id")
businessnum_to_id, businessid_to_num = get_id_to_num(business_json, "business_id")
usernum_to_id, userid_to_num = get_id_to_num(user_json, "user_id")

Yelp500Dataset = YelpDataset("../yelp/review-500k.json", reviewid_to_num, userid_to_num, businessid_to_num)
Yelp500DataLoader = DataLoader(dataset = Yelp500Dataset,
                              batch_size = args.batch_size,
                              shuffle = True,
                              num_workers = 4,
                              pin_memory = True,
                              drop_last = True)

t_names = ('adj_UwR', 'adj_RaB', 'adj_UtB', 'adj_BcB', 'adj_BcateB', 'adj_UfU', 'UrateB', 'UfUwR', 'UfUrB', 'UrBcateB', 'UrBcityB', 'UrateBrateU', 'RaBaR', 'RwUwR')
Metapath_list = []
for i in range(len(t_names)):
    adj = pickle.load("../yelp/adjs/"+t_names[i])
    Metapath_list.append(Metapath(t_names[i], t_types[i], adj))