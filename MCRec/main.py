from model import MCRec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import argparse
import utils
import pickle
import utils as utils
from utils import YelpDataset
import os
import sys
from sklearn.externals import joblib

def parse_args():
    parse = argparse.ArgumentParser(description="Run MCRec.")
    parse.add_argument('--dataset', default = 'yelp', help = 'Choose a dataset.')
    parse.add_argument('--epochs', type = int, default = 30)
    parse.add_argument('--adjs_path', type = str, default = '../yelp_dataset/adjs')
    parse.add_argument('--negetives', type = int, default = 5)
    parse.add_argument('--batch_size', type = int, default = 60)
    parse.add_argument('--dim', type = int, default = 64)
    parse.add_argument('--sample', type = int, default = 5)
    parse.add_argument('--cuda', type = bool, default = True)
    parse.add_argument('--lr', type = float, default = 0.001)
    parse.add_argument('--decay_step', type = int, default = 1)
    parse.add_argument('--decay', type = float, default = 0.98, help = 'learning rate decay rate')
    parse.add_argument('--feature_dim', type = int, default = 1)
    # parse.add_argument()

    return parse.parse_args()

def train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch):
    print('train')
    model.train()
    train_loss = []

    for step, data in enumerate(train_data_loader):
        user_input, item_input, label, paths = data
        user_input = torch.cat(user_input, 0).to(device)
        item_input = torch.cat(item_input, 0).to(device)
        label = torch.cat(label, 0).double().to(device)

        # paths: paths * batch_size * negetives + 1 * path_num * timestamps * length
        path_input = []
        for path in paths:
            batch_size, items_size, path_num, timestamps, length = path.shape
            path_input.append(path.view(batch_size * items_size, path_num, timestamps, length).to(device))
        # path_input: paths * (batch_size * nege + 1) * path_num * timestamps * length

        output = model(user_input, item_input, path_input).squeeze(-1)

        loss = loss_fn(output.double(), label)
        train_loss.append(loss.data)
        # print(loss.shape)
        loss = torch.mean(torch.sum(loss, 0))
        loss.backward()

        optimizer.step()
    return np.mean(train_loss)

def eval(model, eval_data_loader, device, K):
    model.eval()

    eval_p = []
    eval_r = []
    eval_ndcg = []

    with torch.no_grad():
        for step, batch_data in enumerate(eval_data_loader):
            user_input, item_input, labels, path_input = batch_data
            logit = model(user_input, item_input, path_input)

    return 0

def valid(model, valid_data_loader, loss_fn):
    mean_p, mean_r, mean_ndcg = eval(model, valid_data_loader, device, 20)
    print('Valid:\tprecision@10:%f, recall@10:%f, ndcg@10:%f' % (mean_p, mean_r, mean_ndcg))
    return mean_p, mean_r, mean_ndcg

if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'yelp':
        path_name = ['ub_path', 'uub_path', 'ubub_path', 'ubcab_path', 'ubcib_path']
        if os.path.exists('path_data/' + path_name[0]):
            paths = []
            for name in path_name:
                with open('path_data/' + name, 'rb') as f:
                    paths.append(pickle.load(f))
        else:
            adjs_path = args.adjs_path
            with open(adjs_path + '/adj_BCa', 'rb') as f:
                adj_BCa = pickle.load(f)
            with open(adjs_path + '/adj_BCi', 'rb') as f:
                adj_BCi = pickle.load(f)
            with open(adjs_path + '/adj_UB', 'rb') as f:
                adj_UB = pickle.load(f)
            with open(adjs_path + '/adj_UU', 'rb') as f:
                adj_UU = pickle.load(f)

            paths = utils.get_path(adj_BCa, adj_BCi, adj_UB, adj_UU)
            if not os.path.exists('path_data'): os.mkdir('path_data')
            for i in range(5):
                with open('path_data/' + path_name[i], 'wb') as f:
                    pickle.dump(paths[i], f)
        path_num = [1] + [args.sample] * 4
        timestamps = [2, 3, 4, 4, 4]

        with open(args.adjs_path + '/adj_UB', 'rb') as f:
            adj_UB = pickle.load(f)
        users, items = adj_UB.shape

        '''
        num_to_ids = []
        num_to_id_names = ['num_to_userid', 'num_to_businessid', 'num_to_cityid', 'num_to_categoryid']
        for name in num_to_id_names:     
            num_to_id_paths.append('../yelp_dataset/adjs/' + name)
        for path in num_to_id_paths:
            with open(path, 'rb') as f:
                num_to_ids.append(pickel.load(f))
        n_node_list = [len(num_to_id) for num_to_id in num_to_ids]
        '''

        train_data_path = '../yelp_dataset/rates/rate_train'
        with open(train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        train_data_loader = DataLoader(dataset = YelpDataset(users, items, train_data, paths, path_num, timestamps, adj_UB, args.negetives),
                                       batch_size = args.batch_size,
                                       shuffle = True,
                                       num_workers = 20,
                                       pin_memory = True)

        valid_data_path = '../yelp_dataset/rates/rate_valid'
        with open(valid_data_path, 'rb') as f:
            valid_data = pickle.load(f)
        valid_data_loader = DataLoader(dataset = YelpDataset(users, items, valid_data, paths, path_num, timestamps, adj_UB, 0),
                                       batch_size = 1,
                                       shuffle = True,
                                       num_workers = 20,
                                       pin_memory = True)

    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = MCRec(users, items, path_num, timestamps, args.feature_dim, args.dim, device)
    model = model.to(device)

    loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(device)
    valid_loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = args.decay_step, gamma = args.decay)
    
    best_loss = 100000
    for epoch in range(args.epochs):
        mean_loss = train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch)
        valid_loss = valid(model, valid_data_load, valid_loss_fn)
        if valid_loss < best_loss:
            best_loss = valid_loss
            with open(args.save, 'wb') as f:
                torch.save(model.state_dict(), f)
