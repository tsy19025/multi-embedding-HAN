from model import MCRec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
from metrics import *

import argparse
import utils
import pickle
import utils
from utils import YelpDataset
import os
import sys
from sklearn.externals import joblib
import time

def parse_args():
    parse = argparse.ArgumentParser(description="Run MCRec.")
    parse.add_argument('--dataset', default = 'yelp', help = 'Choose a dataset.')
    parse.add_argument('--epochs', type = int, default = 100)
    parse.add_argument('--data_path', type = str, default = '/home1/wyf/Projects/gnn4rec/multi-embedding-HAN/yelp_dataset/')
    parse.add_argument('--negatives', type = int, default = 2)
    parse.add_argument('--batch_size', type = int, default = 60)
    parse.add_argument('--dim', type = int, default = 64)
    parse.add_argument('--sample', type = int, default = 20)
    parse.add_argument('--cuda', type = bool, default = True)
    parse.add_argument('--lr', type = float, default = 0.0001)
    parse.add_argument('--decay_step', type = int, default = 1)
    parse.add_argument('--decay', type = float, default = 0.98, help = 'learning rate decay rate')
    parse.add_argument('--feature_dim', type = int, default = 64)
    parse.add_argument('--save', type = str, default = 'model/')
    parse.add_argument('--K', type = int, default = 20)
    parse.add_argument('--mode', type = str, default = 'train')
    # parse.add_argument()

    return parse.parse_args()

def train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch):
    print("train ", epoch)
    model.train()
    train_loss = []

    begin_ticks = time.time()
    for step, data in enumerate(train_data_loader):
        user_input_tmp, item_input_tmp, label_tmp, paths = data
        user_input = []
        item_input = []
        label = []
        for i in range(args.batch_size):
            for j in range(args.negatives + 1):
                user_input.append(user_input_tmp[j][i])
                item_input.append(item_input_tmp[j][i])
                label.append(label_tmp[j][i])
        user_input = torch.tensor(user_input).to(device)
        item_input = torch.tensor(item_input).to(device)
        label = torch.DoubleTensor(label).to(device)
        # paths: paths * batch_size * negatives + 1 * path_num * timestamps
        path_input = []
        for path in paths:
            batch_size, items_size, path_num, timestamps = path.shape
            path_input.append(path.view(batch_size * items_size, path_num, timestamps).to(device))
        # path_input: paths * (batch_size * nega + 1) * path_num * timestamps

        output = model(user_input, item_input, path_input).squeeze(-1).double()
        # print(output)
        # print(label)
        loss = loss_fn(output, label)
        # print(loss)
        # print("----------------------------------------------------------------")
        loss = torch.mean(loss)
        #print(loss)
        #print("----------------------------------------------------------------")
        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    end_ticks = time.time()
    print("cost time: ", end_ticks - begin_ticks)
    return np.mean(train_loss)

def eval(model, eval_data_loader, device, K):
    print('eval')
    model.eval()

    eval_p = []
    eval_r = []
    eval_ndcg = []

    with torch.no_grad():
        for step, batch_data in enumerate(eval_data_loader):
            user_input, item_input, label, paths, pos, neg = batch_data
            user_input = torch.cat(user_input, 0).to(device)
            item_input = torch.cat(item_input, 0).to(device)
            label = torch.cat(label, 0).double().to(device)

            # paths: paths * batch_size * negatives + 1 * path_num * timestamps * length
            path_input = []
            for path in paths:
                batch_size, items_size, path_num, timestamps, length = path.shape
                path_input.append(path.view(batch_size * items_size, path_num, timestamps, length).to(device))
            # print(path_input)
            # path_input: paths * (batch_size * nega + 1) * path_num * timestamps * length
            output = model(user_input, item_input, path_input).squeeze(-1)
            # print("output")
            pred_items, indexs = torch.topk(output, K)
            gt_items = torch.nonzero(label)[:, 0].tolist()
            indexs = indexs.tolist()
            # indexs = indexs.tolist()
            # for index in indexs:
            #    if index < pos: gt_items.append(index)
            
            p_at_k = getP(indexs, gt_items)
            r_at_k = getR(indexs, gt_items)
            ndcg_at_k = getNDCG(indexs, gt_items)
            # print(indexs)
            # print(gt_items)
            # print(p_at_k, r_at_k, ndcg_at_k)
            # print("--------------------------------------------------")

            eval_p.append(p_at_k)
            eval_r.append(r_at_k)
            eval_ndcg.append(ndcg_at_k)
    
    mean_p = np.mean(eval_p)
    mean_r = np.mean(eval_r)
    mean_ndcg = np.mean(eval_ndcg)
    return mean_p, mean_r, mean_ndcg

def valid(model, valid_data_loader, loss_fn):
    mean_p, mean_r, mean_ndcg = eval(model, valid_data_loader, device, args.K)
    print('Valid:\tprecision@', args.K, ':', mean_p, ', recall@', args.K, ':', mean_r, ', ndcg@', args.K, ':', mean_ndcg)
    return mean_p, mean_r, mean_ndcg

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("Begin")
    args = parse_args()
    if args.dataset == 'yelp':
        path_name = ['ub_path', 'uub_path', 'ubub_path', 'ubcab_path', 'ubcib_path']
        adjs_path = args.data_path + 'adjs/'
        with open(adjs_path + 'adj_BCa', 'rb') as f: adj_BCa = pickle.load(f)
        with open(adjs_path + 'adj_BCi', 'rb') as f: adj_BCi = pickle.load(f)
        with open(adjs_path + 'adj_UB', 'rb') as f: adj_UB = pickle.load(f)
        with open(adjs_path + 'adj_UU', 'rb') as f: adj_UU = pickle.load(f)
        
        path_num = [1] + [args.sample] * 4
        timestamps = [2, 3, 4, 4, 4]
        path_type = [[0, 1], [0, 0, 1], [0, 1, 0, 1], [0, 1, 3, 1], [0, 1, 2, 1]]

        num_to_ids = []
        num_to_id_paths = []
        num_to_id_names = ['num_to_userid', 'num_to_businessid', 'num_to_cityid', 'num_to_categoryid']
        for name in num_to_id_names:     
            num_to_id_paths.append(adjs_path + name)
        for path in num_to_id_paths:
            with open(path, 'rb') as f:
                num_to_ids.append(pickle.load(f))
        n_type = [len(num_to_id) for num_to_id in num_to_ids]
        
        if args.mode == 'train':
            train_data_path = args.data_path + 'rates/rate_train'
            with open(train_data_path, 'rb') as f:
                train_data = pickle.load(f)
            train_data_loader = DataLoader(dataset = YelpDataset(train_data, path_num, timestamps, adj_BCa, adj_BCi, adj_UB, adj_UU, args.negatives, args.sample, 'train'),
                                           batch_size = args.batch_size,
                                           shuffle = True,
                                           num_workers = 20,
                                           pin_memory = True)

            valid_data_path = args.data_path + 'rates/valid_with_neg'
            with open(valid_data_path, 'rb') as f:
                valid_data = pickle.load(f)
            valid_data_loader = DataLoader(dataset = YelpDataset(valid_data, path_num, timestamps,adj_BCa, adj_BCi, adj_UB, adj_UU, 0, args.sample, 'valid'),
                                           batch_size = 1,
                                           shuffle = True,
                                           num_workers = 20,
                                           pin_memory = True)

        print("read test data")
        test_data_path = args.data_path + 'rates/test_with_neg'
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
            test_data_loader = DataLoader(dataset = YelpDataset(test_data, path_num, timestamps, adj_BCa, adj_BCi, adj_UB, adj_UU, 0, args.sample, 'test'),
                                          batch_size = 1, 
                                          shuffle = True,
                                          num_workers = 20,
                                          pin_memory = True)

    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = MCRec(n_type, path_num, timestamps, args.feature_dim, args.dim, path_type, device)
    model = model.to(device)
    print("MCRec have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    loss_fn = nn.BCELoss(reduction='none').to(device)
    valid_loss_fn = nn.BCELoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = args.decay_step, gamma = args.decay)
    if args.mode == 'train':
        best_loss = 100000
        before_loss = 0
        for epoch in range(args.epochs):
            mean_loss = train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch)
            _, _, valid_loss = valid(model, valid_data_loader, valid_loss_fn)
            print("epoch: ", epoch, "   loss: ", mean_loss)
            if valid_loss < best_loss:
                best_loss = valid_loss
                # with open(args.save + '', 'wb') as f:
                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, '/home1/tsy/multi-embedding-HAN/tmp/model/modelpara1.pth')
                print('Model save for lower valid loss %f' % best_loss)
            if abs(mean_loss - before_loss) < 0.0001:
                print("stop training at epoch: ", epoch)
                break
            sys.exit()
            before_loss = mean_loss
    # test
    state = torch.load('/home1/tsy/multi-embedding-HAN/tmp/model/modelpara1.pth')
    model.load_state_dict(state['net'])
    model.to(device)

    test_loss = valid(model, test_data_loader, valid_loss_fn)
    print("test: ", test_loss)

