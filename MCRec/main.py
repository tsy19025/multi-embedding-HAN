from model import MCRec
# from modeltmp import MCRec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
from metrics import *

import argparse
import pickle
from utils import YelpDataset
import time

def parse_args():
    parse = argparse.ArgumentParser(description="Run MCRec.")
    parse.add_argument('--dataset', default = 'yelp', help = 'Choose a dataset.')
    parse.add_argument('--epochs', type = int, default = 100)
    parse.add_argument('--data_path', type = str, default = '/home1/wyf/Projects/gnn4rec/multi-embedding-HAN/yelp_dataset/')
    # parse.add_argument('--data_path', type = str, default = '../tmpdataset/')
    parse.add_argument('--negatives', type = int, default = 4)
    parse.add_argument('--batch_size', type = int, default = 64)
    parse.add_argument('--dim', type = int, default = 100)
    parse.add_argument('--sample', type = int, default = 50)
    parse.add_argument('--cuda', type = bool, default = True)
    parse.add_argument('--lr', type = float, default = 0.005)
    parse.add_argument('--decay_step', type = int, default = 1)
    parse.add_argument('--log_step', type=int, default=1e2)
    parse.add_argument('--decay', type = float, default = 0.98, help = 'learning rate decay rate')
    parse.add_argument('--save', type = str, default = 'model/bigdata_modelpara.pth')
    parse.add_argument('--K', type = int, default = 20)
    parse.add_argument('--mode', type = str, default = 'train')
    parse.add_argument('--load', type = bool, default = True)
    parse.add_argument('--patience', type = int, default = 10)
    return parse.parse_args()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch):
    print("epoch ", epoch)
    model.train()
    train_loss = []
    for step, data in enumerate(train_data_loader):
        user_input, item_input, label, paths = data
        user_input = user_input.to(device)
        item_input = item_input.to(device)
        label = label.to(device)
        path_input = []
        for path in paths:
            path_input.append(path.to(device))
        optimizer.zero_grad()
        output = model(user_input, item_input, path_input).squeeze(-1)
        # output = model(user_input, item_input).squeeze(-1)
        loss = loss_fn(output, label)
        loss = torch.sum(loss, -1)
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if (step % args.log_step == 0) and step > 0:
            print('Train epoch: {}[{}/{} ({:.0f}%)]\tLr:{:.6f}, Loss: {:.6f}, AvgL: {:.6f}'.format(epoch, step, len(train_data_loader),
                                                    100. * step / len(train_data_loader), get_lr(optimizer), loss.item(), np.mean(train_loss)))
    # l1_regularization, l2_regularization = torch.tensor([0],dtype =torch.float32), torch.tensor([0],dtype=torch.float32)
    # for param in model.parameters():
    #     l1_regularization += torch.norm(param, 1)
    #     l2_regularization += torch.norm(param, 2)
    # print("l1:", l1_regularization)
    # print("l2:", l2_regularization)
    return np.mean(train_loss)

def eval(model, eval_data_loader, device, K, loss_fn):
    model.eval()
    eval_p = []
    eval_r = []
    eval_ndcg = []
    eval_loss = []
    with torch.no_grad():
        for step, batch_data in enumerate(eval_data_loader):
            user_input, item_input, label, paths = batch_data
            user_input = user_input.to(device)
            item_input = item_input.to(device)
            label = label.to(device)
            # paths: paths * batch_size * negatives + 1 * path_num * timestamps * length
            path_input = []
            for path in paths:
                path_input.append(path.to(device))
            output = model(user_input, item_input, path_input).squeeze()
            # output = model(user_input, item_input).squeeze()
            label = label.squeeze()
            loss = loss_fn(output, label)
            eval_loss.append(torch.mean(loss).item())
            pred_items, indexes = torch.topk(output, K)
            indexes = indexes.tolist()
            gt_items = torch.nonzero(label)[:,0].tolist()
            p_at_k = getP(indexes, gt_items)
            r_at_k = getR(indexes, gt_items)
            ndcg_at_k = getNDCG(indexes, gt_items)
            eval_p.append(p_at_k)
            eval_r.append(r_at_k)
            eval_ndcg.append(ndcg_at_k)
    mean_p = np.mean(eval_p)
    mean_r = np.mean(eval_r)
    mean_ndcg = np.mean(eval_ndcg)
    mean_loss = np.mean(eval_loss)
    return mean_p, mean_r, mean_ndcg, mean_loss

def valid(model, valid_data_loader, loss_fn):
    mean_p, mean_r, mean_ndcg, mean_loss = eval(model, valid_data_loader, device, args.K, loss_fn)
    print('Valid:\tloss:', mean_loss)
    print('Valid:\tprecision@', args.K, ':', mean_p, ', recall@', args.K, ':', mean_r, ', ndcg@', args.K, ':', mean_ndcg)
    return mean_p, mean_r, mean_ndcg

def test(model, test_data_loader, loss_fn):
    mean_p, mean_r, mean_ndcg, mean_loss = eval(model, test_data_loader, device, args.K, loss_fn)
    print('Test:\tloss:', mean_loss)
    print('Test:\tprecision@', args.K, ':', mean_p, ', recall@', args.K, ':', mean_r, ', ndcg@', args.K, ':', mean_ndcg)
    return mean_p, mean_r, mean_ndcg

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("Begin")
    args = parse_args()
    if args.dataset == 'yelp':
        # path_name = ['ub_path', 'uub_path', 'ubub_path', 'ubcab_path', 'ubcib_path']
        adjs_path = args.data_path + 'adjs/'
        with open(adjs_path + 'adj_BCa', 'rb') as f: adj_BCa = pickle.load(f)
        with open(adjs_path + 'adj_BCi', 'rb') as f: adj_BCi = pickle.load(f)
        with open(adjs_path + 'adj_UB', 'rb') as f: adj_UB = pickle.load(f)
        with open(adjs_path + 'adj_UU', 'rb') as f: adj_UU = pickle.load(f)
        
        path_num = [args.sample] * 4
        timestamps = [3, 4, 4, 4]
        path_type = [[0, 0, 1], [0, 1, 0, 1], [0, 1, 2, 1], [0, 1, 3, 1]]

        num_to_ids = []
        num_to_id_paths = []
        num_to_id_names = ['num_to_userid', 'num_to_businessid', 'num_to_cityid', 'num_to_categoryid']
        for name in num_to_id_names:     
            num_to_id_paths.append(adjs_path + name)
        for path in num_to_id_paths:
            with open(path, 'rb') as f:
                num_to_ids.append(pickle.load(f))
        n_type = [len(num_to_id) + 1 for num_to_id in num_to_ids]
        
        # if args.mode == 'train':
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
        valid_data_loader = DataLoader(dataset = YelpDataset(valid_data, path_num, timestamps, adj_BCa, adj_BCi, adj_UB, adj_UU, 0, args.sample, 'valid'),
                                       batch_size = 1,
                                       shuffle = True,
                                       num_workers = 20,
                                       pin_memory = True)

        # else:
        test_data_path = args.data_path + 'rates/test_with_neg'
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        test_data_loader = DataLoader(dataset = YelpDataset(test_data, path_num, timestamps, adj_BCa, adj_BCi, adj_UB, adj_UU, 0, args.sample, 'test'),
                                      batch_size = 1,
                                      shuffle = True,
                                      num_workers = 20,
                                      pin_memory = True)

    print("n_type:", n_type)
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = MCRec(n_type, path_num, timestamps, args.dim, path_type, args.dataset, device)
    model = model.to(device)
    # for m in model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.xavier_uniform_(m.weight)
    # print("MCRec have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    loss_fn = nn.BCELoss(reduction = 'none').to(device)
    # valid_loss_fn = nn.BCELoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=0.000001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = args.decay_step, gamma = args.decay)
    best_recall = 0
    best_epoch = -1
    if args.load == True:
        state = torch.load(args.save)
        model.load_state_dict(state['net'])
        best_recall = state['recall']
        best_epoch = state['epoch']
        model.to(device)
    if args.mode == 'train':
        # __, _, _ = valid(model, valid_data_loader, valid_loss_fn)
        for epoch in range(best_epoch + 1, args.epochs):
            mean_loss = train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch)
            # print("epoch:", epoch, "    loss:", mean_loss)
            valid_precision, valid_recall, valid_ndcg = valid(model, valid_data_loader, loss_fn)
            scheduler.step()
            if valid_recall > best_recall:
                best_recall = valid_recall
                best_epoch = epoch
                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'recall': best_recall, 'epoch': epoch}
                torch.save(state, args.save)
                print('Model save for better valid recall: ', best_recall)
            if epoch - best_epoch >= args.patience:
                print("stop training at epoch ", epoch)
                break
    # test
    state = torch.load(args.save)
    model.load_state_dict(state['net'])
    model.to(device)
    _ = test(model, test_data_loader, loss_fn)
    # print("test: ", test)

