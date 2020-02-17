import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from data import *
from loss import MFLoss
from metrics import *
from model_fmg import FactorizationMachine, MatrixFactorizer
from utils_fmg import *

gettime = lambda: time.time()

def parse_args():
    parse = argparse.ArgumentParser(description="Run MCRec.")
    parse.add_argument('--dataset', default='yelp', help='Choose a dataset.')
    parse.add_argument('--epochs', type=int, default=500)
    parse.add_argument('--data_path', type=str, default='/home1/wyf/Projects/gnn4rec/multi-embedding-HAN/yelp_dataset/')
    parse.add_argument('--write_path', type=str, default='/home1/wyf/Projects/gnn4rec/multi-embedding-HAN/FMG/tmp/')
    parse.add_argument('--negatives', type=int, default=4)
    parse.add_argument('--batch_size', type=int, default=64)
    # parse.add_argument('--batch_size', type=int, default=64)
    # parse.add_argument('--dim', type=int, default=100)
    # parse.add_argument('--sample', type=int, default=64)
    parse.add_argument('--cuda', type=bool, default=True)
    # parse.add_argument('--lr', type=float, default=0.0001)
    # parse.add_argument('--decay_step', type=int, default=5)
    # parse.add_argument('--log_step', type=int, default=1e2)
    # parse.add_argument('--decay', type=float, default=0.95, help='learning rate decay rate')
    # parse.add_argument('--save', type=str, default='model/bigdata_modelpara1_dropout0.5.pth')
    parse.add_argument('--fm-factor', type=int, default=10)
    parse.add_argument('--mode', type=str, default='train')
    parse.add_argument('--mf_log_step', type=int, default=1e2)
    parse.add_argument('--fm_log_step', type=int, default=1e2)
    # parse.add_argument('--load', type=bool, default=False)
    # parse.add_argument('--patience', type=int, default=10)
    # parse.add_argument('--cluster', action='store_true', default=False, help="Run the program on cluster or PC")
    # parse.add_argument('--toy', action='store_true', default=False, help="Toy dataset for debugging")
    parse.add_argument('--mf-train', action='store_true', default=True, help="Run Matrix Factorization training")
    parse.add_argument('--mf-factor', type=int, default=10, help="n_factor for MF")
    parse.add_argument('--mf_patience', type=int, default=100)
    parse.add_argument('--fm_patience', type=int, default=10)
    parse.add_argument('--save', type=str, default='../tmp/modelpara.pth')

    return parse.parse_args()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_MF(metapaths, loadpath, ratepath, savepath, n_factor, reg_user, reg_item, lr, epoch, decay_step, decay, dataset, patience, log_step, cuda):
    if dataset == 'yelp':
        num_to_ids = []
        num_to_id_paths = []
        num_to_id_names = ['num_to_userid', 'num_to_businessid', 'num_to_cityid', 'num_to_categoryid']
        for name in num_to_id_names:
            num_to_id_paths.append(loadpath + name)
        for path in num_to_id_paths:
            num_to_ids.append(read_pickle(path))
        n_type = [len(num_to_id) for num_to_id in num_to_ids]
        valid_with_neg = read_pickle(ratepath+'valid_with_neg')
        adj_UB_valid = np.zeros([n_type[0], n_type[1]])
        for valid in valid_with_neg:
            user = valid['user_id']
            pos_items = valid['pos_business_id']
            adj_UB_valid[user, pos_items] = 1
        adj_UB = read_pickle(loadpath + 'adj_UB')
        adj_UU = read_pickle(loadpath + 'adj_UU')
        adj_BCi = read_pickle(loadpath + 'adj_BCi')
        adj_BCa = read_pickle(loadpath + 'adj_BCa')
        adj_UUB = read_pickle(loadpath + 'adj_UUB')
        adj_UBUB = read_pickle(loadpath + 'adj_UBUB')
        adj_UBCiB = read_pickle(loadpath + 'adj_UBCiB')
        adj_UBCaB = read_pickle(loadpath + 'adj_UBCaB')
        adj_UB_valid = adj_UB + adj_UB_valid
        adj_UUB_valid = adj_UU.dot(adj_UB_valid)
        adj_UBUB_valid = adj_UB_valid.dot(adj_UB_valid.T).dot(adj_UB_valid)
        adj_UBCiB_valid = adj_UB_valid.dot(adj_BCi).dot(adj_BCi.T)
        adj_UBCaB_valid = adj_UB_valid.dot(adj_BCa).dot(adj_BCa.T)
        adj_list = []
        adj_valid_list = []
        for metapath in metapaths:
            if metapath == 'UB':
                adj_list.append(adj_UB)
                adj_valid_list.append(adj_UB_valid)
            elif metapath == 'UUB':
                adj_list.append(adj_UUB)
                adj_valid_list.append(adj_UUB_valid)
            elif metapath == 'UBUB':
                adj_list.append(adj_UBUB)
                adj_valid_list.append(adj_UBUB_valid)
            elif metapath == 'UBCiB':
                adj_list.append(adj_UBCiB)
                adj_valid_list.append(adj_UBCiB_valid)
            elif metapath == 'UBCaB':
                adj_list.append(adj_UBCaB)
                adj_valid_list.append(adj_UBCaB_valid)
            else:
                print('metapath wrong!')
    i = 0
    for metapath in metapaths:
        # instance the MF trainer
        MFTrainer(metapath, adj_list[i], adj_valid_list[i], savepath, n_factor, epoch[i], lr=lr[i], reg_user=reg_user[i], reg_item=reg_item[i], decay_step=decay_step, decay=decay, patience=patience, log_step=log_step, cuda=cuda)
        i += 1

def MFTrainer(metapath, adj_mat, adj_valid_mat, savepath, n_factor, epochs, lr, reg_user, reg_item, decay_step, decay, patience, log_step, cuda):
    device = torch.device('cuda' if cuda else 'cpu')
    n_user = adj_mat.shape[0]
    n_item = adj_mat.shape[1]
    adj_mat = torch.tensor(adj_mat, dtype=torch.float32, requires_grad=False).to(device)
    adj_valid_mat = torch.tensor(adj_valid_mat, dtype=torch.float32, requires_grad=False).to(device)
    mf = MatrixFactorizer(n_user, n_item, n_factor, cuda).to(device)
    criterion = MFLoss(reg_user, reg_item)
    optimizer = torch.optim.Adam([mf.user_factors, mf.item_factors], lr=lr, weight_decay=0.000001)  # use weight_decay
    # optimizer = torch.optim.Adam(mf.parameters(), lr=lr, weight_decay=0.000001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay)
    # def _load_data(filepath, metapath, device):
    #     # data = []
    #     # if args.cluster:
    #     file = filepath + 'adj_' + metapath
    #     # else:
    #     #     file = filepath + 'adj_' + metapath + '.pickle'
    #
    #     with open(file, 'rb') as fw:
    #         adjacency = pickle.load(fw)
    #         data = torch.tensor(adjacency, dtype=torch.float32, requires_grad=False).to(device)
    #     n_user, n_item = data.shape
    #     return n_user, n_item, data

    def _MFtrain(metapath, model, adj_mat, optimizer, epoch, reg_user, reg_item):
        # print("epoch ", epoch)
        model.train()
        optimizer.zero_grad()
        adj_t = model()
        loss = criterion(model.user_factors, model.item_factors, adj_t, adj_mat)  # this line is ugly
        loss.backward()
        optimizer.step()
        # print("metapath: %s, epoch %d: loss = %.4f, lr = %.10f, reg_user = %f, reg_item = %f"
        #       % (metapath, epoch, loss, get_lr(optimizer), reg_user, reg_item))
        return loss.item()

    def _MFvalid(metapath, model, adj_valid_mat):
        # print('Valid')
        model.eval()
        adj_t = model()
        loss = criterion(model.user_factors, model.item_factors, adj_t, adj_valid_mat)  # this line is ugly
        # print('metapath: %s, valid loss: %.4f'%(metapath, loss.item()))
        return loss.item()



    best_loss = 1e8
    best_epoch = -1
    for epoch in range(epochs):
        train_loss = _MFtrain(metapath, mf, adj_mat, optimizer, epoch, reg_user, reg_item)
        scheduler.step()
        valid_loss = _MFvalid(metapath, mf, adj_valid_mat)
        if (epoch % log_step == 0) and epoch > 0:
            print("metapath: %s, epoch %d: loss = %.4f, lr = %.10f, reg_user = %f, reg_item = %f"
              % (metapath, epoch, train_loss, get_lr(optimizer), reg_user, reg_item))
            print('metapath: %s, valid loss: %.4f'%(metapath, valid_loss))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            mf.export(savepath, metapath)
            # state = {'net': mf.state_dict(), 'optimizer': optimizer.state_dict(), 'recall': best_recall,
            #          'epoch': epoch}
            # torch.save(state, args.save)
            # print('Model save for better valid loss: ', best_loss)
        if epoch - best_epoch >= patience:
            print("stop training at epoch ", epoch)
            break




    # n_user, n_item, adj_mat = _load_data(loadpath, metapath, device)


    # print("--------------------- n_user: {}, n_item: {}---------------------".format(n_user, n_item))

    # prev_loss = 0
    # set loss function

    # mf.train()
    # prev_loss = 0
    # # print("n_user: %d, n_item: %d" % (n_user, n_item))
    # for i in range(epochs):
    #     optimizer.zero_grad()
    #     adj_t = mf()
    #     loss = criterion(mf.user_factors, mf.item_factors, adj_t, adj_mat)   # this line is ugly
    #     loss.backward()
    #     optimizer.step()
    #     if i % 100 == 0:
    #         print("metapath: %s, epoch %d: loss = %.4f, lr = %.10f, reg_user = %f, reg_item = %f"
    #             % (metapath, i, loss, lr, reg_user, reg_item))
    #         if abs(int(loss) - prev_loss) < 1e-4:
    #             break
    #         prev_loss = int(loss)
    #


def train_FM(model, train_data, valid_data, epochs, lr, batch_size, decay_step, decay, patience, save, log_step, cuda):
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
    device = torch.device('cuda' if cuda else 'cpu')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True)
    # FM = model

    # if not criterion:
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss(reduction = 'none').to(device)
    # criterion.to(device)

    # validation and training interruption
    best_recall = 0.0
    best_epoch = -1
    # i = 0
    # set optimizer
    # optimizer = torch.optim.Adam([model.V, model.W, model.w0], lr=lr)  # maybe use weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.000001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay)
    model.train()
    train_loss = []
    for epoch in range(epochs):
        # t0 = gettime()
        for step, data in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = data       # indices: [[i, j], [i, j], ...]
            x = x.to(device)
            target = target.to(device).float()
            # batchsize, neg, dim = x.shape
            # x = x.view(-1, dim)
            # indices = indices.view(-1, 2)
            # target = target.view(-1)
            out = model(x)
            # Use a Sigmoid to classify
            # y_t = torch.sigmoid(out)
            # y_t = out.clone()
            # y_t = out.unsqueeze(1).repeat(1, 2)
            # y_t[:, 1] = 1 - y_t[:, 0]
            out = out.squeeze(-1)
            # print(out)
            # print(criterion(out, target))
            # time.sleep(10)
            # print(criterion(out, target))
            # time.sleep(10)
            loss = torch.mean(torch.sum(criterion(out, target), -1))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if (step % log_step == 0) and step > 0:
                print('Train epoch: {}[{}/{} ({:.0f}%)]\tLr:{:.6f}, Loss: {:.6f}, AvgL: {:.6f}'.format(epoch, step, len(train_loader),
                                100. * step / len(train_loader), get_lr(optimizer), loss.item(), np.mean(train_loss)))
        # print("epoch %d, loss = %f, lr = %f" % (epoch, loss, lr))
        scheduler.step()

        # Validation
        # if epoch % 2 == 0:
        loss, prec, recall, ndcg = eval(valid_data, model, criterion, 20, cuda=cuda)
        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'recall': best_recall, 'epoch': epoch}
            torch.save(state, save)
            print('Model save for better valid recall: ', best_recall)
        if epoch - best_epoch >= patience:
            print("stop training at epoch ", epoch)
            break
            # i += 1
            # if i > 2:
            #     break
        # elif loss < best_loss:
        #     best_loss = loss
        #     print("saving current model...")
                # FM.export()

def eval(valid_data, model, criterion, topK, cuda=False):
    r"""
    Calculate precision, recall and ndcg of the prediction of the model.
    Returns
    -------
    precision, recall, ndcg
    """
    device = torch.device('cuda:0' if cuda else 'cpu')

    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True, num_workers = 20, pin_memory = True)
    model.eval()

    loss_list = []
    prec_list = []
    recall_list = []
    ndcg_list = []
    for i, data in enumerate(valid_loader):
        x, items, labels = data
        x = x.to(device)
        labels = labels.to(device).float()
        # x = x.view(-1, x.shape[2])
        # items = items.view(-1)
        # labels = labels.view(-1)
        out = model(x)
        out = out.squeeze(-1)
        # Use a Sigmoid to classify
        # y_t = torch.sigmoid(out)
        # print(out)
        # y_t = out.clone()
        # y_t = y_t.unsqueeze(1).repeat(1, 2)
        # y_t[:, 1] = 1 - y_t[:, 0]
        # loss = criterion(y_t, labels)
        loss = torch.mean(torch.sum(criterion(out, labels), -1))
        values, indices = torch.topk(out, topK)
        ranklist = indices[0].tolist()
        gtItems = torch.nonzero(labels)[:, 1].tolist()
        # print(ranklist)
        # print(gtItems)
        # time.sleep(10)
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
    adj_path = args.data_path + 'adjs/'
    rate_path = args.data_path + 'rates/'
    feat_path = args.write_path + args.dataset + '_mf_features/'
    # train MF
    metapaths = ['UB', 'UUB', 'UBUB', 'UBCiB', 'UBCaB']
    if args.mf_train:
        train_MF(metapaths, 
                adj_path,
                rate_path,
                feat_path, 
                n_factor=args.mf_factor,
                lr=[5e-3, 5e-3, 5e-3, 5e-3, 5e-3], #, 5e-3, 5e-3, 7e-3, 7e-3],
                reg_user=[1e-1, 1e-1, 1e-1, 1e-1, 1e-1], #, 5e-1, 5e-1, 5e-1, 5e-1],
                reg_item=[1e-1, 1e-1, 1e-1, 1e-1, 1e-1], #, 5e-1, 5e-1, 5e-1, 5e-1],
                epoch=[50000, 50000, 50000, 50000, 50000], #, 20000, 10000, 50000, 50000],
                decay_step=100,
                decay=0.95,
                dataset=args.dataset,
                patience=args.mf_patience,
                log_step=args.mf_log_step,
                cuda=args.cuda)

    # train FM (cross entropy loss)
    # t0 = gettime()
    print("loading mf data...")
    # Do we need negative samples? Yes!
    # if args.cluster:
    train_data = read_pickle(rate_path+'rate_train')
    valid_data = read_pickle(rate_path+'valid_with_neg')
    test_data = read_pickle(rate_path+'test_with_neg')
    # else:
    #     train_data = read_pickle(rate_path+'train_data.pickle')
    #     valid_data = read_pickle(rate_path+'valid_with_neg_sample.pickle')
    #     test_data = read_pickle(rate_path+'test_with_neg_sample.pickle')
    # print("time cost: %f" % (gettime() - t0))

    # t0 = gettime()
    # print("loading features")

    # if args.cluster:
    #     temporary solution
        # adj_UB = read_pickle(adj_path+'adj_UB')
        # n_users = adj_UB.shape[0]
        # n_items = adj_UB.shape[1]
        # del adj_UB
    # else:
    #     if args.toy:
    #         small dataset
            # users = read_pickle(filtered_path+'users-small.pickle')
            # businesses = read_pickle(filtered_path+'businesses-small.pickle')
        # else:
        #     full dataset
            # users = read_pickle(filtered_path+'users-complete.pickle')
            # businesses = read_pickle(filtered_path+'businesses-complete.pickle')
        # n_users = len(users)
        # n_items = len(businesses)
    #
    # print("time cost: %f" % (gettime() - t0))

    # t0 = gettime()
    # print("making datasets...")
    # business_ids = set(i for i in range(n_items))
    # print("n_users:", n_users, "n_items:", n_items)
    user_features, item_features = load_feature(feat_path, metapaths)
    train_dataset = FMG_YelpDataset(train_data, user_features, item_features, neg_sample_n=args.negatives, mode='train', cuda=args.cuda)
    valid_dataset = FMG_YelpDataset(valid_data, user_features, item_features, neg_sample_n=20, mode='valid', cuda=args.cuda)
    test_dataset = FMG_YelpDataset(test_data, user_features, item_features, neg_sample_n=20, mode='test', cuda=args.cuda)
    # print("time cost: %f" % (gettime() - t0))
    #
    # t0 = gettime()
    print("start training FM...")
    device = torch.device('cuda' if args.cuda else 'cpu')
    model = FactorizationMachine(2*len(metapaths)*args.mf_factor, args.fm_factor, cuda=args.cuda).to(device)

    train_FM(model, train_dataset, valid_dataset, epochs=600, lr=5e-3, batch_size=args.batch_size, decay_step=1, decay=0.95, patience=args.fm_patience, save=args.save, log_step=args.fm_log_step, cuda=args.cuda)

    # print("time cost: %f" % (gettime() - t0))

    # result: loss gets lower as n_neg gets higher
    # Testing
    print("------------------test---------------")
    state = torch.load(args.save)
    model.load_state_dict(state['net'])
    model.to(device)
    eval(test_dataset, model, nn.BCELoss(reduction = 'none').to(device), 20, args.cuda)
    # print(model.W)
    # print(model.V)