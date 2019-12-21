import argparse
import numpy as np
import torch
import pickle
import math
from torch.utils.data import DataLoader
import torch.nn as nn
from models import multi_HAN
from torch.optim import lr_scheduler
from utils import YelpDataset
from metrics import *
import time

def parse_args():
    parser = argparse.ArgumentParser(description='multi-embedding-HAN')
    parser.add_argument('--emb_dim', type=int, default=10,
                        help='dimension of embeddings')
    parser.add_argument('--n_facet', type=int, default=10,
                        help='number of facet for each embedding')
    parser.add_argument('--n_neigh', type=int, default=50,
                        help='number of neighbor to sample')
    parser.add_argument('--n_iter', type=int, default=5,
                        help='number of iterations when routing')
    parser.add_argument('--n_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='initial learning rate')
    parser.add_argument('--decay', type=float, default=0.98,
                        help='learning rate decay rate')
    parser.add_argument('--decay_step', type=int, default=1,
                        help='learning rate decay step')
    parser.add_argument('--log_step', type=int, default=1e2,
                        help='log print step')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=30,
                        help='upper epoch limit')
    parser.add_argument('--patience', type=int, default=5,
                        help='Extra iterations before early-stopping')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='batch size')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use GPU for training')
    parser.add_argument('--save', type=str, default='model/',
                        help='path to save the final model')
    # parser.add_argument('--load_checkpoint_path', type=str, default='',
    #                     help='path to load checkpoint')
    # parser.add_argument('--optimizer', type=str, default='adam',
    #                     help='optimizer to use (sgd, adam)')
    parser.add_argument('--dataset', default='yelp',
                        help='dataset name')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', type=str,
                        help='Passing "test" will only run a single evaluation, otherwise full training will be performed')
    args = parser.parse_args()
    args.save = args.save + args.dataset
    args.save = args.save + '_batch{}'.format(args.batch_size)
    args.save = args.save + '_lr{}'.format(args.lr)
    args.save = args.save + '_emb{}'.format(args.emb_dim)
    args.save = args.save + '_facet{}'.format(args.n_facet)
    args.save = args.save + '_iter{}'.format(args.n_iter)
    args.save = args.save + '_neighsize{}'.format(args.n_neigh)
    args.save = args.save + '_negsize{}'.format(args.n_neg)
    args.save = args.save + '_decay{}'.format(args.decay)
    args.save = args.save + '_decaystep{}'.format(args.decay_step)
    args.save = args.save + '_patience{}.pt'.format(args.patience)
    return args


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch, device, args):
    model.train()
    epoch_loss = []
    for step, batch_data in enumerate(train_data_loader):
        # user, pos_business, neg_businesses, label, user_neigh_list_lists, pos_business_neigh_list_lists, neg_business_neigh_list_lists = batch_data
        users, businesses, label, user_neigh_list_lists, business_neigh_list_lists = batch_data
        users = users.to(device)
        businesses = businesses.to(device)
        # neg_businesses = [neg_business.to(device) for neg_business in neg_businesses]
        label = label.to(device)
        user_neigh_list_lists = [[neigh.to(device) for neigh in user_neigh_list] for user_neigh_list in
                                 user_neigh_list_lists]
        business_neigh_list_lists = [[neigh.to(device) for neigh in business_neigh_list] for business_neigh_list in
                                     business_neigh_list_lists]
        # neg_business_neigh_list_lists = [[[neigh.to(device) for neigh in neg_business_neigh_list] for neg_business_neigh_list in
        #                              neg_business_neigh_list_lists[neg]] for neg in range(args.n_neg)]
        # print('user')
        # print(user.shape)
        # print('pb')
        # print(pos_business.shape)
        # print('nb')
        # print([n.shape for n in neg_businesses])
        # print('un')
        # print([[n.shape for n in list] for list in user_neigh_list_lists])
        # print('pn')
        # print([[n.shape for n in list] for list in business_neigh_list_lists])
        # time.sleep(10)

        optimizer.zero_grad()
        output = model(users, businesses, user_neigh_list_lists, business_neigh_list_lists)
        loss = loss_fn(output, label)
        loss = torch.mean(torch.sum(loss, 1))
        loss.backward()
        # for name, parms in model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        optimizer.step()
        epoch_loss.append(loss.item())
        if (step % args.log_step == 0) and step > 0:
            print('Train epoch: {}[{}/{} ({:.0f}%)]\tLr:{:.6f}, Loss: {:.6f}, AvgL: {:.6f}'.format(epoch, step, len(train_data_loader),
                                                    100. * step / len(train_data_loader), get_lr(optimizer), loss.item(), np.mean(epoch_loss)))

    mean_epoch_loss = np.mean(epoch_loss)
    return mean_epoch_loss

# def evaluate(logit, label, K):
#     # out = torch.cat([logit, label], 0)
#     pred_items = torch.topk(logit, K, dim=1)[1][0]
#     pred_items = pred_items.tolist()
#     gt_items = torch.nonzero(label)[:, 1].tolist()
#     p_at_k = getP(pred_items, gt_items)
#     r_at_k = getR(pred_items, gt_items)
#     ndcg_at_k = getNDCG(pred_items, gt_items)
#     # ndcg = (math.log(2) / (torch.nonzero(ifzero)[:,1].view(-1).to(torch.float)+2).log_()).sum()
#     return p_at_k, r_at_k, ndcg_at_k

def eval(model, eval_data_loader, device, K):
    model.eval()
    # valid_loss = 0
    # n_valid = 0
    eval_p = []
    eval_r = []
    eval_ndcg = []
    with torch.no_grad():
        for step, batch_data in enumerate(eval_data_loader):
            users, businesses, label, user_neigh_list_lists, business_neigh_list_lists = batch_data
            # n_valid += len(label)
            # n_valid += 1
            users = users.to(device)
            businesses = businesses.to(device)
            label = label.to(device)
            user_neigh_list_lists = [[neigh.to(device) for neigh in user_neigh_list] for user_neigh_list in
                                     user_neigh_list_lists]
            business_neigh_list_lists = [[neigh.to(device) for neigh in business_neigh_list] for
                                                            business_neigh_list in business_neigh_list_lists]
            logit = model(users, businesses, user_neigh_list_lists, business_neigh_list_lists)
            # loss = loss_fn(output, label)
            # valid_loss += loss.item()
            # p_at_k, r_at_k, ndcg_at_k = evaluate(logit, label, 20)
            pred_items = torch.topk(logit, K, dim=1)[1][0]
            pred_items = pred_items.tolist()
            gt_items = torch.nonzero(label)[:, 1].tolist()
            p_at_k = getP(pred_items, gt_items)
            r_at_k = getR(pred_items, gt_items)
            ndcg_at_k = getNDCG(pred_items, gt_items)

            eval_p.append(p_at_k)
            eval_r.append(r_at_k)
            eval_ndcg.append(ndcg_at_k)
        # mean_valid_loss = valid_loss/n_valid
    # print('Valid:\tLoss:%f' % (mean_valid_loss))
    mean_p = np.mean(eval_p)
    mean_r = np.mean(eval_r)
    mean_ndcg = np.mean(eval_ndcg)
    return mean_p, mean_r, mean_ndcg

def valid(model, valid_data_loader, device):
    print('Start Valid')
    mean_p, mean_r, mean_ndcg = eval(model, valid_data_loader, device, 20)
    print('Valid:\tprecision@10:%f, recall@10:%f, ndcg@10:%f' % (mean_p, mean_r, mean_ndcg))
    return mean_p, mean_r, mean_ndcg

def test(model, test_data_loader, device):
    print('Start Test')
    mean_p, mean_r, mean_ndcg = eval(model, test_data_loader, device, 20)
    print('Test:\tprecision@10:%f, recall@10:%f, ndcg@10:%f' % (mean_p, mean_r, mean_ndcg))
    # model.eval()
    # # evaluation = 0
    # # n_eval = 0
    # test_p = []
    # test_r = []
    # test_ndcg = []
    # with torch.no_grad():
    #     for step, batch_data in enumerate(evaluate_data_loader):
    #         users, businesses, label, user_neigh_list_lists, business_neigh_list_lists = batch_data
    #         users = users.to(device)
    #         businesses = businesses.to(device)
    #         label = label.to(device)
    #         user_neigh_list_lists = [[neigh.to(device) for neigh in user_neigh_list] for user_neigh_list in
    #                                  user_neigh_list_lists]
    #         business_neigh_list_lists = [[neigh.to(device) for neigh in business_neigh_list] for
    #                                      business_neigh_list in business_neigh_list_lists]
    #         logit = model(users, businesses, user_neigh_list_lists, business_neigh_list_lists)
    #         p_at_k, r_at_k, ndcg_at_k = evaluate(logit, label, 20)
    #         test_p.append(p_at_k)
    #         test_r.append(r_at_k)
    #         test_ndcg.append(ndcg_at_k)
    #         eval = evaluate_fn(output, label)
    #         evaluation += eval.item()
    #     mean_evaluation = evaluation/n_eval
    # print('Eval:\tMAE:%f' % (mean_evaluation))
    # return mean_evaluation

if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'yelp':
        train_data_path = 'yelp_dataset/rates/rate_train'
        valid_data_path = 'yelp_dataset/rates/valid_with_neg'
        adj_paths = []
        adj_names = ['adj_UU', 'adj_UB', 'adj_BCi', 'adj_BCa', 'adj_UUB', 'adj_UBU', 'adj_UBUB', 'adj_UBCi', 'adj_UBCa', 'adj_BCaB', 'adj_BCiB']
        for name in adj_names:
            adj_paths.append('yelp_dataset/adjs/' + name)
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
        Dataset = YelpDataset
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = multi_HAN(n_nodes_list, args)
    if args.mode == 'train':
        model = model.to(device)
        train_data_loader = DataLoader(dataset=Dataset(n_nodes_list, train_data_path, adj_paths, args.n_neigh, args.n_neg, 'train'),
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=20,
                                       pin_memory=True)

        valid_data_loader = DataLoader(dataset=Dataset(n_nodes_list, valid_data_path, adj_paths, args.n_neigh, args.n_neg, 'valid'),
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=20,
                                       pin_memory=True)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay)
        # loss_fn = nn.MSELoss(reduction='mean')
        loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(device)
        best_loss = 100.0
        best_epoch = -1
        for epoch in range(args.epochs):
            print('Start epoch: ', epoch)
            mean_loss = train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch, device, args)
            # total_loss += loss.data[0]
            # valid_fn = nn.MSELoss(reduction='sum')
            valid_fn = nn.BCEWithLogitsLoss(reduction='none').to(device)
            _, _, valid_loss = valid(model, valid_data_loader, device)
            scheduler.step()
            if valid_loss < best_loss:
                best_epoch = epoch
                best_loss = valid_loss
                with open(args.save, 'wb') as f:
                    torch.save(model.state_dict(), f)
                print('Model save for lower valid loss %f' % best_loss)
            if epoch - best_epoch >= args.patience:
                print('Stop training after %i epochs without improvement on validation.' % args.patience)
                break
    else:
        test_data_path = 'yelp_dataset/rates/rate_test'
        test_data_loader = DataLoader(dataset=YelpDataset(n_nodes_list, test_data_path, adj_paths, args.n_neigh, args.n_neg, 'test'),
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=20,
                                        pin_memory=True)
        model.load_state_dict(torch.load(args.save))
        model.to(device)
        test(model, test_data_loader, device)
        # evaluate_fn = nn.L1Loss(reduction='sum')
        # evaluate(model, evaluate_data_loader, evaluate_fn)
