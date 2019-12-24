from model import MCRec
import torch
from torch.utils.data import DataLoader
import argparse
import utils
import pickle
import utils
from utils import YelpDataset
import os
import sys
from sklearn.externals import joblib

def parse_args():
    parse = argparse.ArgumentParser(description="Run MCRec.")
    parse.add_argument('--dataset', default = 'yelp', help = 'Choose a dataset.')
    parse.add_argument('--epochs', type = int, default = 30)
    parse.add_argument('--adjs_path', type = str, default = '../yelp_dataset/adjs')
    parse.add_argument('--negetives', type = int, default = 10)
    parse.add_argument('--batch_size', type = int, default = 60)
    parse.add_argument('--dim', type = int, default = 64)
    parse.add_argument('--sample', type = int, default = 5)
    # parse.add_argument()

    return parse.parse_args()

def train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch):
    model.train()
    sum_loss = 0
    cnt = 0

    for step, data in enumerate(train_data_loader):
        user_input, item_input, paths = data
        user_input = user_input.to(device)
        item_input = item_input.to(device)
        path_input = [path.to(device) for path in paths]

        output = model(user_input, item_input, path_input)

        loss = loss_fn(output, label)
        sum_loss = sum_loss + loss.data
        loss.backward()

        optimzer.step()
        cnt = cnt + 1
    return sum_loss / cnt

def valid(model, valid_data_loader, loss_fn):
    return 0

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
            if not os.path.exists('path_data/'): os.mkdir('path_data/')
            for i in range(5):
                with open('path_data/' + path_name[i], 'wb') as f:
                    pickle.dump(paths[i], f)
        path_num = [args.sample] * 5
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
        train_data_loader = DataLoader(dataset = YelpDataset(users, items, train_data, paths, path_num, timestamps, args.negetives),
                                       batch_size = args.batch_size,
                                       shuffle = True,
                                       num_workers = 20,
                                       pin_memory = True)

        valid_data_path = '../yelp_dataset/rates/valid_train'
        with open(valid_data_path, 'rb') as f:
            valid_data = pickle.load(f)
        valid_data_loader = DataLoader(dataset = YelpDataset(users, items, valid_data, paths, path_num, timestamps, 0),
                                       batch_size = 1,
                                       shuffle = True,
                                       num_workers = 20,
                                       pin_memory = True)
    sys.exit(0)
    use_cuda = torch.cuda.isavailable() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = MCRec(users, items, paths, path_num, timestampe, args.dim)
    model = model.to(device)

    loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(device)
    valid_loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay)
    
    best_loss = 100000
    for epoch in range(args.epochs):
        mean_loss = train_one_epoch(model, train_data_load, optimizer, loss_fn, epoch)
        valid_loss = valid(model, valid_data_load, valid_loss_fn)
        if valid_loss < best_loss:
            best_loss = valid_loss
            with open(args.save, 'wb') as f:
                torch.save(model.state_dict(), f)
