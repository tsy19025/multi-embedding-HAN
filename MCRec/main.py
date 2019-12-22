from model import MCRec
import torch
import argparse
import utils
import pickle
from utils import yelpDataset

def parse_args():
    parse = argparse.ArgumentParser(description="Run MCRec.")
    parse.add_argument('--dataset', default = 'yelp', help = 'Choose a dataset.')
    parse.add_argument('--epochs', type = int, default = 30)
    parse.add_argument()

    return parse

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
    return loss

if __name__ == __main__:
    args = parse_args()
    if args.dataset == 'yelp':
        path_name = ['ub_path', 'uub_path', 'ubub_path', 'ubcab_path', 'ubcib_path']
        if os.exist('path_data/' + path_name[0]):
            paths = []
            for name in range(len(path_name)):
                with open('path_data/' + name, 'rb') as f:
                    paths.append(pickle.load(f))
        else:
            adj_BCa = pickel.load('../yelp_dataset/adjs/adj_BCa')
            adj_BCi = pickel.load('../yelp_dataset/adjs/adj_BCi')
            adj_UB = pickel.load('../yelp_dataset/adjs/adj_UB')
            adj_UU = pickel.load('../yelp_dataset/adjs/adj_UU')

            path_name = ['ub_path', 'uub_path', 'ubub_path', 'ubcab_path', 'ubcib_path']
            paths = get_path(adj_BCa, adj_BCi, adj_UB, adj_UU)
            for i in range(len(paths)):
                with open('path_data/' + path_name[i], 'wb') as f:
                    pickel.dump(paths[i], f, protocol = 4)

        num_to_ids = []
        num_to_id_names = ['num_to_userid', 'num_to_businessid', 'num_to_cityid', 'num_to_categoryid']
        for name in num_to_id_names:     
            num_to_id_paths.append('../yelp_dataset/adjs/' + name)
        for path in num_to_id_paths:
            with open(path, 'rb') as f:
                num_to_ids.append(pickle.load(f))
        n_node_list = [len(num_to_id) for num_to_id in num_to_ids]

        train_data_path = '../yelp_dataset/rates/rate_train'
        with open(train_data_path, 'rb') as f:
            train_data = pickel.load(f)
        train_data_loader = DataLoader(dataset = YelpDataset(n_node_list[0], n_node_list[1], train_data, paths, args.negetive_number),
                                       batch_size = args.batch_size,
                                       shuffle = True,
                                       num_workers = 20,
                                       pin_memory = True)

        valid_data_path = '../yelp_dataset/rates/valid_train'
        with open(valid_data_path, 'rb') as f:
            valid_data = pickel.load(f)
        valid_data_loader = DataLoader(dataset = YelpDataset(n_node_list[0], n_node_list[1], valid_data, paths, 0),
                                       batch_size = 1,
                                       shuffle = True,
                                       num_workers = 20,
                                       pin_memory = True)

    use_cuda = torch.cuda.isavailable() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = MCRec(n_node_list, args)
    model = model.to(device)

    loss_fn = 
    valid_loss_fn = 
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
