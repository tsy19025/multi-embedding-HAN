from model import MCRec
import torch
import argparse
import utils
import pickle

def parse_args():
    parse = argparse.ArgumentParser(description="Run MCRec.")
    parse.add_argument('--dataset', default = 'yelp', help = 'Choose a dataset.')
    parse.add_argument('--epochs', type = int, default = 30)
    parse.add_argument()

    return parse

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
        Dataset = YelpDataset
    
    use_cuda = torch.cuda.isavailable() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = MCRec(n_node_list, args)
    model = model.to(device)
    
    for epoch in range(args.epochs):
        train_one_epoch() 
