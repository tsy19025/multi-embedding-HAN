from model import NeuACF
from utils import YelpDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
from metrics import *
import argparse
import pickle
import pandas as pd
import sys

def parse_args():
    parse = argparse.ArgumentParser(description="Run NeuACF")
    parse.add_argument('--dataset', default = 'yelp')
    # parse.add_argument('--data_path', type = str, default = '/home1/tsy/Project/multi-embedding-HAN/tmpdataset/')
    parse.add_argument('--data_path', type = str, default = '/home1/wyf/Projects/gnn4rec/multi-embedding-HAN/yelp_dataset/')
    parse.add_argument('--mat_path', type = str, default = '/home1/tsy/Project/multi-embedding-HAN/NeuACF/mat_100/')
    # parse.add_argument('--mat', type = list, default = ['U.UBU', 'B.BUB', 'U.UBCiBU', 'B.BCiB', 'U.UBCaBU', 'B.BCaB'])
    parse.add_argument('--mat', type = list, default = ['U.UBU', 'B.BUB'])
    parse.add_argument('--epochs', type = int, default = 10000)
    parse.add_argument('--last_layer_size', type = int, default = 64)
    parse.add_argument('--nlayer', type = int, default = 2)
    parse.add_argument('--hidden_size', type = int, default = 600)
    parse.add_argument('--negatives', type = int, default = 2)
    parse.add_argument('--learn_rate', type = float, default = 0.00005)
    parse.add_argument('--batch_size', type = int, default = 64)
    parse.add_argument('--mat_select', type = str, default = 'median')
    parse.add_argument('--merge', type = str, default = 'attention')
    parse.add_argument('--K', type = int, default = 10)
    parse.add_argument('--patience', type = int, default = 15)
    parse.add_argument('--decay_step', type = int, default = 1)
    parse.add_argument('--log_step', type=int, default=1e2)
    parse.add_argument('--decay', type = float, default = 0.98)
    parse.add_argument('--save', type = str, default = 'model/modelpara.pth')
    parse.add_argument('--cuda', type = bool, default = True)
    parse.add_argument('--mode', type = str, default = 'train')
    
    return parse.parse_args()

def train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch, device):
    print("Epoch:", epoch)
    model.train()
    train_loss = []
    for step, data in enumerate(train_data_loader):
        user_input, item_input, label = data

        batch_size, negatives, nmat, features = user_input.shape
        user_input = user_input.view(-1, nmat, features).to(device)
        item_input = item_input.view(batch_size * negatives, nmat, -1).to(device)
        label = label.view(-1).to(device)

        optimizer.zero_grad()
        output = model(user_input, item_input).squeeze(-1)
        loss = loss_fn(output, label)
        loss = torch.mean(torch.sum(loss, -1))
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if (step % args.log_step == 0) and step > 0:
            print('Train epoch: {}[{}/{} ({:.0f}%)] Loss: {:.6f}, AvgL: {:.6f}'.format(epoch, step, len(train_data_loader),
                  100. * step / len(train_data_loader), loss.item(), np.mean(train_loss)))
    return np.mean(train_loss)

def eval(model, eval_data_loader, device, K, loss_fn):
    model.eval()
    eval_p = []
    eval_r = []
    eval_ndcg = []
    eval_loss = []
    with torch.no_grad():
        for step, batch_data in enumerate(eval_data_loader):
            user_input, item_input, label = batch_data
            user_input = user_input.squeeze(0).to(device)
            item_input = item_input.squeeze(0).to(device)
            label = label.view(-1).to(device)
            output = model(user_input, item_input).squeeze()

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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("Begin")
    args = parse_args()

    if args.dataset == 'yelp':
        adjs_path = args.data_path + 'adjs_100/'
        with open(adjs_path + 'adj_UB', 'rb') as f: adj_UB = pickle.load(f)

        mat_list = args.mat
        n_mat = len(mat_list)
        mat_path = args.mat_path
        mat_select = args.mat_select
        U_feature_dir = mat_path + mat_list[0]+".pathsim.feature." + mat_select
        I_feature_dir = mat_path + mat_list[1]+".pathsim.feature." + mat_select
        
        if n_mat == 2:
            U_feature_dir2 = mat_path + mat_list[0]+".pathsim.feature."+ mat_select
            I_feature_dir2 = mat_path + mat_list[1]+".pathsim.feature."+ mat_select
            U_feature_dir3 = mat_path + mat_list[0]+".pathsim.feature."+ mat_select
            I_feature_dir3 = mat_path + mat_list[1]+".pathsim.feature."+ mat_select
            U_feature_dir4 = mat_path + mat_list[0]+".pathsim.feature."+ mat_select
            I_feature_dir4 = mat_path + mat_list[1]+".pathsim.feature."+ mat_select
        elif n_mat == 6:
            U_feature_dir2 = mat_path + mat_list[2]+".pathsim.feature." + mat_select
            I_feature_dir2 = mat_path + mat_list[3]+".pathsim.feature."+ mat_select
            U_feature_dir3 = mat_path + mat_list[4]+".pathsim.feature."+ mat_select
            I_feature_dir3 = mat_path + mat_list[5]+".pathsim.feature."+ mat_select
            U_feature_dir4 = mat_path + mat_list[0]+".pathsim.feature."+ mat_select
            I_feature_dir4 = mat_path + mat_list[1]+".pathsim.feature."+ mat_select
        elif n_mat == 8:
            U_feature_dir2 = mat_path + mat_list[2]+".pathsim.feature." + mat_select
            I_feature_dir2 = mat_path + mat_list[3]+".pathsim.feature."+ mat_select
            U_feature_dir3 = mat_path + mat_list[4]+".pathsim.feature."+ mat_select
            I_feature_dir3 = mat_path + mat_list[5]+".pathsim.feature."+ mat_select
            U_feature_dir4 = mat_path + mat_list[6]+".pathsim.feature."+ mat_select
            I_feature_dir4 = mat_path + mat_list[7]+".pathsim.feature."+ mat_select

        u_embedding = []
        i_embedding = []
        u_feature_num = []
        i_feature_num = []

        U_feature = pd.read_csv( U_feature_dir, sep=",", header=None ).fillna( 0 ).as_matrix()
        print( "# U_feature1 shape:", U_feature.shape  )
        I_feature = pd.read_csv( I_feature_dir, sep=",", header=None ).fillna( 0 ).as_matrix()
        print( "# I_feature1 shape:", I_feature.shape  )
        u_embedding.append(U_feature)
        i_embedding.append(I_feature)
        i_feature_num.append(I_feature.shape[1])
        u_feature_num.append(U_feature.shape[1])
        
        U_feature2 = pd.read_csv( U_feature_dir2, sep=",", header=None ).fillna( 0 ).as_matrix()
        print( "# U_feature2 shape:", U_feature2.shape  )
        I_feature2 = pd.read_csv( I_feature_dir2, sep=",", header=None ).fillna( 0 ).as_matrix()
        print( "# I_feature2 shape:", I_feature2.shape  )
        u_embedding.append(U_feature2)
        i_embedding.append(I_feature2)
        i_feature_num.append(I_feature2.shape[1])
        u_feature_num.append(U_feature2.shape[1])
        
        U_feature3 = pd.read_csv( U_feature_dir3, sep=",", header=None ).fillna( 0 ).as_matrix()
        print( "# U_feature3 shape:", U_feature3.shape  )
        I_feature3 = pd.read_csv( I_feature_dir3, sep=",", header=None ).fillna( 0 ).as_matrix()
        print( "# I_feature3 shape:", I_feature3.shape  )
        u_embedding.append(U_feature3)
        i_embedding.append(I_feature3)
        i_feature_num.append(I_feature3.shape[1])
        u_feature_num.append(U_feature3.shape[1])
        
        U_feature4 = pd.read_csv( U_feature_dir4, sep=",", header=None ).fillna( 0 ).as_matrix()
        print( "# U_feature3 shape:", U_feature4.shape  )
        I_feature4 = pd.read_csv( I_feature_dir4, sep=",", header=None ).fillna( 0 ).as_matrix()
        print( "# I_feature3 shape:", I_feature4.shape  )
        u_embedding.append(U_feature4)
        i_embedding.append(I_feature4)
        i_feature_num.append(I_feature4.shape[1])
        u_feature_num.append(U_feature4.shape[1])

        with open(args.data_path + 'rates_100/rate_train', 'rb') as f: train_data = pickle.load(f)
        train_data_loader = DataLoader(dataset = YelpDataset(train_data, args.negatives, adj_UB, 'train', u_embedding, i_embedding),
                                       batch_size = args.batch_size, shuffle = True,
                                       num_workers = 20, pin_memory = True)
        
        with open(args.data_path + 'rates_100/valid_with_neg', 'rb') as f: valid_data = pickle.load(f)
        valid_data_loader = DataLoader(dataset = YelpDataset(valid_data, args.negatives, adj_UB, 'valid', u_embedding, i_embedding),
                                       batch_size = 1, shuffle = True, num_workers = 20, pin_memory = True)
        
        with open(args.data_path + 'rates_100/test_with_neg', 'rb') as f: test_data = pickle.load(f)
        test_data_loader = DataLoader(dataset = YelpDataset(test_data, args.negatives, adj_UB, 'test', u_embedding, i_embedding),
                                      batch_size = 1, shuffle = True, num_workers = 20, pin_memory = True)
    
    print("Begin Training.")
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = NeuACF(u_feature_num, i_feature_num, args.hidden_size, args.last_layer_size, n_mat, args.nlayer, args.merge, device)
    model = model.to(device)
    # for m in model.modules():
    #     if isinstance(m, (nn.Linear)):
    #         nn.init.xavier_uniform_(m.weight)

    loss_fn = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learn_rate, weight_decay=0.000001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = args.decay_step, gamma = args.decay)

    if args.mode == 'train':
        best_recall = 0
        best_epoch = -1
        for epoch in range(args.epochs):
            mean_loss = train_one_epoch(model, train_data_loader, optimizer, loss_fn, epoch, device)
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
    state = torch.load(args.save)
    model.load_state_dict(state['net'])
    model.to(device)
    _ = test(model, test_data_loader, loss_fn)
