import json
import numpy as np
from numpy import array
from scipy import sparse
import torch
from torch.utils.data import Dataset

def load_jsondata_from_file(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def matrix_factorization(adj_user_review, adj_user_business, adj_review_business, k, lr = 0.001):
    tot_user, tot_review = adj_user_review.shape
    _, tot_business = adj_user_business.shape
    
    W_user = Variable(torch.randn(tot_user, k), requires_grad = True)
    W_review = Variable(torch.randn(tot_review, k), requires_grad = True)
    W_business = Variable(torch.randn(tot_business, k), requires_grad = True)
    
    old_loss = 0
    while (True):
        user_review = W_user.mm(W_review.t())
        user_business = W_user.mm(W_business.t())
        review_business = W_review.mm(W_business.t())
        loss = (user_review - adj_user_review).pow(2).sum() + (user_business - adj_user_business).pow(2).sum() + (review_business - adj_review_business).pow(2).sum()

        # print("loss:{:.4f}".format(loss.data))
        if abs(loss.data - old_loss) < 1e-5: break
        old_loss = loss.data
        loss.backward()
        
        W_user.data -= lr * W_user.grad.data
        W_review.data -= lr * W_review.grad.data
        W_business.data -= lr * W_business.grad.data
        
        W_user.grad.data.zero_()
        W_review.grad.data.zero_()
        W_business.grad.data.zero_()
        
        # print(W_user)
        # print(W_review)
        # print(W_business)     
    return W_user, W_review, W_business

class YelpDataset(Dataset):

    def __init__(self, review_path, reviewid_to_num, userid_to_num, businessid_to_num):
        xy = load_jsondata_from_file(review_path)
        self.x_user = torch.LongTensor([userid_to_num[r['user_id']] for r in xy], requires_grad=False)
        self.x_business = torch.LongTensor([userid_to_num[r['business_id']] for r in xy], requires_grad=False)
        self.y = torch.FloatTensor([float(r["stars"]) for r in xy], requires_grad=False)
        self.len = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x_user[index], self.x_business[index], self.y[index]

    def __len__(self):
        return self.len

class Metapath():
    def __init__(self, name, pathtype, adj):
        # pathtype: 0.user 1.review 2.business
        self.name = name
        self.pathtype = pathtype
        ix = torch.LongTensor(adj.row)
        iy = torch.LongTensor(adj.col)
        idata = torch.FloatTensor(adj.data)
        ixy = torch.cat([ix.unsqueeze(0), iy.unsqueeze(0)], dim = 0)
        self.adj = torch.sparse.FloatTensor(ixy, idata, adj.shape)
        
