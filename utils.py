import json
import numpy as np
from numpy import array
from scipy import sparse
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import NMF

def load_jsondata_from_file(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_type_embedding(adj, type_feature):
    def matrix_factorization(Q, n):
        model = NMF(n_components = n, alpha = 0.01)
        W = model.fit_transform(Q)
        return W, model.components_

    W, H = matrix_factorization(adj, type_feature)
    H = H.T
    
    def add_sum(mat):
        m = len(mat[0])
        z = np.zeros(m)
        for line in mat:
            z = z + line
        # Softmax
        return np.exp(z)/sum(np.exp(z))
    return add_sum(W), add_sum(H)

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
        
