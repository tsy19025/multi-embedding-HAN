import copy
import os
import pickle
import random
import time

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from loss import MFLoss
from utils import read_pickle, write_pickle

gettime = lambda: time.time()

class MatrixFactorizer(nn.Module):
    """
    Updating Embeddings within the model is OK
    """
    def __init__(self, n_user, n_item, n_factor=10, cuda=False):
        r"""
        Parameters
        ----------
        n_user: int
            number of users

        n_item: int
            number of items

        n_factor: int
            embedding dim
        """
        super(MatrixFactorizer, self).__init__()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.user_factors = torch.randn([n_user, n_factor], dtype=torch.float32, requires_grad=True, device=self.device)
        self.item_factors = torch.randn([n_item, n_factor], dtype=torch.float32, requires_grad=True, device=self.device)

    def forward(self):
        r"""
        Return
        ------
        adj_t: Tensor with shape n_user*n_item, 
            the predicted adjacent matrix
        """
        # to calculate matrix factorization, we need
        # to compute the adj matrix by multiplying
        # the two embedding matrices
        return self.user_factors.mm(self.item_factors.T)

    def export(self, filepath, metapath):
        r"""
        export the embeddings to files
        """
        user_factors = self.user_factors.detach()
        item_factors = self.item_factors.detach()
        user_file = filepath + metapath + '_user' + '.pickle'
        item_file = filepath + metapath + '_item' + '.pickle'
        write_pickle(user_file, user_factors)
        write_pickle(item_file, item_factors)

class MatrixFactorization(nn.Module):
    
    def __init__(self, n_users, n_items, n_factors=20, cuda=False):
        super().__init__()
        device = torch.device('cuda:0' if cuda else 'cpu')
        self.user_factors = torch.nn.Embedding(n_users, 
                                               n_factors,
                                               device=device)
        self.item_factors = torch.nn.Embedding(n_items, 
                                               n_factors,
                                               device=device)
        
    def forward(self, user, item):
        r"""
        Parameter
        ---------
        user: int, index of user

        item: int, index of item
        """
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

class FactorizationMachine(nn.Module):
    def __init__(self, n=None, k=None, cuda=False):
        r"""
        Parameters
        ----------
        n: int
            number of embeddings, n = n_user + n_item
        
        k: int
            dimension of each embedding
        """
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.V = torch.randn([n, k], dtype=torch.float32, requires_grad=True, device=self.device)
        self.W = torch.randn(n, dtype=torch.float32, requires_grad=True, device=self.device)
        self.w0 = torch.randn(1, dtype=torch.float32, requires_grad=True, device=self.device)
        # self.lin = nn.Linear(n, 1).to(self.device)  # nn.Linear also contains the bias

    def forward(self, x):
        r"""
        Parameters
        ----------
        x: torch.Tensor, shape [batch_size, (2*n_metapath*K)], dimension 2\
        
        Return
        ------
        out: scalar
            prediction of y
        """
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)        # S_1^2, S_1 can refer to statistics book
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        # x = x.unsqueeze(1)
        # out_1 = torch.mul(self.V, x).pow(2).sum(1, keepdim=True)         # (\sum v_ik*x)^2
        # out_2 = torch.mul(self.V.pow(2), x.pow(2)).sum(1, keepdim=True)  # \sum (v_ik^2 * x^2)
        out_inter = 0.5*(out_1 - out_2).sum(1)                             # \sum(<vi, vj>*xi*xj)
        # out_lin = torch.matmul(self.W, x) + self.w0
        out_lin = torch.matmul(self.W, x.T) + self.w0
        out = out_inter + out_lin
        out = out.unsqueeze(1).repeat(1, 2)
        out[:, 1] = -out[:, 0]
        return out

    def export(self):
        path = 'yelp_dataset/fm_res/'
        V_filename = 'FM_V.pickle'
        W_filename = 'FM_W.pickle'
        V = self.V.detach()
        W = [self.W.detach(), self.w0.detach()]
        write_pickle(path+V_filename, V)
        write_pickle(path+W_filename, W)

    def load(self, filenames):
        r"""
        load parameters from files
        """
        V_file, W_file = filenames
        self.V = read_pickle(V_file)
        self.W, self.w0 = read_pickle(W_file)



if __name__ == "__main__":
    # Test function
    pass