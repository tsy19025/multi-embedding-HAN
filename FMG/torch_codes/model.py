import os
import pickle
import time

import numpy as np
import torch
from torch import nn
from loss import MFLoss

gettime = lambda: time.time()

class MatrixFactorizer(nn.Module):
    """
    Updating Embeddings within the model is OK
    """
    def __init__(self, n_user, n_item, n_factor=10, iters=500, cuda=False):
        r"""
        Parameters
        ----------
        n_user: int
            number of users

        n_item: int
            number of items

       n_factor: int
            embedding dim

        iters: int
            number of iteration
        """
        super(MatrixFactorizer, self).__init__()
        self.iters = iters
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor
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
        user_file = filepath + metapath + '_user'
        item_file = filepath + metapath + '_item'
        with open(user_file+'.pickle', 'wb') as fw:
            pickle.dump(self.user_factors, fw)
        with open(item_file+'.pickle', 'wb') as fw:
            pickle.dump(self.item_factors, fw)

class MFTrainer(object):
    r"""
    Training wrapper of MatrixFactorizer
    """
    def __init__(self, metapath, loadpath, savepath, epoch=20, n_factor=10, iters=500, is_binary=True, cuda=False):
        self.metapath = metapath
        self.loadpath = loadpath
        self.savepath = savepath
        self.epoch = epoch
        self.n_factor = n_factor
        self.iters = iters
        self.cuda = cuda
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.n_user, self.n_item, self.adj_mat = self._load_data(loadpath, metapath, is_binary)

        # instance model
        self.mf = MatrixFactorizer(self.n_user, self.n_item, self.n_factor, self.iters, self.cuda)
    
    def _load_data(self, filepath, metapath, is_binary=True):
        r"""
        Parameters
        ----------
        filepath: str

        metapath: str

        is_binary: bool, 
            if the files are binary files
        
        Return
        ------
        n_user: int
            number of users

        n_item: int
            number of businesses

        data: torch.Tensor(requires_grad=False)
            the adjacency matrix to be factorized
        """
        data = []
        if is_binary == True:
            file = filepath + 'adj_' + metapath
            with open(file, 'rb') as fw:
                adjacency = pickle.load(fw)
                data = torch.tensor(adjacency, dtype=torch.float32, requires_grad=False).to(self.device)
        if is_binary == False:
            """ TODO: read txt file """
            raise NotImplementedError

        n_user, n_item = data.shape
        return n_user, n_item, data

    def _export(self, filepath, metapath):
        r"""
        export the matrix factors to files
        """
        self.mf.export(filepath, metapath)

    def train(self, lr=1e-4, reg_user=5e-2, reg_item=5e-2):
        r"""
        Parameters
        ----------
        lr: learning rate

        reg_user: regularization coefficient

        reg_item: regularization coefficient
        """
        # set loss function
        criterion = MFLoss(reg_user, reg_item).to(self.device)
        optimizer = torch.optim.SGD([self.mf.user_factors, self.mf.item_factors], lr=lr, weight_decay=0.2)  # use weight_decay
        self.mf.train()
        print("n_user: %d, n_item: %d" % (self.n_user, self.n_item))
        for i in range(self.epoch + 1):
            self.mf.zero_grad()
            adj_t = self.mf()
            loss = criterion(self.mf.user_factors, self.mf.item_factors, adj_t, self.adj_mat)   # this line is ugly
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print("metapath: %s, epoch %d: loss = %f, lr = %f, reg_user = %f, reg_item = %f" 
                    % (self.metapath, i, loss, lr, reg_user, reg_item))
                
        self._export(self.savepath, self.metapath)


# class FactorizationMachine(nn.Module):
#     r"""
#     Parameters
#     ----------

#     """
#     def __init__(self, x):
#         super(FactorizationMachine, self).__init__()
#         self.factor = nn.Embedding()
#         self.weight = nn.Embedding()

#     def forward(self, x):
#         r"""
#         Parameters
#         ----------
#         x: torch.Tensor.sparse, shape 1*((n_user+n_item)*K)
#             input embedding of each user-business pair, saved in COO form
        
#         Return
#         ------
#         y_t: scalar
#             prediction of 
#         """
#         return 

# class BayesianPersonalizedRanking(nn.Module):
#     def __init__(self, ):

if __name__ == "__main__":
    # Test function
    n_user = 1000
    n_item = 1000
    n_factor = 10
    U = torch.tensor([i for i in range(n_user)]).cuda()
    I = torch.tensor([i for i in range(n_item)]).cuda()
    mf = MatrixFactorizer(n_user, n_item).cuda()
    t0 = gettime()
    D = mf(U, I)
    t1 = gettime()
    print("time cost:", t1 - t0)
    print(D)
