import torch
import torch.nn.functional as F
from torch import norm
import numpy as np
import time
from model import MFTrainer, MatrixFactorizer
from loss import MFLoss

gettime = lambda: time.time()

def train_MF(metapaths, loadpath, savepath, reg_user=5e-2, reg_item=5e-2, lr=1e-2, epoch=5000, cuda=False):
    r"""
    Parameters
    ----------
    metapaths: list
        list of metapaths

    epoch: int
        number of epochs
    """
    i = 0
    for metapath in metapaths:
        # instance the MF trainer
        trainer = MFTrainer(metapath, loadpath, savepath, epoch[i], cuda=cuda)
        trainer.train(lr=lr[i], reg_user=reg_user[i], reg_item=reg_item[i])
        i += 1

def train_FM_bpr(dataloader, epoch=50):
    pass
    
if __name__ == "__main__":
    matrix_path = '../yelp_dataset/adjs/'
    featurepath = '../yelp_dataset/mf_features/'

    # train MF
    metapaths = ['UB', 'UBUB', 'UUB']
    t0 = gettime()
    train_MF(metapaths, matrix_path, featurepath, epoch=[200000, 20000, 15000], lr=[2e-9, 5e-2, 9e-3], reg_user=[1e-2, 1e-1, 1e-2], reg_item=[1e-2, 1e-1, 1e-2], cuda=True)
    t1 = gettime()
    print("time cost: %f" % (t1 - t0))

    # train FM (BPR)
