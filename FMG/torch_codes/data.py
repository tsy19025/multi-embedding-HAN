import os
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset

def make_dataset(data, cuda=False):
    r"""
    Parameters
    ----------
    data: labels with user id and item id, type: list

    Return
    ------
    dataset: torch.utils.data.Dataset
    """
    device = torch.device('cuda:0' if cuda else 'cpu')
    x_tensor = torch.as_tensor(data[:, 0:2], dtype=torch.long, device=device)   # avoid copying memory
    y_tensor = torch.as_tensor(data[:, 2], dtype=torch.long, device=device)
    x_tensor.requires_grad_(False)
    y_tensor.requires_grad_(False)
    dataset = TensorDataset(x_tensor, y_tensor)
    return dataset

if __name__ == "__main__":
    pass