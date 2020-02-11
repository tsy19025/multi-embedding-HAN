import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def read_pickle(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def write_pickle(path, data):
    with open(path, 'wb') as fw:
        pickle.dump(data, fw);

if __name__ == "__main__":
    pass
