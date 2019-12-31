import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class AdjacencyMatrix(Dataset):
    r"""
    You must first define the interfaces
    You want the code to be easy to hack. So you define a simple and obvious interface about file names

    Interface
    ---------
    In: the filepath and the metapath
        filename: ${filepath}/adj_${metapath}

    Out: the adjacent matrix of a given metapath
    """
    def __init__(self, filepath, metapaths, bin=False):
        r"""
        load the pickle files, including:
        adjacency matrix, rates and matrix factorization embeddings

        Parameters
        ----------
        paths: list
            file paths

        bin: Bool.
            Decide if the file to read is binary file.
            If binary file, recommended to use pickle,
            else you can use textfile
        """
        self.filepath  = filepath
        self.metapaths = metapaths
        self.is_binary = bin
        self.data = {}
        if self.is_binary == True:
            for metapath in metapaths:
                file = self.filepath + 'adj_' + metapath
                with open(file, 'rb') as fw:
                    adjacency = pickle.load(fw)
                    self.data[metapath] = adjacency
        if self.is_binary == False:
            """ TODO: read txt file """
            raise NotImplementedError
        
    def __getitem__(self, metapath):
        r"""
        Parameters
        ----------
        metapath: str
            metapath with adjacency matrix already computed and stored into a file

        Return
        ------
        adjacency: numpy.ndarray
            the adjacency matrix of the metapath
        """
        if metapath in self.metapaths:
            return self.data[metapath]
        else:
            return None # TODO: this seems buggy

    def __len__(self):
        return len(self.data)

class MatFactFeatures(Dataset):
    r"""
    Concat all features and original rates of a user into a single vector

    Interface
    ---------



    """
    def __init__(self, ):
        super(MatFactFeatures, self).__init__()

if __name__ == "__main__":
    pass