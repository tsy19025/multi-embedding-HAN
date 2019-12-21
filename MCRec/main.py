from model import MCRec
import torch
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description="Run MCRec.")
    parse.add_argument('--dataset', narge = '?', default = '', help = 'Choose a dataset.')
    parse.add_argument('--epochs', type = int, default = 30)
    parse.add_argument()

if __name__ == __main__:
    args = parse_args()
    dataset = yelpDataset()

    model = MCRec()
    
    for epoch in range(args.epochs):
        train_one_epoch()
