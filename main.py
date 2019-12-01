from utils import *
import argparse
import sys, os
from scipy import sparse

# ???
def cal(*params):
    col = np.concatenate((mat.col for mat in params), axis = 0)
    row = np.concatenate((mat.row for mat in params), axis = 0)
    data = np.concatenate((mat.data for mat in params), axis = 0)
    return sparse.coo_matrix((data, (col, row)), shape = params[0].shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type = int, default = 100)

    adj_UwR, adj_RaB, adj_UtB, adj_BcB, adj_BcateB, adj_UfU, UrateB, UfUwR, UfUrB, UrateBrateU, UtBtU, BrateUrateB, UrBrUrB, RaBaR, RwUwR, yelpdataset = read_data(parser.datapath)
    
    adj_user_review = cal(adj_UwR, UfUwR)
    adj_user_business = cal(adj_UtB, UrateB, UfUrB, UrBrUrB)
    user_embedding, review_embedding, business_embedding = matrix_factorization(adj_user_review, adj_user_business, adj_RaB)

    

