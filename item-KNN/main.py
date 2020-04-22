import os
import time
import gc
import argparse
import pickle
import numpy as np

# data science imports
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# metrics imports
from metrics import *

def read_pickle(path):
    ret = None
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def write_pickle(path, data):
    with open(path, 'wb') as fw:
        pickle.dump(data, fw)

def make_matrix(n_user, n_item, reviews):
    """
    row for item, column for user
    """
    user_ids = [i for i in range(n_user)]
    item_ids = [i for i in range(n_item)]
    train_mat = np.zeros((n_item, n_user))
    for i in reviews:
        user = i['user_id']
        item = i['business_id']
        train_mat[item][user] = 1
    return user_ids, item_ids, train_mat

def evaluation(test_data, train_mat, predictions):
    """
    predictions is calculated above
    """
    precs = []
    hrs = []
    recalls = []
    ndcgs = []
    
    for i in test_data:
        user = i['user_id']
        gt_items = i['pos_business_id']
#         print(gt_items)
        interactions = np.nonzero(train_mat[:,user])
#         print(interactions)
        for item in gt_items:
            try:
                assert item not in interactions[0]
            except AssertionError:
                print("user id:", user)
#                 print(gt_items)
#                 print(interactions)
                print(item)
        # predictions[interactions] is the top 10 neighbors of the item
        #-----------------------
        # step 1: select preds
        #-----------------------
        unsorted = predictions[interactions].reshape(-1, 2)
#         print("unsorted:", unsorted)
        #-----------------------
        # step 2: sort preds
        #-----------------------
        sorted_preds = unsorted[np.argsort(unsorted[:, 0])]
#         print("sorted:", sorted_preds)
        #-----------------------------------------
        # step 3: select top 10, but keep unique
        #-----------------------------------------
        pred_items = []
        idx = 0
        while(len(pred_items) < 10):
            item = int(sorted_preds[idx, 1])
            if item not in pred_items:
                pred_items.append(item)
            idx += 1
#         print("top10:", pred_items)
        
        #-----------------------------
        # step 4: Calculate metrics
        #-----------------------------
        prec = getP(pred_items, gt_items)
        hr = getHitRatio(pred_items, gt_items)
        recall = getR(pred_items, gt_items)
        ndcg = getNDCG(pred_items, gt_items)
#         print("prec: %.4f, hr: %.4f, recall: %4f, ndcg: %4f" % (prec, hr, recall, ndcg))
        precs.append(prec)
        hrs.append(hr)
        recalls.append(recall)
        ndcgs.append(ndcg)
        
    mean_prec = np.mean(precs)
    mean_hr = np.mean(hrs)
    mean_recall = np.mean(recalls)
    mean_ndcg = np.mean(ndcgs)
    
    return mean_prec, mean_hr, mean_recall, mean_ndcg

homedir = os.getenv('HOME')
datapath = os.path.realpath(os.path.join(homedir, 'datasets/yelp_dataset/rates'))
print(datapath)

train_data = read_pickle(os.path.join(datapath, 'rate_train'))
users = read_pickle(os.path.join(datapath, 'num_to_userid'))
items = read_pickle(os.path.join(datapath, 'num_to_businessid'))
test_data = read_pickle(os.path.join(datapath, 'test_with_neg'))

print(train_data[0])
print(len(test_data))
print(test_data[1])
print(len(users))
print(len(items))

test_users = set(i['user_id'] for i in test_data)

user_ids, item_ids, train_mat = make_matrix(len(users), len(items), train_data)
print(train_mat.shape)

from scipy.stats import pearsonr
def my_pearson(x, y):
    pearson, p_value = pearsonr(x, y)
    return -pearson

metric = 'pearson'
model = NearestNeighbors(10, algorithm='brute', metric=my_pearson, n_jobs=16)
model.fit(train_mat)    # the shape of train_mat need to be (n_queries, n_features), thus (n_items, n_users)
t0 = time.time()
distance, indices = model.kneighbors(train_mat, 11)
t1 = time.time()
print("time cost:", t1 - t0)

a = np.expand_dims(distance[:,1:], -1)
b = np.expand_dims(indices[:,1:], -1)
predictions = np.concatenate((a, b), axis=2)

prec, hr, recall, ndcg = evaluation(test_data, train_mat, predictions)
print("%s final: prec@10: %.4f, hr@10: %.4f, recall@10: %4f, ndcg@10: %4f" % (metric, prec, hr, recall, ndcg))