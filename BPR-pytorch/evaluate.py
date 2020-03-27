import numpy as np
import torch
import math
from metrics import *

def metrics(model, test_loader, top_k):
    prec_list   = []
    recall_list = []
    hit_list    = []
    ndcg_list   = []
    eval_len    = []
    ranklists = []
    predictions = []
    for i, data in enumerate(test_loader):
        users, items, labels = data
        users = users.cuda()
        items = items.cuda()
        labels = labels.cuda()
        prediction_i, _ = model(users, items, labels)

        
        values, indices = torch.topk(prediction_i, top_k)
        ranklist = indices[0].tolist()

        ranklists.append(ranklist)
        predictions.append(prediction_i)
        gtItems = torch.nonzero(labels)[:, 1].tolist()
        prec_list.append(getP(ranklist, gtItems))
        recall_list.append(getR(ranklist, gtItems))
        hit_list.append(getHitRatio(ranklist, gtItems))
        ndcg_list.append(getNDCG(ranklist, gtItems))
        eval_len.append(len(gtItems))

    print("prediction of the last user:", predictions[-1])
    print("output of the last user:", ranklists[-1])
    prec = np.mean(np.asarray(prec_list))
    recall = np.mean(np.asarray(recall_list))
    hit_ratio = np.sum(hit_list)/np.sum(eval_len)
    ndcg = np.mean(np.asarray(ndcg_list))

    return prec, recall, hit_ratio, ndcg
