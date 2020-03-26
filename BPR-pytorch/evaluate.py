import numpy as np
import torch
import math


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0

def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0

def getP(ranklist, gtItems):
    r"""
    Compute precision
    Parameters
    ----------
    ranklist: list
        result generated by FMG model

    gtItems: list
        result from the dataset
    """
    p = 0
    for item in ranklist:
        if item in gtItems:
            p += 1
    return p * 1.0 / len(ranklist)

def getR(ranklist, gtItems):
    r"""
    Compute recall
    Parameters
    ----------
    ranklist: list
        result generated by FMG model

    gtItems: list
        result from the dataset
    """
    r = 0
    for item in ranklist:
        if item in gtItems:
            r += 1
    return r * 1.0 / len(gtItems)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getDCG(ranklist, gtItems):
    dcg = 0.0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in gtItems:
            dcg += 1.0 / math.log(i + 2)
    return  dcg

def getIDCG(ranklist, gtItems):
    idcg = 0.0
    i = 0
    for item in ranklist:
        if item in gtItems:
            idcg += 1.0 / math.log(i + 2)
            i += 1
    return idcg

def getNDCG(ranklist, gtItems):
    dcg = getDCG(ranklist, gtItems)
    idcg = getIDCG(ranklist, gtItems)
    if idcg == 0:
        return 0
    return dcg / idcg

def metrics(model, test_loader, top_k):
    prec_list = []
    recall_list = []
    ndcg_list = []
    for i, data in enumerate(test_loader):
        users, items, labels = data
        users = users.cuda()
        items = items.cuda()
        labels = labels.cuda()
        prediction_i, _ = model(users, items, labels)
        values, indices = torch.topk(prediction_i, top_k)
        ranklist = indices[0].tolist()
        gtItems = torch.nonzero(labels)[:, 1].tolist()
        prec_list.append(getP(ranklist, gtItems))
        recall_list.append(getR(ranklist, gtItems))
        ndcg_list.append(getNDCG(ranklist, gtItems))

    prec = np.mean(np.asarray(prec_list))
    recall = np.mean(np.asarray(recall_list))
    ndcg = np.mean(np.asarray(ndcg_list))

    return prec, recall, ndcg
