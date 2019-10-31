import json
import numpy as np
from numpy import array
from scipy import sparse
import torch

def load_jsondata_from_file(path):
    data = []
    with open(path, 'r') as f:
        # tot = 0
        for line in f:
            data.append(json.loads(line))
        #    tot = tot + 1
        #    if tot > 10: break
    return data

def get_feature(jsondata, not_feature_list):
    # user: ["user_id", "name", "friends", "elite"]
    # business: ["business_id", "name", "state", "hours"]
    data_features = []
    for data in jsondata:
        features = []
        for key in data:
            if key in not_feature_list: continue
            else: features.append(user[key])
        data_features.append(features)
    return data_features

def get_id_to_num(json_datas, id_name):
    num_to_id = []
    id_to_num = {}
    tot = 0
    for data in json_datas:
        data_id = data[id_name]
        if data_id not in id_to_num:
            num_to_id.append(data_id)
            id_to_num[data_id] = tot
            tot = tot +1
    return num_to_id, id_to_num

'''
def get_graph(userid_to_num, businessid_to_num, reviews, tips):
    tot_users = len(userid_to_num)
    tot_business = len(businessid_to_num)
    n = tot_users + tot_business + len(reviews)
    adj = np.zeros([n, n])
    for i in range(len(reviews)):
        review = reviews[i]
        user_id = userid_to_num[review["user_id"]]
        business_id = businessid_to_num[review["business_id"]] + tot_users
        review_id = tot_users + tot_business + i
        adj[user_id][review_id] = 1
        adj[review_id][user_id] = 1

        adj[business_id][review_id] = 3
        adj[review_id][business_id] = 3

    for tip in tips:
        user_id = tip["user_id"]
        business_id = tip["business_id"] + tot_users
        adj[user_id][business_id] = 2
        adj[business_id][user_id] = 2
    return adj
'''
def get_adj_matrix(userid_to_num, businessid_to_num, reviewid_to_num, users, businesses, reviews, tips, city_to_num):
    tot_users = len(userid_to_num)
    tot_business = len(businessid_to_num)
    tot_reviews = len(reviews)
    tot_tips = len(tips)

    # User(write)Review
    user_in_review = array([userid_to_num[r["user_id"]] for r in reviews])
    review_in_review = array([reviewid_to_num[r["review_id"]] for r in reviews])
    adj_UwR = sparse.coo_matrix((np.ones(tot_reviews),(user_in_review,user_in_review)),shape=(tot_users,tot_reviews))
'''
    # Review(about)Business
    business_in_review = torch.LongTensor([businessid_to_num[r["business_id"]] for r in reviews])
    ind = torch.cat([review_in_review.unsqueeze(0), business_in_review.unsqueeze(0)], dim=0)
    adj_RaB = torch.sparse.FloatTensor(ind, torch.ones(tot_reviews), (tot_reviews, tot_business))

    # User(tip)Business
    user_in_tip = torch.LongTensor([userid_to_num[r["user_id"]] for r in tips])
    business_in_tip = torch.LongTensor([businessid_to_num[r["business_id"]] for r in tips])
    ind = torch.cat([user_in_tip.unsqueeze(0), business_in_tip.unsqueeze(0)], dim=0)
    adj_RaB = torch.sparse.FloatTensor(ind, torch.ones(tot_tips), (tot_users, tot_business))

    # Business(city)Business
    

    # Business(same category)Business

    # User(friends)User
'''
    return adj_UwR

if __name__ == "__main__":
    user_json = load_jsondata_from_file("../yelp/user.json")
    business_json = load_jsondata_from_file("../yelp/business.json")
    review_json = load_jsondata_from_file("../yelp/review.json")
    tip_json = load_jsondata_from_file("../yelp/tip.json")
    usernum_to_id, userid_to_num = get_id_to_num(user_json, "user_id")
    businessnum_to_id, businessid_to_num = get_id_to_num(business_json, "business_id")
    reviewnum_to_id, reviewid_to_num = get_id_to_num(review_json, "review_id")
    _, city_to_num = get_id_to_num(business_json, "city")

    adj = get_adj_matrix(userid_to_num, businessid_to_num, reviewid_to_num, user_json, business_json, review_json, tip_json, city_to_num)
    print(adj)
