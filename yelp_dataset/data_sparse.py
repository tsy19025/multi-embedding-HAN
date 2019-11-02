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
        if (id_name == "categories" or id_name == "friends") and isinstance(data_id, str):
            for data_iid in data_id.split(", "):
                if data_iid not in id_to_num:
                    num_to_id.append(data_iid)
                    id_to_num[data_iid] = tot
                    tot = tot +1
        else:
            #if id_name == "categories":
            #    print(type(data_id))
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
def get_adj_matrix(userid_to_num, businessid_to_num, reviewid_to_num, users, businesses, reviews, tips, city_to_num, cate_to_num):
    tot_users = len(userid_to_num)
    tot_business = len(businessid_to_num)
    tot_reviews = len(reviews)
    tot_tips = len(tips)

    # User(write)Review
    user_in_review = array([userid_to_num[r["user_id"]] for r in reviews])
    review_in_review = array([reviewid_to_num[r["review_id"]] for r in reviews])
    adj_UwR = sparse.coo_matrix((np.ones(tot_reviews),(user_in_review,review_in_review)),shape=(tot_users,tot_reviews))
    
    # Review(about)Business
    business_in_review = array([businessid_to_num[r["business_id"]] for r in reviews])
    adj_RaB = sparse.coo_matrix((np.ones(tot_reviews),(review_in_review, business_in_review)),shape=(tot_reviews,tot_business))

    # User(tip)Business
    user_in_tip = array([userid_to_num[r["user_id"]] for r in tips])
    business_in_tip = array([businessid_to_num[r["business_id"]] for r in tips])
    adj_UtB = sparse.coo_matrix((np.ones(tot_tips),(user_in_tip, business_in_tip)),shape=(tot_users,tot_business))

    # Business(city)Business
    city_for_busi = array([city_to_num[b["city"]] for b in businesses])
    B2Csparse = sparse.coo_matrix((np.ones(tot_business),(range(tot_business), city_for_busi)),shape=(tot_business,len(city_to_num)))
    adj_BcB = B2Csparse.dot(B2Csparse.transpose())

    # Business(same category)Business
    cate_for_busi = array([cate_to_num[cate] for b in businesses for cate in (b["categories"].split(", ") if isinstance(b["categories"], str) else [])])
    busi_for_cate = array([businessid_to_num[b["business_id"]] for b in businesses for cate in (b["categories"].split(", ") if isinstance(b["categories"], str) else [])])
    adj_B2C = sparse.coo_matrix((np.ones(cate_for_busi.size),(busi_for_cate, cate_for_busi)),shape=(tot_business,len(cate_to_num)))
    adj_BcateB = adj_B2C.dot(adj_B2C.transpose())

    # User(friends)User
    frineds_list = array([userid_to_num[f] for u in users for f in u["friends"].split(", ") if f in userid_to_num])
    from_list = array([userid_to_num[u["user_id"]] for u in users for f in u["friends"].split(", ") if f in userid_to_num])
    adj_UfU = sparse.coo_matrix((np.ones(from_list.size),(from_list, frineds_list)),shape=(tot_users,tot_users))
    
    return adj_UwR, adj_RaB, adj_UtB, adj_BcB, adj_BcateB, adj_UfU

if __name__ == "__main__":
    user_json = load_jsondata_from_file("../yelp/user.json")
    business_json = load_jsondata_from_file("../yelp/business.json")
    review_json = load_jsondata_from_file("../yelp/review.json")
    tip_json = load_jsondata_from_file("../yelp/tip.json")
    usernum_to_id, userid_to_num = get_id_to_num(user_json, "user_id")
    businessnum_to_id, businessid_to_num = get_id_to_num(business_json, "business_id")
    reviewnum_to_id, reviewid_to_num = get_id_to_num(review_json, "review_id")
    _, city_to_num = get_id_to_num(business_json, "city")
    _, cate_to_num = get_id_to_num(business_json, "categories")
    #print(cate_to_num)
    #print(business_json[0]["categories"])
    #print(user_json[0]["friends"])

    adj_UwR, adj_RaB, adj_UtB, adj_BcB, adj_BcateB, adj_UfU = get_adj_matrix(userid_to_num, businessid_to_num, reviewid_to_num, user_json, business_json, review_json, tip_json, city_to_num, cate_to_num)
    print("adj get!")
    
    UrateB = adj_UwR.dot(adj_RaB)
    UfUwR = adj_UfU.dot(adj_UwR)
    UfUrB = adj_UfU.dot(UrateB)
    #UrBcateB = UrateB.dot(adj_BcateB)
    UrBcityB = UrateB.dot(adj_BcB)
    UrateBrateU = UrateB.dot(UrateB.transpose())
    UtBtU = adj_UtB.dot(adj_UtB.transpose())
    BrateUrateB = UrateB.transpose().dot(UrateB)
    #RaBcateBaR = adj_RaB.dot(adj_BcateB).dot(adj_RaB.transpose())
    RaBcityBaR = adj_RaB.dot(adj_BcityB).dot(adj_RaB.transpose())
    print("metapath get!")