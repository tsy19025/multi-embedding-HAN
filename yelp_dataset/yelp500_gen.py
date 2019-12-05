import json
import numpy as np
from numpy import array
from scipy import sparse
import torch
import pickle

def load_jsondata_from_file(path):
    data = []
    with open(path, 'r') as f:
        # tot = 0
        for line in f:
            data.append(json.loads(line))
        #    tot = tot + 1
        #    if tot > 10: break
    return data

def save_jsondata_to_file(path, data):
    with open(path, 'w') as f:
        for line in data:
            f.write(json.dumps(line)+"\n")

r_data_scale = 500000

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

# As we limit the scale of review dataset, the USERID, REVIEWID and BUSINESSID should be subsets.
def get_adj_matrix(userid_to_num, businessid_to_num, reviewid_to_num, users, businesses, reviews, tips, city_to_num, cate_to_num):
    tot_users = len(userid_to_num)
    tot_business = len(businessid_to_num)
    tot_reviews = len(reviews)
    #tot_tips = len(tips)

    # User(write)Review
    user_in_review = array([userid_to_num[r["user_id"]] for r in reviews])
    review_in_review = array([reviewid_to_num[r["review_id"]] for r in reviews])
    adj_UwR = sparse.coo_matrix((np.ones(tot_reviews),(user_in_review,review_in_review)),shape=(tot_users,tot_reviews))
    
    # Review(about)Business
    business_in_review = array([businessid_to_num[r["business_id"]] for r in reviews])
    adj_RaB = sparse.coo_matrix((np.ones(tot_reviews),(review_in_review, business_in_review)),shape=(tot_reviews,tot_business))

    # User(tip)Business
    user_in_tip = array([userid_to_num[r["user_id"]] for r in tips if (r["user_id"] in userid_to_num and r["business_id"] in businessid_to_num)])
    business_in_tip = array([businessid_to_num[r["business_id"]] for r in tips  if (r["user_id"] in userid_to_num and r["business_id"] in businessid_to_num)])
    adj_UtB = sparse.coo_matrix((np.ones(len(user_in_tip)),(user_in_tip, business_in_tip)),shape=(tot_users,tot_business))

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
    user_json2 = load_jsondata_from_file("../yelp/user.json")
    business_json2 = load_jsondata_from_file("../yelp/business.json")
    review_json2 = load_jsondata_from_file("../yelp/review.json")
    tip_json2 = load_jsondata_from_file("../yelp/tip.json")
    
    review_json = review_json2[0:r_data_scale]
    reviewnum_to_id, reviewid_to_num = get_id_to_num(review_json, "review_id")
    _, business2id_to_num = get_id_to_num(business_json2, "business_id")
    _, user2id_to_num = get_id_to_num(user_json2, "user_id")
    user_filter = set([user2id_to_num[r["user_id"]] for r in review_json])
    business_filter = set([business2id_to_num[r["business_id"]] for r in review_json])
    #user_json = [u for u in user_json2 if u["user_id"] in user_filter]
    user_json = [user_json2[i] for i in user_filter]
    print('A')
    business_json = [business_json2[i] for i in business_filter]
    print('B')
    #business_json = [b for b in business_json2 if b["business_id"] in business_filter]
    tip_json = [r for r in tip_json2]
    
    save_jsondata_to_file("../yelp/user-500k.json", user_json)
    save_jsondata_to_file("../yelp/business-500k.json", business_json)
    save_jsondata_to_file("../yelp/review-500k.json", review_json)
    save_jsondata_to_file("../yelp/tip-500k.json", tip_json)
    del user_filter, business_filter
    
    businessnum_to_id, businessid_to_num = get_id_to_num(business_json, "business_id")
    usernum_to_id, userid_to_num = get_id_to_num(user_json, "user_id")
    _, city_to_num = get_id_to_num(business_json, "city")
    _, cate_to_num = get_id_to_num(business_json, "categories")
    print(len(reviewid_to_num), len(businessid_to_num), len(userid_to_num), len(city_to_num), len(cate_to_num))
    
    adj_UwR, adj_RaB, adj_UtB, adj_BcB, adj_BcateB, adj_UfU = get_adj_matrix(userid_to_num, businessid_to_num, reviewid_to_num, user_json, business_json, review_json, tip_json, city_to_num, cate_to_num)
    print("adj get!")
    
    t = (adj_UwR, adj_RaB, adj_UtB, adj_BcB, adj_BcateB, adj_UfU)
    t_names = ('adj_UwR', 'adj_RaB', 'adj_UtB', 'adj_BcB', 'adj_BcateB', 'adj_UfU')
    for i in range(len(t)):
        with open("../yelp/adjs/"+t_names[i], 'wb') as f:
            pickle.dump(t[i], f, protocol = 4)
        
    UrateB = adj_UwR.dot(adj_RaB)
    UfUwR = adj_UfU.dot(adj_UwR)
    UfUrB = adj_UfU.dot(UrateB)
    UrBcateB = UrateB.dot(adj_BcateB)
    UrBcityB = UrateB.dot(adj_BcB)
    UrateBrateU = UrateB.dot(UrateB.transpose())
    RaBaR = adj_RaB.dot(adj_RaB.transpose())
    RwUwR = adj_UwR.transpose().dot(adj_UwR)
    print("metapath get!")
    del adj_UwR, adj_RaB, adj_UtB, adj_BcB, adj_BcateB, adj_UfU
    '''
    t = (UrateB, UfUwR, UfUrB, UrBcateB, UrBcityB, UrateBrateU, RaBaR, RwUwR)
    t_names = ('UrateB', 'UfUwR', 'UfUrB', 'UrBcateB', 'UrBcityB', 'UrateBrateU', 'RaBaR', 'RwUwR')
    for i in range(len(t)):
        with open("../yelp/adjs/"+t_names[i], 'wb') as f:
            pickle.dump(t[i], f, protocol = 4)
            del t[i]
    '''
    with open("../yelp/adjs/UrateB", 'wb') as f:
        pickle.dump(UrateB,f, protocol = 4)
    del UrateB
    with open("../yelp/adjs/UfUwR", 'wb') as f:
        pickle.dump(UfUwR,f, protocol = 4)
    del UfUwR
    with open("../yelp/adjs/UfUrB", 'wb') as f:
        pickle.dump(UfUrB,f, protocol = 4)
    del UfUrB
    with open("../yelp/adjs/UrBcateB", 'wb') as f:
        pickle.dump(UrBcateB,f, protocol = 4)
    del UrBcateB
    with open("../yelp/adjs/UrBcityB", 'wb') as f:
        pickle.dump(UrBcityB,f, protocol = 4)
    del UrBcityB
    with open("../yelp/adjs/UrateBrateU", 'wb') as f:
        pickle.dump(UrateBrateU,f, protocol = 4)
    del UrateBrateU
    with open("../yelp/adjs/RaBaR", 'wb') as f:
        pickle.dump(RaBaR,f, protocol = 4)
    del RaBaR
    with open("../yelp/adjs/RwUwR", 'wb') as f:
        pickle.dump(RwUwR,f, protocol = 4)
    del RwUwR
    