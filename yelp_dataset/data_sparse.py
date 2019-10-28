import json
import numpy as np

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

if __name__ == "__main__":
    user_json = load_jsondata_from_file("yelp_academic_dataset_user.json")
    business_json = load_jsondata_from_file("yelp_academic_dataset_business.json")
    review_json = load_jsondata_from_file("yelp_academic_dataset_review.json")
    tip_json = load_jsondata_from_file("yelp_academic_dataset_tip.json")
    userid_to_num, _ = get_id_to_num(user_json, "user_id")
    businessid_to_num, _ = get_id_to_num(business_json, "business_id")

    adj = get_graph(userid_to_num, businessid_to_num, review_json, tip_json)
    print(adj)
