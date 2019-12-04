import json
import numpy as np
from copy import deepcopy
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

# def get_feature(jsondata, not_feature_list):
#     # user: ["user_id", "name", "friends", "elite"]
#     # business: ["business_id", "name", "state", "hours"]
#     data_features = []
#     for data in jsondata:
#         features = []
#         for key in data:
#             if key in not_feature_list: continue
#             else: features.append(user[key])
#         data_features.append(features)
#     return data_features

def get_id_to_num(json_datas, filtered_list, filtered_name, id_name, tot, multi_value):
    num_to_id = {}
    id_to_num = {}
    for data in json_datas:
        if data[filtered_name] not in filtered_list:
            continue
        if multi_value:
            data_ids = data[id_name].split(",")
        else:
            data_ids = data[id_name]
        for data_id in data_ids:
            data_id = data_id.strip()
            if data_id not in id_to_num:
                num_to_id[tot] = data_id
                id_to_num[data_id] = tot
                tot = tot +1
    return num_to_id, id_to_num

def filter_rare_node(users, businesses, reviews, user_threshold, business_threshold, friend_threshold):
    users_interact_num = {}
    business_interact_num = {}
    filtered_users = {}
    filtered_businesses = []
    for review in reviews:
        user_id = review["user_id"]
        business_id = review["business_id"]
        users_interact_num[user_id] = users_interact_num.get(user_id, 0) + 1
        business_interact_num[business_id] = business_interact_num.get(business_id, 0) + 1
    filtered_review_users = [u for u in users_interact_num.keys() if users_interact_num[u]>=user_threshold]
    filtered_review_businesses = [b for b in business_interact_num.keys() if business_interact_num[b]>=business_threshold]
    # print(len(filtered_review_users))
    # print(len(filtered_review_businesses))
    for user in users:
        user_id = user["user_id"]
        if user_id not in filtered_review_users:
            continue
        if not user["friends"]:
            continue
        # filtered_friends = []
        filtered_friends = [friend.strip() for friend in user["friends"].split(",") if friend.strip() in filtered_review_users]
        # for friend in user["friends"].split(","):
        #     friend = friend.strip()
        #     if friend in filtered_review_users:
        #         filtered_friends.append(friend)
        if len(filtered_friends) >= friend_threshold:
            filtered_users[user_id] = filtered_friends
    continue_filter = True
    # print(len(filtered_users))
    while(continue_filter):
        friends = {}
        continue_filter = False
        for user, user_friends in filtered_users.items():
            filtered_friends = [friend for friend in user_friends if friend in filtered_users]
            if len(filtered_friends) >= friend_threshold:
                friends[user] = filtered_friends
            else:
                continue_filter = True
        filtered_users = deepcopy(friends)
        # print(len(filtered_users))
    for business in businesses:
        business_id = business["business_id"]
        if business_id not in filtered_review_businesses:
            continue
        if not business["categories"]:
            continue
        if not business["city"]:
            continue
        filtered_businesses.append(business_id)
    # print(len(filtered_businesses))
    return filtered_users.keys(), filtered_businesses

# def get_graph(userid_to_num, businessid_to_num, cityid_to_num, categoryid_to_num, users, businesses, reviews):
#     tot_users = len(userid_to_num)
#     tot_business = len(businessid_to_num)
#     tot_city = len(cityid_to_num)
#     tot_category = len(categoryid_to_num)
#     n = tot_users + tot_business + tot_city + tot_category
#     adj = np.zeros([n, n])
#     for user in users:
#         if user["user_id"] not in userid_to_num:
#             continue
#         user_id = userid_to_num[user["user_id"]]
#         for friend in user["friends"].split(","):
#             friend = friend.strip()
#             if friend in userid_to_num:
#                 friend_id = userid_to_num[friend]
#                 adj[user_id][friend_id] = 1
#                 adj[friend_id][user_id] = 1
#     for review in reviews:
#         if (review["user_id"] not in userid_to_num) and (review["business_id"] not in businessid_to_num):
#             continue
#         user_id = userid_to_num[review["user_id"]]
#         business_id = businessid_to_num[review["business_id"]] + tot_users
#         adj[user_id][business_id] = 2
#         adj[business_id][user_id] = 2
#     for business in businesses:
#         if business["business_id"] not in businessid_to_num:
#             continue
#         business_id = businessid_to_num[business["business_id"]] + tot_users
#         city_id = cityid_to_num[business["city"]] + tot_users + tot_business
#         category_id = categoryid_to_num[business["categories"]] + tot_users + tot_business + tot_city
#         adj[business_id][city_id] = 3
#         adj[city_id][business_id] = 3
#         adj[business_id][category_id] = 4
#         adj[category_id][business_id] = 4
#     return adj

def dataset_split(reviews, userid_to_num, businessid_to_num, train_ratio, valid_ratio, test_ratio):
    selected_review = []
    for review in reviews:
        if (review["user_id"] not in userid_to_num) or (review["business_id"] not in businessid_to_num):
            continue
        selected_review.append(review)
    n_reviews = len(selected_review)
    test_indices = np.random.choice(selected_review, size=int(n_reviews*test_ratio), replace=False)
    left = set(range(n_reviews))-set(test_indices)
    n_left = len(left)
    valid_indices = np.random.choice(list(left), size=int(n_left*valid_ratio), replace=False)
    train_indices = list(left-set(valid_indices))
    train_data = [selected_review[index] for index in train_indices]
    valid_data = [selected_review[index] for index in valid_indices]
    test_data = [selected_review[index] for index in test_indices]
    return train_data, valid_data, test_data

def get_adj_matrix(userid_to_num, businessid_to_num, cityid_to_num, categoryid_to_num, users, businesses, reviews):
    tot_users = len(userid_to_num)
    tot_business = len(businessid_to_num)
    tot_city = len(cityid_to_num)
    tot_category = len(categoryid_to_num)
    n = tot_users + tot_business + tot_city + tot_category
    #relation
    adj_UU = np.zeros([n, n])
    adj_UB = np.zeros([n, n])
    adj_BCa = np.zeros([n, n])
    adj_BCi = np.zeros([n, n])
    #metapath relation
    # adj_UUB = np.zeros([n, n])
    # adj_UBUB = np.zeros([n, n])
    # adj_UBU = np.zeros([n, n])
    # adj_UBCa = np.zeros([n, n])
    # adj_UBCi = np.zeros([n, n])
    # adj_BCaB = np.zeros([n, n])
    # adj_BCiB = np.zeros([n, n])
    # adj_BUB = np.zeros([n, n])
    #relation U-U
    for user in users:
        if user["user_id"] not in userid_to_num:
            continue
        user_id = userid_to_num[user["user_id"]]
        for friend in user["friends"].split(","):
            friend = friend.strip()
            if friend in userid_to_num:
                friend_id = userid_to_num[friend]
                adj_UU[user_id][friend_id] = 1
                adj_UU[friend_id][user_id] = 1
    #relation U-B
    for review in reviews:
        if (review["user_id"] not in userid_to_num) or (review["business_id"] not in businessid_to_num):
            continue
        user_id = userid_to_num[review["user_id"]]
        business_id = businessid_to_num[review["business_id"]]
        adj_UB[user_id][business_id] = 1
        adj_UB[business_id][user_id] = 1
    #relation B_Ca B_Ci
    for business in businesses:
        if business["business_id"] not in businessid_to_num:
            continue
        business_id = businessid_to_num[business["business_id"]]
        city_id = cityid_to_num[business["city"]]
        adj_BCi[business_id][city_id] = 1
        adj_BCi[city_id][business_id] = 1
        for category in business["categories"].split(","):
            category = category.strip()
            category_id = categoryid_to_num[category]
            adj_BCa[business_id][category_id] = 1
            adj_BCa[category_id][business_id] = 1
    #metapath U_U_B
    adj_UUB = adj_UU.dot(adj_UB)
    adj_UBU = adj_UB.dot(adj_UB.T)
    adj_UBUB = adj_UBU.dot(adj_UB)
    adj_UBCa = adj_UB.dot(adj_BCa)
    adj_UBCi = adj_UB.dot(adj_BCi)
    adj_BCaB = adj_BCa.dot(adj_BCa.T)
    adj_BCiB = adj_BCi.dot(adj_BCi.T)
    return adj_UU, adj_UB, adj_BCa, adj_BCi, adj_UUB, adj_UBU, adj_UBUB, adj_UBCa, adj_UBCi, adj_BCaB, adj_BCiB

if __name__ == "__main__":
    user_json = load_jsondata_from_file("json/yelp_academic_dataset_user.json")
    business_json = load_jsondata_from_file("json/yelp_academic_dataset_business.json")
    review_json = load_jsondata_from_file("json/yelp_academic_dataset_review.json")
    # tip_json = load_jsondata_from_file("json/yelp_academic_dataset_tip.json")
    filtered_user, filtered_business = filter_rare_node(user_json, business_json, review_json, 30, 100, 5)
    num_to_userid, userid_to_num = get_id_to_num(user_json, filtered_user, "user_id", "user_id", 0, False)
    num_to_businessid, businessid_to_num = get_id_to_num(business_json, filtered_business, "business_id", "business_id", len(num_to_userid), False)
    num_to_cityid, cityid_to_num = get_id_to_num(business_json, filtered_business, "business_id", "city", len(num_to_userid)+len(num_to_businessid), False)
    num_to_categoryid, categoryid_to_num = get_id_to_num(business_json, filtered_business, "business_id", "categories", len(num_to_userid)+len(num_to_businessid)+len(num_to_cityid), True)
    print(len(userid_to_num))
    print(len(businessid_to_num))
    print(len(cityid_to_num))
    print(len(categoryid_to_num))
    r = (num_to_userid, num_to_businessid, num_to_cityid, num_to_categoryid)
    r_names = ('num_to_userid', 'num_to_businessid', 'num_to_cityid', 'num_to_categoryid')
    for i in range(len(r)):
        with open("adjs/" + r_names[i], 'wb') as f:
            pickle.dump(r[i], f, protocol=4)
    #to do add aspect for each review
    # adj = get_graph(userid_to_num, businessid_to_num, cityid_to_num, categoryid_to_num, user_json, business_json, review_json)

    review_train, review_valid, review_test = dataset_split(review_json, userid_to_num, businessid_to_num, 0.8, 0.1, 0.2)
    adj_UU, adj_UB, adj_BCa, adj_BCi, adj_UUB, adj_UBU, adj_UBUB, adj_UBCa, adj_UBCi, adj_BCaB, adj_BCiB = \
        get_adj_matrix(userid_to_num, businessid_to_num, cityid_to_num, categoryid_to_num, user_json, business_json, review_train)
    # relation save
    t = (adj_UU, adj_UB, adj_BCa, adj_BCi, adj_UUB, adj_UBU, adj_UBUB, adj_UBCa, adj_UBCi, adj_BCaB, adj_BCiB)
    t_names = ('adj_UU', 'adj_UB', 'adj_BCa', 'adj_BCi', 'adj_UUB', 'adj_UBU', 'adj_UBUB', 'adj_UBCa', 'adj_UBCi', 'adj_BCaB', 'adj_BCiB')
    for i in range(len(t)):
        with open("adjs/" + t_names[i], 'wb') as f:
            pickle.dump(t[i], f, protocol=4)
    # train valid test data save
    d = (review_train, review_valid, review_test)
    d_names = ('rate_train', 'rate_valid', 'rate_test')
    # num_all = len(userid_to_num) + len(businessid_to_num) + len(cityid_to_num) + len(categoryid_to_num)
    for i in range(len(d)):
        # rates = np.zeros([num_all, num_all])
        with open("rates/" + d_names[i], 'wb') as f:
            # for review in d_names[i]:
            #     user_id = userid_to_num[review["user_id"]]
            #     business_id = businessid_to_num[review["business_id"]]
            #     rate = int(review["stars"])
            #     rates[user_id, business_id] = rate
            #     rates[business_id, user_id] = rate
            # pickle.dump(rates, f, protocol=4)
            pickle.dump(d[i], f, protocol=4)
