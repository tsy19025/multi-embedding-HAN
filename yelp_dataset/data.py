import json
import numpy as np
from copy import deepcopy
import pickle

def load_jsondata_from_file(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_id_to_num(json_datas, filtered_list, filtered_name, id_name, multi_value):
    num_to_id = {}
    id_to_num = {}
    tot = 0
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
        user_id = review['user_id']
        business_id = review['business_id']
        users_interact_num[user_id] = users_interact_num.get(user_id, 0) + 1
        business_interact_num[business_id] = business_interact_num.get(business_id, 0) + 1
    filtered_review_users = [u for u in users_interact_num.keys() if users_interact_num[u]>=user_threshold]
    filtered_review_businesses = [b for b in business_interact_num.keys() if business_interact_num[b]>=business_threshold]
    for user in users:
        user_id = user['user_id']
        if user_id not in filtered_review_users:
            continue
        if not user['friends']:
            continue
        filtered_friends = [friend.strip() for friend in user['friends'].split(',') if friend.strip() in filtered_review_users]
        if len(filtered_friends) >= friend_threshold:
            filtered_users[user_id] = filtered_friends
    continue_filter = True
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
    for business in businesses:
        business_id = business['business_id']
        if business_id not in filtered_review_businesses:
            continue
        if not business['categories']:
            continue
        if not business['city']:
            continue
        filtered_businesses.append(business_id)
    return filtered_users.keys(), filtered_businesses

def dataset_split(reviews, userid_to_num, businessid_to_num, train_ratio, valid_ratio, test_ratio):
    selected_reviews = []
    for review in reviews:
        if (review['user_id'] not in userid_to_num) or (review['business_id'] not in businessid_to_num):
            continue
        filtered_review = {}
        filtered_review['user'] = userid_to_num[review['user_id']]
        filtered_review['business'] = businessid_to_num[review['business_id']]
        filtered_review['rate'] = int(review['stars'])
        selected_reviews.append(filtered_review)
    n_reviews = len(selected_reviews)
    test_indices = np.random.choice(selected_reviews, size=int(n_reviews*test_ratio), replace=False)
    left = set(range(n_reviews))-set(test_indices)
    n_left = len(left)
    valid_indices = np.random.choice(list(left), size=int(n_left*valid_ratio), replace=False)
    train_indices = list(left-set(valid_indices))
    train_data = [selected_reviews[index] for index in train_indices]
    valid_data = [selected_reviews[index] for index in valid_indices]
    test_data = [selected_reviews[index] for index in test_indices]
    return train_data, valid_data, test_data

def get_adj_matrix(userid_to_num, businessid_to_num, cityid_to_num, categoryid_to_num, users, businesses, reviews):
    tot_users = len(userid_to_num)
    tot_business = len(businessid_to_num)
    tot_city = len(cityid_to_num)
    tot_category = len(categoryid_to_num)
    #relation U-U
    adj_UU = np.zeros([tot_users, tot_users])
    adj_UB = np.zeros([tot_users, tot_business])
    adj_BCa = np.zeros([tot_business, tot_category])
    adj_BCi = np.zeros([tot_business, tot_city])
    for user in users:
        if user['user_id'] not in userid_to_num:
            continue
        user_id = userid_to_num[user['user_id']]
        for friend in user['friends'].split(','):
            friend = friend.strip()
            if friend in userid_to_num:
                friend_id = userid_to_num[friend]
                adj_UU[user_id][friend_id] = 1
                adj_UU[friend_id][user_id] = 1
    #relation U-B
    for review in reviews:
        if (review['user_id'] not in userid_to_num) or (review['business_id'] not in businessid_to_num):
            continue
        user_id = userid_to_num[review['user_id']]
        business_id = businessid_to_num[review['business_id']]
        adj_UB[user_id][business_id] = 1
        adj_UB[business_id][user_id] = 1
    #relation B_Ca B_Ci
    for business in businesses:
        if business['business_id'] not in businessid_to_num:
            continue
        business_id = businessid_to_num[business['business_id']]
        city_id = cityid_to_num[business['city']]
        adj_BCi[business_id][city_id] = 1
        adj_BCi[city_id][business_id] = 1
        for category in business['categories'].split(','):
            category = category.strip()
            category_id = categoryid_to_num[category]
            adj_BCa[business_id][category_id] = 1
            adj_BCa[category_id][business_id] = 1
    #metapath
    adj_UUB = adj_UU.dot(adj_UB)
    adj_UBU = adj_UB.dot(adj_UB.T)
    adj_UBUB = adj_UBU.dot(adj_UB)
    adj_UBCa = adj_UB.dot(adj_BCa)
    adj_UBCi = adj_UB.dot(adj_BCi)
    adj_BCaB = adj_BCa.dot(adj_BCa.T)
    adj_BCiB = adj_BCi.dot(adj_BCi.T)
    return adj_UU, adj_UB, adj_BCa, adj_BCi, adj_UUB, adj_UBU, adj_UBUB, adj_UBCa, adj_UBCi, adj_BCaB, adj_BCiB

if __name__ == '__main__':
    user_json = load_jsondata_from_file('json/yelp_academic_dataset_user.json')
    business_json = load_jsondata_from_file('json/yelp_academic_dataset_business.json')
    review_json = load_jsondata_from_file('json/yelp_academic_dataset_review.json')
    filtered_user, filtered_business = filter_rare_node(user_json, business_json, review_json, 30, 100, 5)
    num_to_userid, userid_to_num = get_id_to_num(user_json, filtered_user, 'user_id', 'user_id', False)
    num_to_businessid, businessid_to_num = get_id_to_num(business_json, filtered_business, 'business_id', 'business_id', False)
    num_to_cityid, cityid_to_num = get_id_to_num(business_json, filtered_business, 'business_id', 'city', False)
    num_to_categoryid, categoryid_to_num = get_id_to_num(business_json, filtered_business, 'business_id', 'categories', True)
    print(len(userid_to_num))
    print(len(businessid_to_num))
    print(len(cityid_to_num))
    print(len(categoryid_to_num))
    r = (num_to_userid, num_to_businessid, num_to_cityid, num_to_categoryid)
    r_names = ('num_to_userid', 'num_to_businessid', 'num_to_cityid', 'num_to_categoryid')
    for i in range(len(r)):
        with open('adjs/' + r_names[i], 'wb') as f:
            pickle.dump(r[i], f, protocol=4)
    review_train, review_valid, review_test = dataset_split(review_json, userid_to_num, businessid_to_num, 0.8, 0.1, 0.2)
    adj_UU, adj_UB, adj_BCa, adj_BCi, adj_UUB, adj_UBU, adj_UBUB, adj_UBCa, adj_UBCi, adj_BCaB, adj_BCiB = \
        get_adj_matrix(userid_to_num, businessid_to_num, cityid_to_num, categoryid_to_num, user_json, business_json, review_train)
    # relation save
    t = (adj_UU, adj_UB, adj_BCa, adj_BCi, adj_UUB, adj_UBU, adj_UBUB, adj_UBCa, adj_UBCi, adj_BCaB, adj_BCiB)
    t_names = ('adj_UU', 'adj_UB', 'adj_BCa', 'adj_BCi', 'adj_UUB', 'adj_UBU', 'adj_UBUB', 'adj_UBCa', 'adj_UBCi', 'adj_BCaB', 'adj_BCiB')
    for i in range(len(t)):
        with open('adjs/' + t_names[i], 'wb') as f:
            pickle.dump(t[i], f, protocol=4)
    # train valid test data save
    d = (review_train, review_valid, review_test)
    d_names = ('rate_train', 'rate_valid', 'rate_test')
    for i in range(len(d)):
        with open('rates/' + d_names[i], 'wb') as f:
            pickle.dump(d[i], f, protocol=4)
