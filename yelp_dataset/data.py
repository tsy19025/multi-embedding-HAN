import json
import numpy as np
from copy import deepcopy
import pickle
import time, datetime

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
            data_ids = [data_id.strip() for data_id in data[id_name].split(',')]
        else:
            data_ids = [data[id_name]]
        for data_id in data_ids:
            if data_id not in id_to_num:
                num_to_id[tot] = data_id
                id_to_num[data_id] = tot
                tot = tot +1
    return num_to_id, id_to_num

def filter_rare_node(users, businesses, reviews, user_threshold, business_threshold, friend_threshold):
    continue_filter = True
    filtered_users = set()
    filtered_businesses = set()
    while(continue_filter):
        continue_filter = False
        # filter step 1
        # users_posinteract_num = {}
        # business_posinteract_num = {}
        # users_neginteract_num = {}
        # business_neginteract_num = {}
        user_interact_num = {}
        business_interact_num = {}
        user_business_interact = set()
        for review in reviews:
            if not review['date']:
                continue
            user_id = review['user_id']
            business_id = review['business_id']
            user_business = str(user_id)+str(business_id)
            if user_business not in user_business_interact:
                user_interact_num[user_id] = user_interact_num.get(user_id, 0) + 1
                business_interact_num[business_id] = business_interact_num.get(business_id, 0) + 1
                user_business_interact.add(user_business)
            # if review['stars'] > 3:
            #     users_posinteract_num[user_id] = users_posinteract_num.get(user_id, 0) + 1
            #     business_posinteract_num[business_id] = business_posinteract_num.get(business_id, 0) + 1
            # else:
            #     users_neginteract_num[user_id] = users_neginteract_num.get(user_id, 0) + 1
            #     business_neginteract_num[business_id] = business_neginteract_num.get(business_id, 0) + 1
        # user_interact = set(users_posinteract_num.keys()).intersection(set(users_neginteract_num.keys()))
        # business_interact = set(business_posinteract_num.keys()).intersection(set(business_neginteract_num.keys()))
        # filtered_review_users = set(u for u in user_interact if ((users_posinteract_num[u]+users_neginteract_num[u])>=user_threshold
        #                                                     and (users_posinteract_num[u]*users_posinteract_num[u])>0))
        # filtered_review_businesses = set(b for b in business_interact if ((business_posinteract_num[b]+business_neginteract_num[b])>=business_threshold
        #                                                     and (business_posinteract_num[b]*business_neginteract_num[b])>0))
        # filtered_review_users = set(u for u in user_interact if (users_posinteract_num[u]>=user_threshold and users_neginteract_num[u])>=user_threshold)
        # filtered_review_businesses = set(b for b in business_interact if (business_posinteract_num[b]>=business_threshold and business_neginteract_num[b])>=business_threshold)
        filtered_review_users = set(u for u in user_interact_num.keys() if user_interact_num[u]>=user_threshold)
        filtered_review_businesses = set(b for b in business_interact_num.keys() if business_interact_num[b]>=business_threshold)
        if (filtered_users != filtered_review_users) or (filtered_businesses != filtered_review_businesses):
            continue_filter = True
        # filter step 2
        #filter user and business
        user_friends_dict = {}
        for user in users:
            user_id = user['user_id']
            if user_id not in filtered_review_users:
                continue
            if not user['friends']:
                continue
            filtered_friends = [friend.strip() for friend in user['friends'].split(',') if friend.strip() in filtered_review_users]
            if len(filtered_friends) >= friend_threshold:
                user_friends_dict[user_id] = filtered_friends
        continue_inside = True
        while (continue_inside):
            friends = {}
            continue_inside = False
            for user, user_friends in user_friends_dict.items():
                filtered_friends = [friend for friend in user_friends if friend in user_friends_dict]
                if len(filtered_friends) >= friend_threshold:
                    friends[user] = filtered_friends
                else:
                    continue_inside = True
            user_friends_dict = deepcopy(friends)
        filtered_users = set(user_friends_dict.keys())
        filtered_businesses_list = []
        for business in businesses:
            business_id = business['business_id']
            if business_id not in filtered_review_businesses:
                continue
            if not business['categories']:
                continue
            if not business['city']:
                continue
            filtered_businesses_list.append(business_id)
        filtered_businesses = set(filtered_businesses_list)
        filtered_review = []
        user_business_interact = set()
        for review in reviews:
            if not review['date']:
                continue
            if (review['user_id'] in filtered_users) and (review['business_id'] in filtered_businesses):
                user_id = review['user_id']
                business_id = review['business_id']
                user_business = str(user_id) + str(business_id)
                if user_business not in user_business_interact:
                    filtered_review.append(review)
                    user_business_interact.add(user_business)
        reviews = deepcopy(filtered_review)
        print(len(list(filtered_users)))
        print(len(list(filtered_businesses)))
        print(len(reviews))
        print('filter loop')
    print('filter complete')
    return filtered_users, filtered_businesses, filtered_review

def dataset_split(reviews, userid_to_num, businessid_to_num, train_ratio, valid_ratio, test_ratio, n_neg_sample):
    selected_reviews = []
    for review in reviews:
        filtered_review = {}
        filtered_review['user_id'] = userid_to_num[review['user_id']]
        filtered_review['business_id'] = businessid_to_num[review['business_id']]
        filtered_review['rate'] = 1.0
        filtered_review['timestamp'] = time.mktime(datetime.datetime.strptime(review['date'], '%Y-%m-%d %H:%M:%S').timetuple())
        selected_reviews.append(filtered_review)
    selected_reviews_sorted = sorted(selected_reviews, key=lambda k: k['timestamp'])
    n_reviews = len(selected_reviews_sorted)
    train_size = int(n_reviews*train_ratio)
    valid_size = int(n_reviews*valid_ratio)
    train_data = [selected_reviews_sorted[index] for index in range(train_size)]
    valid_data = [selected_reviews_sorted[index] for index in range(train_size, train_size+valid_size)]
    test_data = [selected_reviews_sorted[index] for index in range(train_size+valid_size, n_reviews)]
    selected_users = set()
    selected_businesses = set()
    for review in train_data:
        selected_users.add(review['user_id'])
        selected_businesses.add(review['business_id'])
    eval_datas = [valid_data, test_data]
    selected_eval_datas = [[] for _ in range(len(eval_datas))]
    for eval_index in range(len(eval_datas)):
        eval_data = eval_datas[eval_index]
        for review in eval_data:
            if review['user_id'] in selected_users and review['business_id'] in selected_businesses:
                selected_eval_datas[eval_index].append(review)
    selected_valid_data, selected_test_data = selected_eval_datas
    data_list = [train_data, selected_valid_data, selected_test_data]
    data_for_user_list = [{} for _ in range(len(data_list))]
    train_data_for_item = set()
    for index in range(len(data_list)):
        data = data_list[index]
        data_for_user = data_for_user_list[index]
        for review in data:
            user = review['user_id']
            item = review['business_id']
            if index == 0:
                train_data_for_item.add(item)
            if user not in data_for_user:
                data_for_user[user] = [item]
            else:
                data_for_user[user].append(item)
    train_data_for_user, valid_data_for_user, test_data_for_user = data_for_user_list
    with_neg_list = [valid_data_for_user, test_data_for_user]

    # valid_with_neg = [{} for _ in range(len(valid_data_for_user))]
    # test_with_neg = [{} for _ in range(len(test_data_for_user))]
    data_with_neg_list = [[] for _ in range(len(with_neg_list))]
    # data_with_neg_list = [{} for _ in range(len(with_neg_list))]
    for index in range(len(with_neg_list)):
        current_data = with_neg_list[index]
        for user in current_data.keys():
            if user not in selected_users:
                continue
            user_eval = {}
            business_set = selected_businesses - set(train_data_for_user[user]) - set(current_data[user])
            sample_businesses = np.random.choice(list(business_set), size=n_neg_sample, replace=False)
            user_eval['user_id'] = user
            user_eval['pos_business_id'] = current_data[user]
            user_eval['neg_business_id'] = list(sample_businesses)
            data_with_neg_list[index].append(user_eval)
            # data_with_neg_list[index][user] = list(current_data[user])
            # data_with_neg_list[index][user].extend(list(sample_item))
    valid_with_neg, test_with_neg = data_with_neg_list
    # test_indices = np.random.choice(range(n_reviews), size=int(n_reviews*test_ratio), replace=False)
    # left = set(range(n_reviews))-set(test_indices)
    # n_left = len(left)
    # valid_indices = np.random.choice(list(left), size=int(n_left*valid_ratio), replace=False)
    # train_indices = list(left-set(valid_indices))
    # train_data = [selected_reviews[index] for index in train_indices]
    # valid_data = [selected_reviews[index] for index in valid_indices]
    # test_data = [selected_reviews[index] for index in test_indices]
    # return train_data, valid_data, test_data, train_data_for_user, valid_data_for_user, test_data_for_user, valid_with_neg, test_with_neg
    return train_data, selected_valid_data, selected_test_data, valid_with_neg, test_with_neg

def get_adj_matrix(userid_to_num, businessid_to_num, cityid_to_num, categoryid_to_num, users, businesses, reviews):
    tot_users = len(userid_to_num)
    tot_business = len(businessid_to_num)
    tot_city = len(cityid_to_num)
    tot_category = len(categoryid_to_num)
    #relation U-U
    adj_UU = np.zeros([tot_users, tot_users])
    adj_UB = np.zeros([tot_users, tot_business])
    # adj_UB_pos = np.zeros([tot_users, tot_business])
    # adj_UB_neg = np.zeros([tot_users, tot_business])
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
        # if (review['user_id'] not in userid_to_num) or (review['business_id'] not in businessid_to_num):
        #     continue
        # user_id = userid_to_num[review['user_id']]
        # business_id = businessid_to_num[review['business_id']]
        user_id = review['user_id']
        business_id = review['business_id']
        adj_UB[user_id][business_id] = 1
        # if review['rate'] > 0:
        #     adj_UB_pos[user_id][business_id] = 1
        # else:
        #     adj_UB_neg[user_id][business_id] = 1
    # print('pos')
    # for i in range(tot_users):
    #     if sum(adj_UB_pos[i,:])==0:
    #         print(i)
    # print('neg')
    # for i in range(tot_users):
    #     if sum(adj_UB_neg[i,:])==0:
    #         print(i)
    # print('split users')
    # for i in range(tot_users):
    #     if sum(adj_UB[i,:]) == 0:
    #         print(i)
    # print('split businesses')
    # for j in range(tot_business):
    #     if sum(adj_UB[:,j]) == 0:
    #         print(j)

    #relation B_Ca B_Ci
    for business in businesses:
        if business['business_id'] not in businessid_to_num:
            continue
        business_id = businessid_to_num[business['business_id']]
        city_id = cityid_to_num[business['city']]
        adj_BCi[business_id][city_id] = 1
        for category in business['categories'].split(','):
            category = category.strip()
            category_id = categoryid_to_num[category]
            adj_BCa[business_id][category_id] = 1
    #metapath
    # adj_UUB_pos = adj_UU.dot(adj_UB_pos)
    # adj_UUB_neg = adj_UU.dot(adj_UB_neg)
    # adj_UB_pos_U = adj_UB_pos.dot(adj_UB_pos.T)
    # adj_UB_neg_U = adj_UB_neg.dot(adj_UB_neg.T)
    # adj_UB_pos_UB_pos = adj_UB_pos_U.dot(adj_UB_pos)
    # adj_UB_neg_UB_neg = adj_UB_neg_U.dot(adj_UB_neg)
    # adj_UB_pos_Ca = adj_UB_pos.dot(adj_BCa)
    # adj_UB_neg_Ca = adj_UB_neg.dot(adj_BCa)
    # adj_UB_pos_Ci = adj_UB_pos.dot(adj_BCi)
    # adj_UB_neg_Ci = adj_UB_neg.dot(adj_BCi)
    adj_UUB = adj_UU.dot(adj_UB)
    adj_UBU = adj_UB.dot(adj_UB.T)
    adj_UBUB = adj_UBU.dot(adj_UB)
    adj_UBCa = adj_UB.dot(adj_BCa)
    adj_UBCi = adj_UB.dot(adj_BCi)
    adj_BCaB = adj_BCa.dot(adj_BCa.T)
    adj_BCiB = adj_BCi.dot(adj_BCi.T)
    return adj_UU, adj_UB, adj_BCi, adj_BCa, adj_UUB, adj_UBU, adj_UBUB, adj_UBCi, adj_UBCa, adj_BCaB, adj_BCiB
    # return adj_UU, adj_UB_pos, adj_UB_neg, adj_BCa, adj_BCi, adj_UUB_pos, adj_UUB_neg, adj_UB_pos_U, adj_UB_neg_U, \
    #     adj_UB_pos_UB_pos, adj_UB_neg_UB_neg, adj_UB_pos_Ca, adj_UB_neg_Ca, adj_UB_pos_Ci, adj_UB_neg_Ci, adj_BCaB, adj_BCiB

if __name__ == '__main__':
    user_json = load_jsondata_from_file('json/yelp_academic_dataset_user.json')
    business_json = load_jsondata_from_file('json/yelp_academic_dataset_business.json')
    review_json = load_jsondata_from_file('json/yelp_academic_dataset_review.json')
    filtered_user, filtered_business, filtered_reviews = filter_rare_node(user_json, business_json, review_json, 40, 40, 5)
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
    # review_train, review_valid, review_test, train_data_for_user, valid_data_for_user, test_data_for_user, valid_data_with_neg, test_data_with_neg = \
    #     dataset_split(filtered_reviews, userid_to_num, businessid_to_num, 0.8, 0.1, 0.1, 50)

    # adj_UU, adj_UB_pos, adj_UB_neg, adj_BCa, adj_BCi, adj_UUB_pos, adj_UUB_neg, adj_UB_pos_U, adj_UB_neg_U, \
    # adj_UB_pos_UB_pos, adj_UB_neg_UB_neg, adj_UB_pos_Ca, adj_UB_neg_Ca, adj_UB_pos_Ci, adj_UB_neg_Ci, adj_BCaB, adj_BCiB = \
    #     get_adj_matrix(userid_to_num, businessid_to_num, cityid_to_num, categoryid_to_num, user_json, business_json, review_train)
    review_train, review_valid, review_test, valid_data_with_neg, test_data_with_neg = dataset_split(filtered_reviews, userid_to_num, businessid_to_num, 0.8, 0.1, 0.1, 50)
    adj_UU, adj_UB, adj_BCi, adj_BCa, adj_UUB, adj_UBU, adj_UBUB, adj_UBCi, adj_UBCa, adj_BCaB, adj_BCiB = \
        get_adj_matrix(userid_to_num, businessid_to_num, cityid_to_num, categoryid_to_num, user_json, business_json, review_train)
    # relation save
    # t = (adj_UU, adj_UB_pos, adj_UB_neg, adj_BCa, adj_BCi, adj_UUB_pos, adj_UUB_neg, adj_UB_pos_U, adj_UB_neg_U, \
    # adj_UB_pos_UB_pos, adj_UB_neg_UB_neg, adj_UB_pos_Ca, adj_UB_neg_Ca, adj_UB_pos_Ci, adj_UB_neg_Ci, adj_BCaB, adj_BCiB)
    # t_names = ('adj_UU', 'adj_UB_pos', 'adj_UB_neg', 'adj_BCa', 'adj_BCi', 'adj_UUB_pos', 'adj_UUB_neg', 'adj_UB_pos_U', 'adj_UB_neg_U', \
    # 'adj_UB_pos_UB_pos', 'adj_UB_neg_UB_neg', 'adj_UB_pos_Ca', 'adj_UB_neg_Ca', 'adj_UB_pos_Ci', 'adj_UB_neg_Ci', 'adj_BCaB', 'adj_BCiB')
    t = (adj_UU, adj_UB, adj_BCi, adj_BCa, adj_UUB, adj_UBU, adj_UBUB, adj_UBCi, adj_UBCa, adj_BCaB, adj_BCiB)
    t_names = ('adj_UU', 'adj_UB', 'adj_BCi', 'adj_BCa', 'adj_UUB', 'adj_UBU', 'adj_UBUB', 'adj_UBCi', 'adj_UBCa', 'adj_BCaB', 'adj_BCiB')
    for i in range(len(t)):
        with open('adjs/' + t_names[i], 'wb') as f:
            pickle.dump(t[i], f, protocol=4)
    # train valid test data save
    # d = (review_train, review_valid, review_test, train_data_for_user, valid_data_for_user, test_data_for_user, valid_data_with_neg, test_data_with_neg)
    # d_names = ('rate_train', 'rate_valid', 'rate_test', 'train_user', 'valid_user', 'test_user', 'valid_with_neg', 'test_with_neg')
    d = (review_train, review_valid, review_test, valid_data_with_neg, test_data_with_neg)
    d_names = ('rate_train', 'rate_valid', 'rate_test', 'valid_with_neg', 'test_with_neg')
    for i in range(len(d)):
        with open('rates/' + d_names[i], 'wb') as f:
            pickle.dump(d[i], f, protocol=4)
