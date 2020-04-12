import os
import sys
import pickle
import time
import numpy as np
from copy import deepcopy

def read_dat(data_path):
    data = []
    with open(data_path, 'r', encoding = 'ISO-8859-1') as f:
        head = f.readline().split('\t')
        head[-1] = head[-1][:-1]
        tot = 1
        for line in f.readlines():
            d = {}
            tmp = line.split('\t')
            tmp[-1] = tmp[-1][:-1]
            for i in range(len(head)):
                try:
                    tmp[i] = int(tmp[i])
                    d.update({head[i] : tmp[i]})
                except: d.update({head[i] : tmp[i]})
            data.append(d)
    return data

def write_pickle(file, data):
    with open(file, 'wb') as fw:
        pickle.dump(data, fw)

def Union(movie_actor, movie_country, movie_director, movie_genre):
    movies = {}
    
    for data in movie_actor:
        movie = data['movieID']
        if movie not in movies.keys():
            movies.update({movie: {'movieID': movie, 'actor': [], 'country': [], 'director': [], 'genre': []}})
        movies[movie]['actor'].append(data['actorID'])
    
    for data in movie_country:
        movie = data['movieID']
        if movie not in movies.keys():
            movies.update({movie: {'movieID':movie, 'actor': [], 'country': [], 'director': [], 'genre': []}})
        movies[movie]['country'].append(data['country'])
    
    director2did = {}
    did2director = []
    n_director = 0
    for data in movie_director:
        movie = data['movieID']
        if movie not in movies.keys():
            movies.update({movie: {'movieID':movie, 'actor': [], 'country': [], 'director': [], 'genre': []}})
        director = data['directorID']
        movies[movie]['director'].append(director)
    
    genre2gid = {}
    gid2genre = []
    n_genre = 0
    for data in movie_genre:
        movie = data['movieID']
        if movie not in movies.keys():
            movies.update({movie: {'movieID':movie, 'actorID': [], 'country': [], 'directorID': [], 'genre': []}})
        genre = data['genre']
        movies[movie]['genre'].append(genre)
    
    return movies
    
gettime = lambda: time.time()

def filter_rare_node(movies, movie_tag, user_threshold, movie_threshold, genre_threshold):
    continue_filter = True
    filtered_user_ids = set()
    filtered_movie_ids = set()
    item_relate = {}

    while (continue_filter):
        t0 = gettime()

        continue_filter = False
        # ------------------------------------
        # filter step 1
        # rough filter
        # filter the active users and items
        # according to the description of the dataset, each user and item has at least 5 reviews
        # ------------------------------------
        
        '''
        # actor?
        actor_num = {}
        for data in movie_actor:
            for actor in data['actor']:
                actor_num[actor] = actor_playin_num.get(actor, 0) + 1
        filter_actor = set(a for a in actor_num.keys() if actor_num[a] >= actor_threshold)
        
        director_num = {}
        for data in movie_director:
            for director in data['director']:
                director_num[director] = director_num.get(director, 0) + 1
        filter_director = set(d for d in director_num.keys() if director_num[d] >= director_threshold)
        
        location_num = {}
        for data in movie_location:
            for location in data['location']:
                location_num[location] = location_num.get(location, 0) + 1
        filter_location = set(l for l in location_num.keys() if location_num[l] >= location_threshold)
        
        genre_num = {}
        for data in movie_genre:
            for genre in data['genre']
                genre_num[genre] = genre_num.get(genre, 0) + 1
        filter_genre = set(g for g in genre_num.keys() if genre_num[g] >= genre_threshold)
        
        country_num = {}
        for data in movie_country:
            for country in data['country']:
                country_num[country] = country_num.get(country, 0) + 1
        filter_country = set(c for c in country_num.keys() if country_num[c] >= country_threshold)
        '''
        user_num = {}
        movie_num = {}
        user_movie_interact = set()
        
        t1 = gettime()
        for data in movie_tag:
            user = data['userID']
            movie = data['movieID']
            tag = data['tagID']
            movie_user = user * 100000 + movie
            if movie_user not in user_movie_interact:
                user_num[user] = user_num.get(user, 0) + 1
                movie_num[movie] = movie_num.get(movie, 0) + 1
                user_movie_interact.add(movie_user)
        filtered_user = set(u for u in user_num.keys() if user_num[u] >= user_threshold)
        filtered_movie = set(m for m in movie_num.keys() if movie_num[m] >= movie_threshold)
        
        print("step 1 time cost:", gettime() - t1)
        print("len filtered_tag_users: %d, len filtered_tag_movies: %d" % (len(filtered_user), len(filtered_movie)))

        if (filtered_user_ids != filtered_user) or (filtered_movie_ids != filtered_movie):
            continue_filter = True

        # ------------------------------------
        # filter step 2
        # filter items
        # keep related items all included in one set
        # ------------------------------------
        t1 = gettime()
        '''
        genre_num = {}
        for data in movies:
            genres = data['genre']
            for genre in genres:
                genre_num[genre] = genre_num.get(genre, 0) + 1
        filter_genre = set(g for g in genre_num.keys() if genre_num[g] >= genre_threshold)
        '''
        
        movie_genre = {}
        for data in movies.values():
            movie = data['movieID']
            if movie not in filtered_movie: continue
            if len(data['genre']) >= genre_threshold:
                movie_genre[movie] = data['genre']
        filtered_movie_ids = set(movie_genre.keys())
        print("movie:", len(filtered_movie_ids))
        print("step 2 time cost:", gettime() - t1)

        # ------------------------------------
        # filter step 3
        # filter users
        # ------------------------------------
        t1 = gettime()
        filtered_user_ids = set(data['userID'] for data in movie_tag \
                                if (data['movieID'] in filtered_movie) \
                                and (data['userID'] in filtered_user))
        print("user:", len(filtered_user))
        print("step 3 time cost:", gettime() - t1)

        # ------------------------------------
        # filter step 4
        # filter reviews
        # make sure that 'reviewerID' in filtered_user_ids and 'asin' in filtered_item_asins
        # ------------------------------------
        t1 = gettime()
        filtered_tag = []
        user_movie_interact = set()
        for data in movie_tag:
            user = data['userID']
            movie = data['movieID']
            if (user in filtered_user_ids) and (movie in filtered_movie_ids):
                user_movie = str(user) + '/' + str(movie)
                if user_movie not in user_movie_interact:  # remove duplication
                    filtered_tag.append(data)
                    user_movie_interact.add(user_movie)
        movie_tag = deepcopy(filtered_tag)
        print("step 4 time cost:", gettime() - t1)

        print("user:", len(list(filtered_user_ids)))
        print("movie:", len(list(filtered_movie_ids)))
        print("tag:", len(movie_tag))
        print('time cost:', gettime() - t0)
        print('filter loop')

    filtered_movie_data = []
    for data in movies.values():
        movie_data = {}
        if data['movieID'] in filtered_movie:
            movie_data['movieID'] = data['movieID']
            movie_data['genre'] = data['genre']
            movie_data['director'] = data['director']
            movie_data['country'] = data['country']
            movie_data['actor'] = data['actor']
            filtered_movie_data.append(movie_data)

    print('filter complete')
    print(len(filtered_user))
    print(len(filtered_movie))
    print(len(movie_tag))

    for data in movie_tag:
        user = data['userID']
        item = data['movieID']
        if user not in filtered_user: print("user erroe")

    return list(filtered_user), filtered_movie_data, movie_tag

def dataset_split(original_tags, uid2ind, mid2ind, train_ratio, valid_ratio, test_ratio, n_neg_sample):
    tags = []

    for tag in original_tags:
        filtered_tag = {}
        filtered_tag['user_id'] = uid2ind[tag['userID']]
        filtered_tag['item_id'] = mid2ind[tag['movieID']]
        filtered_tag['rate'] = 1.0
        filtered_tag['timestamp'] = int(tag['timestamp'])
        tags.append(filtered_tag)

    tags_sorted = sorted(tags, key=lambda k: k['timestamp'])  # use the earlier data to train and the later data to test
    n_tags = len(tags_sorted)
    train_size = int(n_tags * train_ratio)
    valid_size = int(n_tags * valid_ratio)
    train_data = [tags_sorted[i] for i in range(train_size)]
    valid_data = [tags_sorted[i] for i in range(train_size, train_size + valid_size)]
    test_data = [tags_sorted[i] for i in range(train_size + valid_size, n_tags)]

    selected_items = set(mid2ind.values())


    data_list = [train_data, valid_data, test_data]
    data_for_user_list = [{}, {}, {}]
    all_data_for_user = {}
    for index in range(len(data_list)):
        data = data_list[index]
        data_for_user = data_for_user_list[index]
        for tag in data:
            user = tag['user_id']
            item = tag['item_id']
            if user not in data_for_user:
                data_for_user[user] = [item]
                all_data_for_user[user] = [item]
            else:
                data_for_user[user].append(item)
                all_data_for_user[user].append(item)
    train_data_for_user, valid_data_for_user, test_data_for_user = data_for_user_list  # dictionary of user_id:[item_id]

    with_neg_list = [valid_data_for_user, test_data_for_user]
    #     data_with_neg_list = [[] for _ in range(len(with_neg_list))]
    data_with_neg_list = [[], []]
    for index in range(len(with_neg_list)):
        current_data = with_neg_list[index]
        for user in current_data.keys():
            user_eval = {}  # a dict
            business_set = selected_items - set(all_data_for_user[user]) # items not existed in this user's records
            sample_items = np.random.choice(list(business_set), size=n_neg_sample, replace=False)  # sample is random.choice
            user_eval['user_id'] = user
            user_eval['pos_item_id'] = current_data[user]
            user_eval['neg_item_id'] = list(sample_items)
            data_with_neg_list[index].append(user_eval)
    valid_with_neg, test_with_neg = data_with_neg_list
    return train_data, valid_with_neg, test_with_neg

def get_adj_matrix(uid2ind, mid2ind, aid2ind, cid2ind, did2ind, gid2ind, movies, train_data):
    n_user = len(uid2ind)
    n_item = len(mid2ind)
    n_actor = len(aid2ind)
    n_country = len(cid2ind)
    n_director = len(did2ind)
    n_genre = len(gid2ind)
    
    adj_UI = np.zeros([n_user, n_item])
    adj_IA = np.zeros([n_item, n_actor])
    adj_IC = np.zeros([n_item, n_country])
    adj_ID = np.zeros([n_item, n_director])
    adj_IG = np.zeros([n_item, n_genre])
    
    for movie in movies:
        item = mid2ind[movie['movieID']]
        for actor in movie['actor']: adj_IA[item][aid2ind[actor]] = 1
        for country in movie['country']: adj_IC[item][cid2ind[country]] = 1
        for director in movie['director']:
            adj_ID[item][did2ind[director]] = 1
        for genre in movie['genre']:
            adj_IG[item][gid2ind[genre]] = 1
    
    for data in train_data:
        user = data['user_id']
        item = data['item_id']
        adj_UI[user][item] = 1
    
    return adj_UI, adj_IA, adj_IC, adj_ID, adj_IG

if __name__ == '__main__':
    dataset_path = './'
    movie_actor = read_dat('movie_actors.dat')
    movie_country = read_dat('movie_countries.dat')
    movie_director = read_dat('movie_directors.dat')
    movie_genre = read_dat('movie_genres.dat')
    movie_tag = read_dat('user_taggedmovies-timestamps.dat')
    
    movies = Union(movie_actor, movie_country, movie_director, movie_genre)
    
    print('filter rare node.')
    users, movies, tags = filter_rare_node(movies, movie_tag, 5, 5, 1)
    
    items = set(movie['movieID'] for movie in movies)
    actors = set(actor for movie in movies for actor in movie['actor'])
    countries = set(country for movie in movies for country in movie['country'])
    directors = set(director for movie in movies for director in movie['director'])
    genres = set(genre for movie in movies for genre in movie['genre'])

    uinds = [i for i in range(len(users))]
    uid2ind = {user: ind for user, ind in zip(users, uinds)}
    ind2uid = {ind: user for user, ind in zip(users, uinds)}

    minds = [i for i in range(len(movies))]
    mid2ind = {movie: ind for movie, ind in zip(items, minds)}
    ind2mid = {ind: movie for movie, ind in zip(items, minds)}

    ainds = [i for i in range(len(actors))]
    aid2ind = {actor: ind for actor, ind in zip(actors, ainds)}
    ind2aid = {ind: actor for actor, ind in zip(actors, ainds)}

    cinds = [i for i in range(len(countries))]
    cid2ind = {country: ind for country, ind in zip(countries, cinds)}
    ind2cid = {ind: country for country, ind in zip(countries, cinds)}
    
    dinds = [i for i in range(len(directors))]
    did2ind = {director: ind for director, ind in zip(directors, dinds)}
    ind2did = {ind: director for director, ind in zip(directors, dinds)}
    
    ginds = [i for i in range(len(genres))]
    gid2ind = {genre: ind for genre, ind in zip(genres, ginds)}
    ind2gid = {ind: genre for genre, ind in zip(genres, ginds)}
    
    adj_path = 'adjs/'
    if not os.path.exists(adj_path) : os.mkdir(adj_path)
    datas = [uid2ind, ind2uid, mid2ind, ind2mid, aid2ind, ind2aid, cid2ind, ind2cid, did2ind, did2ind, gid2ind, ind2gid]
    filenames = ['user_id2index', 'index2user_id', 'movie_id2index', 'index2movie_id', 'actor_id2index', 'index2actor_id',
                 'country_id2index', 'index2country_id', 'director_id2index', 'index2director_id', 'genre_id2index',
                 'index2genre_id']
    for i in range(12):
        write_pickle(dataset_path + adj_path + filenames[i] + '.pickle', datas[i])
    
    train_data, valid_with_neg_sample, test_with_neg_sample = dataset_split(tags, uid2ind, mid2ind, 0.8, 0.1, 0.1, 100)
    
    rating_path = 'rates/'
    if not os.path.exists(rating_path) : os.mkdir(rating_path)
    filenames = ['train_data', 'valid_with_neg_sample', 'test_with_neg_sample']
    objs = [train_data, valid_with_neg_sample, test_with_neg_sample]
    for file, obj in zip(filenames, objs):
        write_pickle(dataset_path + rating_path + file + '.pickle', obj)

    objs = get_adj_matrix(uid2ind, mid2ind, aid2ind, cid2ind, did2ind, gid2ind, movies, train_data)
    filenames = ['adj_UI', 'adj_IA', 'adj_IC', 'adj_ID', 'adj_IG']
    for file, obj in zip(filenames, objs):
        write_pickle(dataset_path + adj_path + file + '.pickle', obj)
