import numpy as np
from torch.utils.data import Dataset
import pickle

def transID_onehot(n_nodes, IDs):
    onehot = np.zeros([len(IDs), n_nodes], dtype=np.float32)
    onehot[range(len(IDs)), IDs] = 1
    return np.squeeze(onehot)

def sample_neg_business_for_user(user, n_businesses, negative_size, adj_UB):
    neg_businesses = []
    while True:
        if len(neg_businesses) == negative_size:
            break
        neg_business = np.random.choice(range(n_businesses), size=1)[0]
        if (adj_UB[user, neg_business] == 0) and neg_business not in neg_businesses:
            neg_businesses.append(neg_business)
    return neg_businesses

class YelpDataset(Dataset):
    def __init__(self, n_nodes_list, data_path, adj_paths, n_neigh, n_neg, mode):
        self.n_neigh = n_neigh
        self.n_neg = n_neg
        self.mode = mode
        self.n_nodes_list = n_nodes_list
        self.n_users, self.n_businesses, self.n_cities, self.n_categories = self.n_nodes_list
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        # adj_UU, adj_UB, adj_BCi, adj_BCa, adj_UUB, adj_UBU, adj_UBUB, adj_UBCi, adj_UBCa, adj_BCaB, adj_BCiB
        self.adjs = []
        for adj_path in adj_paths:
            with open(adj_path, 'rb') as f:
                self.adjs.append(pickle.load(f))

        user_user_adjs = [self.adjs[0], self.adjs[5]]
        user_business_adjs = [self.adjs[1], self.adjs[4], self.adjs[6]]
        user_city_adjs = [self.adjs[7]]
        user_category_adjs = [self.adjs[8]]
        self.user_neigh_adjs = [user_user_adjs, user_business_adjs, user_city_adjs, user_category_adjs]

        business_user_adjs = [self.adjs[1].T, self.adjs[4].T, self.adjs[6].T]
        business_business_adjs = [self.adjs[9], self.adjs[10]]
        business_city_adjs = [self.adjs[2]]
        business_category_adjs = [self.adjs[3]]
        self.business_neigh_adjs = [business_user_adjs, business_business_adjs, business_city_adjs, business_category_adjs]

    def getitem(self, user, pos_businesses, neg_businesses):
        businesses = pos_businesses + neg_businesses
        n_with_neg = len(businesses)
        # user = self.data[index]['user_id']
        # pos_business = self.data[index]['business_id']
        # neg_businesses = sample_neg_business_for_user(user, self.n_businesses, self.n_neg, self.adjs[1])
        # label = np.array([self.data[index]['rate']], dtype=np.float32)
        user_neigh_list_lists = []
        # pos_business_neigh_list_lists = []
        # neg_business_neigh_list_lists = [[] for _ in range(self.n_neg)]
        # business_neigh_list_lists = [[] for _ in range(self.n_neg+1)]
        business_neigh_list_lists = []

        # n_nodes_list = [self.n_users, self.n_businesses, self.n_cities, self.n_categories]
        for adjs_index in range(len(self.user_neigh_adjs)):
            user_neigh_list = []
            adjs = self.user_neigh_adjs[adjs_index]
            n_nodes = self.n_nodes_list[adjs_index]
            for adj in adjs:
                neighbors_index = np.nonzero(adj[user])[0]
                if len(neighbors_index) < self.n_neigh:
                    neighbors = np.random.choice(neighbors_index, size=self.n_neigh, replace=True)
                else:
                    neighbors = np.random.choice(neighbors_index, size=self.n_neigh, replace=False)
                neighbors = transID_onehot(n_nodes, neighbors)
                user_neigh_list.append(np.repeat(neighbors, n_with_neg, axis=0))
            user_neigh_list_lists.append(user_neigh_list)

        for adjs_index in range(len(self.business_neigh_adjs)):
            adjs = self.business_neigh_adjs[adjs_index]
            n_nodes = self.n_nodes_list[adjs_index]
            business_neigh_list = []
            for adj_index in range(len(adjs)):
                for business in businesses:
                    business_neigh = []
                    adj = adjs[adj_index]
                    # pos business
                    # pos_neighbors_index = np.nonzero(adj[pos_business])[0]
                    # if len(pos_neighbors_index) < self.n_neigh:
                    #     pos_neighbors = np.random.choice(pos_neighbors_index, size=self.n_neigh, replace=True)
                    # else:
                    #     pos_neighbors = np.random.choice(pos_neighbors_index, size=self.n_neigh, replace=False)
                    # pos_neighbors = transID_onehot(n_nodes, pos_neighbors)
                    # business_neigh.append(pos_neighbors)
                    # neg business
                    # for neg in range(self.n_neg):
                    neighbors_index = np.nonzero(adj[business])[0]
                    if len(neighbors_index) < self.n_neigh:
                        neg_neighbors = np.random.choice(neighbors_index, size=self.n_neigh, replace=True)
                    else:
                        neg_neighbors = np.random.choice(neighbors_index, size=self.n_neigh, replace=False)
                    neg_neighbors = transID_onehot(n_nodes, neg_neighbors)
                    business_neigh.append(neg_neighbors)
                business_neigh_list.append(np.concatenate(business_neigh, axis=0))
            business_neigh_list_lists.append(business_neigh_list)
        users = transID_onehot([user] * n_with_neg)
        businesses = transID_onehot(businesses)
        label = np.zeros([len(businesses)], dtype=np.float32)
        label[range(len(pos_businesses))] = 1.0
        # label[0] = 1.0
        return users, businesses, label, user_neigh_list_lists, business_neigh_list_lists

    # def eval_getitem(self, index):
    #     user = self.data[index]['user_id']
    #     eval_label = self.data[index]['eval_label']
    #     eval_item = self.data[index]['eval_item']
    #     user_neigh_list_lists = []
    #     business_neigh_list_lists = []
    #     n_with_neg = len(eval_item)
    #     for adjs_index in range(len(self.user_neigh_adjs)):
    #         user_neigh_list = []
    #         adjs = self.user_neigh_adjs[adjs_index]
    #         n_nodes = self.n_nodes_list[adjs_index]
    #         for adj in adjs:
    #             neighbors_index = np.nonzero(adj[user])[0]
    #             if len(neighbors_index) < self.n_neigh:
    #                 neighbors = np.random.choice(neighbors_index, size=self.n_neigh, replace=True)
    #             else:
    #                 neighbors = np.random.choice(neighbors_index, size=self.n_neigh, replace=False)
    #             neighbors = transID_onehot(n_nodes, neighbors)
    #             user_neigh_list.append(np.repeat(neighbors, n_with_neg, axis=0))
    #         user_neigh_list_lists.append(user_neigh_list)
    #
    #     return users, businesses, label

    def __getitem__(self, index):
        if self.mode == 'train':
            user = self.data[index]['user_id']
            pos_businesses = [self.data[index]['business_id']]
            neg_businesses = sample_neg_business_for_user(user, self.n_businesses, self.n_neg, self.adjs[1])
            # businesses = pos_business + neg_businesses
            return self.getitem(user, pos_businesses, neg_businesses)
        else:
            user = self.data[index]['user_id']
            pos_businesses = self.data[index]['pos_business_id']
            neg_businesses = self.data[index]['neg_business_id']
            # businesses = pos_businesses + neg_businesses
            return self.getitem(user, pos_businesses, neg_businesses)

        # for adjs_index in range(len(business_neigh_adjs)):
        #     adjs = business_neigh_adjs[adjs_index]
        #     n_nodes = n_nodes_list[adjs_index]
        #     #pos business
        #     pos_business_neigh_list = []
        #     for adj in adjs:
        #         pos_neighbors_index = np.nonzero(adj[pos_business])[0]
        #         if len(pos_neighbors_index) < self.n_neigh:
        #             pos_neighbors = np.random.choice(pos_neighbors_index, size=self.n_neigh, replace=True)
        #         else:
        #             pos_neighbors = np.random.choice(pos_neighbors_index, size=self.n_neigh, replace=False)
        #         # pos_neighbors = transID_onehot(n_nodes, pos_neighbors)
        #         pos_business_neigh_list.append(pos_neighbors)
        #     pos_business_neigh_list_lists.append(pos_business_neigh_list)
        #     #neg business
        #     for neg in range(self.n_neg):
        #         neg_business_neigh_list = []
        #         for adj in adjs:
        #             neg_neighbors_index = np.nonzero(adj[neg_businesses[neg]])[0]
        #             if len(neg_neighbors_index) < self.n_neigh:
        #                 neg_neighbors = np.random.choice(neg_neighbors_index, size=self.n_neigh, replace=True)
        #             else:
        #                 neg_neighbors = np.random.choice(neg_neighbors_index, size=self.n_neigh, replace=False)
        #             # neg_neighbors = transID_onehot(n_nodes, neg_neighbors)
        #             neg_business_neigh_list.append(neg_neighbors)
        #         neg_business_neigh_list_lists[neg].append(neg_business_neigh_list)
        # user = transID_onehot(self.n_users, [user])
        # pos_business = transID_onehot(self.n_businesses, [pos_business])
        # neg_businesses = [transID_onehot(self.n_businesses, [neg_business]) for neg_business in neg_businesses]
        # return user, pos_business, neg_businesses, label, user_neigh_list_lists, pos_business_neigh_list_lists, neg_business_neigh_list_lists

    def __len__(self):
        return len(self.data)
