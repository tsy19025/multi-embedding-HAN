import torch
from torch.utils.data import Dataset

def get_path(adj_BCa, adj_BCi, adj_UB, adj_UU):
    adj_BU = adj.UB.T()
    adj_CaB = adj.CaB.T()
    adj_CiB = adj.CiB.T()

    ub_path = []
    uub_path = []
    ubub_path = []
    ubcab_path = []
    ubcib_path = []

    for i in range(users):
        items = np.nonzero(adj_UB[i])[0]
        ub = []
        ubcab = []
        ubcib = []
        for item in items:
            ub.append(str(i)+'-'+str(item))
            cas = np.nonzero(adj_BCa[item])[0]
            cis = np.nonzero(adj_BCi[item])[0]
            for ca in cas:
                tmp_items = np.nonzero(adj_CaB[ca])[0]
                for tmp_item in tmp_items:
                    ubcab.append(str(i) + '-' + str(item) + '-' + str(ca) + '-' + str(tmp_item))
            for ci in cis:
                tmp_items = np.nonzero(adj_CiB[ci])[0]
                for tmp_item in tmp_items:
                    ubcib.append(str(i) + '-' + str(item) + '-' + str(ci) + '-' + str(tmp_item))
        ub_path.append(ub)
        ubcab_path.append(ubcab)
        ubcib_path.append(ubcib)

    for i in range(users):
        ubub = []
        for j in range(users):
            for pathi in ub_path[i]:
                for pathj in ub_path[j]:
                    ubub.append(pathi+'-'+pathj)
        ubub_path.append(ubub)

    for i in range(users):
        users = np.nonzero(adj_UU[i])[0]
        uub = []
        for j in users:
            uub.append(str(i)+'-'+pathj for pathj in ub_path[j])
        uub_path.append(uub)
    return ub_path, uub_path, ubub_path, ubcab_path, ubcib_path

def yelpDataset(Dataset):
    def __init__(self, users, items, data_path, adj_paths, neighbor_size):

