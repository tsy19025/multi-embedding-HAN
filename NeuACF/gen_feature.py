import matlab
import numpy as np
import pickle
import csv
import numpy
import argparse
import matlab.engine

def parse_args():
    parse = argparse.ArgumentParser(description="Data")
    # parse.add_argument('--dataset', type = str, default = '/home1/wyf/Projects/gnn4rec/multi-embedding-HAN/yelp_dataset/')
    # parse.add_argument('--dataset', type = str, default = '/home1/tsy/Project/multi-embedding-HAN/tmpdataset/')
    parse.add_argument('--dataset', type = str, default = '/home1/wyf/Projects/gnn4rec/multi-embedding-HAN/yelp_dataset/adjs_100/')
    return parse.parse_args()

def modify(mat, path):
    np.where(np.isnan(mat), 0, mat)
    with open(path + '.all', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for i in mat:
            writer.writerow(i)
    
    mat_modify = mat
    mean = np.mean(mat)
    np.where(mat_modify < mean, 0, mat_modify)
    with open(path + '.mean', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for i in mat_modify:
            writer.writerow(i)

    mat_modify = mat
    median = np.median(mat)
    np.where(mat_modify < median, 0, mat_modify)
    with open(path + '.median', 'w') as csv_file:
        writer2 = csv.writer(csv_file)
        for i in mat_modify:
            writer2.writerow(i)
    print('Finish' + path)

if __name__ == "__main__":
    eng = matlab.engine.start_matlab()
    args = parse_args()
    adjs_path = args.dataset
    with open(adjs_path + 'adj_UB', 'rb') as f: adj_UB = pickle.load(f)
    with open(adjs_path + 'adj_UUB', 'rb') as f: adj_UUB = pickle.load(f)
    # with open(adjs_path + 'adj_UBUB', 'rb') as f: adj_UBUB = pickle.load(f)
    with open(adjs_path + 'adj_BCa', 'rb') as f: adj_BCa = pickle.load(f)
    with open(adjs_path + 'adj_BCi', 'rb') as f: adj_BCi = pickle.load(f)
    with open(adjs_path + 'adj_UBCi', 'rb') as f: adj_UBCi = pickle.load(f)
    with open(adjs_path + 'adj_UBCa', 'rb') as f: adj_UBCa = pickle.load(f)
    print("read end.")
    # all_adjs = [adj_UB, adj_UUB, adj_BCa, adj_BCi, adj_UBCi, adj_UBCa]
    # adjs = []
    # for adj in all_adjs:
    #    adjs.append(matlab.double(adj.tolist()))
    # adjs = [numpy.array(adj_UB), numpy.array(adj_UUB), adj_UBUB, adj_UBCaB]
    
    # adjs = [matlab.double(adj_UB.tolist())]
    # simMatUB = eng.PathSim(adjs, matlab.double([1, -1]))
    # modify(simMatUB, 'U.UBU.pathsim.feature')

    # simMatBU = eng.PathSim(adjs, matlab.double([-1, 1]))
    # modify(simMatBU, 'B.BUB.pathsim.feature')

    adjs = [matlab.double(adj_BCa.tolist())]
    modify(eng.PathSim(adjs, matlab.double([1, -1])), 'B.BCaB.pathsim.feature')

    adjs = [matlab.double(adj_BCi.tolist())]
    modify(eng.PathSim(adjs, matlab.double([1, -1])), 'B.BCiB.pathsim.feature')

    adjs = [matlab.double(adj_UBCi.tolist())]
    modify(eng.PathSim(adjs, matlab.double([1, -1])), 'U.UBCiBU.pathsim.feature')

    adjs = [matlab.double(adj_UBCa.tolist())]
    modify(eng.PathSim(adjs, matlab.double([1, -1])), 'U.UBCaBU.pathsim.feature')


