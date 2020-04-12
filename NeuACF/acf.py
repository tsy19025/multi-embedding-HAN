# coding: utf-8
from docopt import docopt
import numpy as np
import tensorflow as tf
import time
import sys
from datetime import datetime
import pickle
import math
import heapq
import pandas as pd
from scipy.io import loadmat
from tensorflow.contrib.data import Dataset, Iterator
from sklearn.utils import shuffle
import argparse
import math
import torch
import os

def getP(ranklist, gtItems):
    p = 0
    for item in ranklist:
        if item in gtItems:
            p += 1
    return p * 1.0 / len(ranklist)

def getR(ranklist, gtItems):
    r = 0
    if len(gtItems) == 0: return 1.0
    for item in ranklist:
        if item in gtItems:
            r += 1
    return r * 1.0 / len(gtItems)

def getDCG(ranklist, gtItems):
    dcg = 0.0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in gtItems:
            dcg += 1.0 / math.log(i + 2)
    return dcg

def getIDCG(ranklist, gtItems):
    idcg = 0.0
    i = 0
    for item in ranklist:
        if item in gtItems:
            idcg += 1.0 / math.log(i + 2)
            i += 1
    return idcg

def getNDCG(ranklist, gtItems):
    dcg = getDCG(ranklist, gtItems)
    idcg = getIDCG(ranklist, gtItems)
    if idcg == 0:
        return 0
    return dcg / idcg

def parse_args():
    parse = argparse.ArgumentParser(description="Run NeuACF")
    parse.add_argument('--data_path', type = str, default = '/home1/tsy/Project/multi-embedding-HAN/tmpdataset/')
    # parse.add_argument('--data_path', type = str, default = '/home1/wyf/Projects/gnn4rec/multi-embedding-HAN/yelp_dataset/')
    parse.add_argument('--mat_path', type = str, default = '/home1/tsy/Project/multi-embedding-HAN/NeuACF/mat_tmp/')
    # parse.add_argument('--mat', type = list, default = ['U.UBU', 'B.BUB', 'U.UBCiBU', 'B.BCiB', 'U.UBCaBU', 'B.BCaB'])
    parse.add_argument('--mat', type = list, default = ['U.UBU', 'B.BUB'])
    parse.add_argument('--epochs', type = int, default = 10000)
    parse.add_argument('--last_layer_size', type = int, default = 64)
    parse.add_argument('--num_of_layers', type = int, default = 2)
    parse.add_argument('--num_of_neg', type = int, default = 2)
    parse.add_argument('--learn_rate', type = float, default = 0.00005)
    parse.add_argument('--batch_size', type = int, default = 64)
    parse.add_argument('--mat_select', type = str, default = 'median')
    parse.add_argument('--merge', type = str, default = 'attention')
    parse.add_argument('--K', type = int, default = 10)
    parse.add_argument('--patience', type = int, default = 15)
    return parse.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

args = parse_args()   
data_path = args.data_path
K = args.K

mat_list = args.mat

mat_select = args.mat_select
merge = args.merge

epochs = args.epochs
last_layer_size = args.last_layer_size

num_of_layers = args.num_of_layers
num_negs = args.num_of_neg
learn_rate = args.learn_rate
batch_size = args.batch_size

train_data_path = data_path + 'rates/rate_train'
with open(train_data_path, 'rb') as f:
    train_data = pickle.load(f)

test_data_path = data_path + 'rates/test_with_neg'
with open(test_data_path, 'rb') as f:
    test_data = pickle.load(f)

with open(data_path + 'adjs/adj_UB', 'rb') as f:
    adj_UB = pickle.load(f)
print("read end.")

mat_path = args.mat_path
U_feature_dir = mat_path + mat_list[0]+".pathsim.feature." + mat_select
I_feature_dir = mat_path + mat_list[1]+".pathsim.feature." + mat_select


if( len(mat_list) == 2 ):
    U_feature_dir2 = mat_path + mat_list[0]+".pathsim.feature." + mat_select
    I_feature_dir2 = mat_path + mat_list[1]+".pathsim.feature."+ mat_select

    U_feature_dir3 = mat_path + mat_list[0]+".pathsim.feature."+ mat_select
    I_feature_dir3 = mat_path + mat_list[1]+".pathsim.feature."+ mat_select
    
    U_feature_dir4 = mat_path + mat_list[0]+".pathsim.feature."+ mat_select
    I_feature_dir4 = mat_path + mat_list[1]+".pathsim.feature."+ mat_select

if( len(mat_list) == 8 ):
    U_feature_dir2 = mat_path + mat_list[2]+".pathsim.feature." + mat_select
    I_feature_dir2 = mat_path + mat_list[3]+".pathsim.feature."+ mat_select

    U_feature_dir3 = mat_path + mat_list[4]+".pathsim.feature."+ mat_select
    I_feature_dir3 = mat_path + mat_list[5]+".pathsim.feature."+ mat_select

    U_feature_dir4 = mat_path + mat_list[6]+".pathsim.feature."+ mat_select
    I_feature_dir4 = mat_path + mat_list[7]+".pathsim.feature."+ mat_select
    
if( len(mat_list) == 6 ):
    U_feature_dir2 = mat_path + mat_list[2]+".pathsim.feature." + mat_select
    I_feature_dir2 = mat_path + mat_list[3]+".pathsim.feature."+ mat_select

    U_feature_dir3 = mat_path + mat_list[4]+".pathsim.feature."+ mat_select
    I_feature_dir3 = mat_path + mat_list[5]+".pathsim.feature."+ mat_select
    
    U_feature_dir4 = mat_path + mat_list[0]+".pathsim.feature."+ mat_select
    I_feature_dir4 = mat_path + mat_list[1]+".pathsim.feature."+ mat_select
'''
if( len(mat_list) == 4):
    U_feature_dir2 = mat_path + mat_list[2]+".pathsim.feature." + mat_select
    I_feature_dir2 = mat_path + mat_list[3]+".pathsim.feature."+ mat_select
    
    U_feature_dir3 = mat_path + mat_list[0]+".pathsim.feature."+ mat_select
    I_feature_dir3 = mat_path + mat_list[1]+".pathsim.feature."+ mat_select
    
    U_feature_dir4 = mat_path + mat_list[2]+".pathsim.feature."+ mat_select
    I_feature_dir4 = mat_path + mat_list[3]+".pathsim.feature."+ mat_select
'''


U_feature = pd.read_csv( U_feature_dir, sep=",", header=None ).fillna( 0 ).as_matrix()
print( "# U_feature1 shape:", U_feature.shape  )
I_feature = pd.read_csv( I_feature_dir, sep=",", header=None ).fillna( 0 ).as_matrix()
print( "# I_feature1 shape:", I_feature.shape  )
I_feature_num = I_feature.shape[1]
U_feature_num = U_feature.shape[1]
# I_feature_num =64
# U_feature_num = 64


U_feature2 = pd.read_csv( U_feature_dir2, sep=",", header=None ).fillna( 0 ).as_matrix()
print( "# U_feature2 shape:", U_feature2.shape  )
I_feature2 = pd.read_csv( I_feature_dir2, sep=",", header=None ).fillna( 0 ).as_matrix()
print( "# I_feature2 shape:", I_feature2.shape  )
I_feature_num2 = I_feature2.shape[1]
U_feature_num2 = U_feature2.shape[1]

U_feature3 = pd.read_csv( U_feature_dir3, sep=",", header=None ).fillna( 0 ).as_matrix()
print( "# U_feature3 shape:", U_feature3.shape  )
I_feature3 = pd.read_csv( I_feature_dir3, sep=",", header=None ).fillna( 0 ).as_matrix()
print( "# I_feature3 shape:", I_feature3.shape  )
I_feature_num3 = I_feature3.shape[1]
U_feature_num3 = U_feature3.shape[1]

U_feature4 = pd.read_csv( U_feature_dir4, sep=",", header=None ).fillna( 0 ).as_matrix()
print( "# U_feature3 shape:", U_feature4.shape  )
I_feature4 = pd.read_csv( I_feature_dir4, sep=",", header=None ).fillna( 0 ).as_matrix()
print( "# I_feature3 shape:", I_feature4.shape  )
I_feature_num4 = I_feature4.shape[1]
U_feature_num4 = U_feature4.shape[1]

U_num, I_num = adj_UB.shape


def get_train_instances( trian_data, num_negs, adj_UB):
    np.random.seed( 1314 )
    user_input, item_input, labels = [],[],[]
    num_users, num_items = adj_UB.shape
    
    for data in train_data:
        u = data['user_id']
        i = data['business_id']
        
        user_input.append( [u] )
        item_input.append( [i] )
        labels.append([1.])

        # negative instances
        for t in range(num_negs):
            j = np.random.randint(num_items)
            while adj_UB[u][j] != 0:
                j = np.random.randint(num_items)
            user_input.append( [u] )
            item_input.append( [j] )
            labels.append([0.])
    return user_input,item_input, labels



# print( "# Starting Negtive Sample..." )
# user_input, item_input, labels = get_train_instances( train_data, num_negs, adj_UB)
# print( "# All Traing Instances:", len( user_input ) )
# print( "# Negtive Sample Done" )


def fm( U, I,  U_num, I_num ,k = 5):
    X = tf.concat([U, I], 1)
    p = U_num + I_num
    print("p:", p)
    w0 = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.zeros([p]))
    print( W )
    print( X )

    # interaction factors, randomly initialized 
    V = tf.Variable(tf.random_normal([k, p], stddev=0.01))

    linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X), 1, keep_dims=True))
    pair_interactions = (tf.multiply(0.5,
                    tf.reduce_sum(
                        tf.subtract(
                            tf.pow( tf.matmul(X, tf.transpose(V)), 2),
                            tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
                        1, keep_dims=True)))
    y_hat = tf.add(linear_terms, pair_interactions)
    return tf.nn.sigmoid( y_hat )



def HIN_MODEL(name,U_embedding, I_embedding, U_feature_num, I_feature_num, hidden_size):
    u_w1 = tf.get_variable(str(name)+"_u_w1", shape=(U_feature_num, hidden_size), initializer=tf.contrib.layers.xavier_initializer())
    u_b1 = tf.get_variable(str(name)+"_u_b1", shape=[hidden_size], initializer=tf.contrib.layers.xavier_initializer())
    
    u_w2 = tf.get_variable(str(name)+"_u_w2", shape=(hidden_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    u_b2 = tf.get_variable(str(name)+"_u_b2", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
    
    u_w3 = tf.get_variable(str(name)+"_u_w3", shape=(last_layer_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    u_b3 = tf.get_variable(str(name)+"_u_b3", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
    
    u_w4 = tf.get_variable(str(name)+"_u_w4", shape=(last_layer_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    u_b4 = tf.get_variable(str(name)+"_u_b4", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
    v_w1 = tf.get_variable(str(name)+"_v_w1", shape=(I_feature_num, hidden_size), initializer=tf.contrib.layers.xavier_initializer())
    v_b1 = tf.get_variable(str(name)+"_v_b1", shape=[hidden_size], initializer=tf.contrib.layers.xavier_initializer())
    v_w2 = tf.get_variable(str(name)+"_v_w2", shape=(hidden_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    v_b2 = tf.get_variable(str(name)+"_v_b2", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
    
    v_w3 = tf.get_variable(str(name)+"_v_w3", shape=(last_layer_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    v_b3 = tf.get_variable(str(name)+"_v_b3", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
    
    v_w4 = tf.get_variable(str(name)+"_v_w4", shape=(last_layer_size, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    v_b4 = tf.get_variable(str(name)+"_v_b4", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())
    
    
    net_u_2 = tf.nn.relu( tf.matmul(U_embedding, u_w1) + u_b1 )
    net_v_2 = tf.nn.relu( tf.matmul(I_embedding, v_w1) + v_b1 )
    
    if( num_of_layers == 1 ):
        return net_u_2, net_v_2 

    net_u_2 = tf.matmul(net_u_2, u_w2) + u_b2 
    net_v_2 = tf.matmul(net_v_2, v_w2) + v_b2 
    if( num_of_layers == 2 ):
        return net_u_2, net_v_2 
    net_u_2 = tf.matmul(net_u_2, u_w3) + u_b3
    net_v_2 = tf.matmul(net_v_2, v_w3) + v_b3
    if( num_of_layers == 3 ):
        return net_u_2, net_v_2 
    net_u_2 = tf.matmul(net_u_2, u_w4) + u_b4
    net_v_2 = tf.matmul(net_v_2, v_w4) + v_b4
    return net_u_2, net_v_2


def cosineSim( U, I ):
    print(U.get_shape().as_list())
    print(I.get_shape().as_list())
    fen_zhi = tf.reduce_sum(U * I, 1, keep_dims=True)
    print(fen_zhi.get_shape().as_list())
    pred_val = tf.nn.sigmoid( fen_zhi )
    print(pred_val.get_shape().as_list())
    return pred_val


def cosineSim_new( name,U, I ):
    X = tf.concat([U, I], 1)
    print( X )
    print( tf.size(X)  )
    
    W = tf.get_variable(name+"W", shape=[ last_layer_size * 2, 1 ], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name+"b", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
    print( tf.matmul(X, W) + b )

    pred_val = tf.nn.sigmoid( tf.matmul(X, W) + b )
    return pred_val

def gen_embedding_matrix( name, U_num, U_feature_num,U_feature, I_num ,I_feature_num, I_feature):
    U_embedding_matrix = np.array(U_feature)
    I_embedding_matrix = np.array(I_feature)
    return U_embedding_matrix, I_embedding_matrix
'''
    U_embedding_matrix = tf.get_variable(name+"embeddings_u", 
                                     shape=[U_num, U_feature_num], 
                                     # initializer=tf.constant_initializer(np.array(U_feature)),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)
                             
    I_embedding_matrix = tf.get_variable(name+"embeddings_i", 
                                     shape=[I_num, I_feature_num], 
                                     # initializer=tf.constant_initializer(np.array(I_feature)),
                                     # trainable=False)
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable = True)
'''
# return U_embedding_matrix, I_embedding_matrix


tf.reset_default_graph()
U_feature_input = tf.placeholder(tf.int32, [None,1])
I_feature_input = tf.placeholder(tf.int32, [None,1])
true_rating = tf.placeholder(tf.float32, [None, 1])

U_embedding_matrix,I_embedding_matrix =  gen_embedding_matrix( "1", U_num, U_feature_num,U_feature, I_num ,I_feature_num, I_feature  )

U_embedding_matrix2,I_embedding_matrix2 =  gen_embedding_matrix( "2", U_num, U_feature_num2,U_feature2, I_num ,I_feature_num2, I_feature2  )

U_embedding_matrix3,I_embedding_matrix3 =  gen_embedding_matrix( "3", U_num, U_feature_num3,U_feature3, I_num ,I_feature_num3, I_feature3  )

U_embedding_matrix4,I_embedding_matrix4 =  gen_embedding_matrix( "4", U_num, U_feature_num4,U_feature4, I_num ,I_feature_num4, I_feature4  )

U_embedding = tf.nn.embedding_lookup(U_embedding_matrix, U_feature_input) 
I_embedding = tf.nn.embedding_lookup(I_embedding_matrix, I_feature_input)  
U_embedding = tf.reshape(U_embedding, [-1,U_feature_num])
I_embedding = tf.reshape(I_embedding, [-1,I_feature_num])


U_embedding2 = tf.nn.embedding_lookup(U_embedding_matrix2, U_feature_input) 
I_embedding2 = tf.nn.embedding_lookup(I_embedding_matrix2, I_feature_input)  
U_embedding2 = tf.reshape(U_embedding2, [-1,U_feature_num2])
I_embedding2 = tf.reshape(I_embedding2, [-1,I_feature_num2])


U_embedding3 = tf.nn.embedding_lookup(U_embedding_matrix3, U_feature_input) 
I_embedding3 = tf.nn.embedding_lookup(I_embedding_matrix3, I_feature_input)  
U_embedding3 = tf.reshape(U_embedding3, [-1,U_feature_num3])
I_embedding3 = tf.reshape(I_embedding3, [-1,I_feature_num3])


U_embedding4 = tf.nn.embedding_lookup(U_embedding_matrix4, U_feature_input) 
I_embedding4 = tf.nn.embedding_lookup(I_embedding_matrix4, I_feature_input)  
U_embedding4 = tf.reshape(U_embedding4, [-1,U_feature_num4])
I_embedding4 = tf.reshape(I_embedding4, [-1,I_feature_num4])


U_embedding_all = tf.concat( [U_embedding,U_embedding2,U_embedding3],1 )
I_embedding_all = tf.concat( [I_embedding,I_embedding2,I_embedding3],1 )


U1, I1 = HIN_MODEL("ob", U_embedding, I_embedding, U_feature_num, I_feature_num, 64)

U2, I2 = HIN_MODEL("iui", U_embedding2, I_embedding2, U_feature_num2, I_feature_num2, 64)

U3, I3 = HIN_MODEL("ui", U_embedding3, I_embedding3, U_feature_num3, I_feature_num3, 64)

U4, I4 = HIN_MODEL("4", U_embedding4, I_embedding4, U_feature_num4, I_feature_num4, 64)

View1 = tf.concat([U1, I1], 1)
View2 = tf.concat([U2, I2], 1)
View3 = tf.concat([U3, I3], 1)
View4 = tf.concat([U4, I4], 1)


def attention(name, input_vec):
    att_w1 = tf.get_variable( name+"att_w1", shape=(last_layer_size, 64 ), initializer=tf.contrib.layers.xavier_initializer())
    att_b1 = tf.get_variable( name+"att_b1", shape=[64], initializer=tf.contrib.layers.xavier_initializer())

    att_w2 = tf.get_variable( name+"att_w2", shape=(64,1), initializer=tf.contrib.layers.xavier_initializer())
    att_b2 = tf.get_variable( name+"att_b2", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
    net_1 = tf.nn.sigmoid( tf.matmul(input_vec, att_w1) + att_b1 )
    net_2 = tf.nn.sigmoid( tf.matmul(net_1, att_w2) + att_b2 )
    return net_2  
    


w1 = tf.exp(  attention( "U1", U1 ) )
w2 = tf.exp(  attention( "U2", U2 ) )
w3 = tf.exp(  attention( "U3", U3 ) )
w4 = tf.exp(  attention( "U4", U4 ) )


if( len(mat_list) == 8 ):
    if merge == "attention":
        U = w1/(w1+w2+w3+w4)*U1 + w2/(w1+w2+w3+w4)*U2 + w3/(w1+w2+w3+w4)*U3 + w4/(w1+w2+w3+w4)*U4
        I = w1/(w1+w2+w3+w4)*I1 + w2/(w1+w2+w3+w4)*I2 + w3/(w1+w2+w3+w4)*I3 + w4/(w1+w2+w3+w4)*I4
    if merge == "avg":
        U = 1/4*U1 + 1/4*U2 + 1/4*U3 + 1/4*U4
        I = 1/4*I1 + 1/4*I2 + 1/4*I3 + 1/4*I4
if( len(mat_list) == 6):
    if merge == "attention":
        U = w1/(w1+w2+w3)*U1 + w2/(w1+w2+w3)*U2 + w3/(w1+w2+w3)*U3
        I = w1/(w1+w2+w3)*I1 + w2/(w1+w2+w3)*I2 + w3/(w1+w2+w3)*I3
    if merge == "avg":
        U = 1/3*U1 + 1/3*U2 + 1/3*U3
        I = 1/3*I1 + 1/3*I2 + 1/3*I3
if( len(mat_list) == 2 ):
    U = w1/(w1+w2+w3)*U1 + w2/(w1+w2+w3)*U2 + w3/(w1+w2+w3)*U3
    I = w1/(w1+w2+w3)*I1 + w2/(w1+w2+w3)*I2 + w3/(w1+w2+w3)*I3


pred_val1 = cosineSim( U1, I1 )
pred_val2 = cosineSim( U2, I2 )
pred_val3 = cosineSim( U3, I3 )
pred_val4 = cosineSim( U4, I4 )
pred_val5 = cosineSim( U, I )


if( len(mat_list) == 2 ):
    pred_val = pred_val1
else:
    pred_val = pred_val5

    
one_constant = tf.constant(1.0, shape=[1, 1])
gmf_loss = tf.reduce_mean(
                - true_rating * tf.log(pred_val + 1e-10) 
                - (one_constant - true_rating) * tf.log(one_constant - pred_val + 1e-10)
            )


loss_all = gmf_loss
train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss_all)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True  
sess = tf.InteractiveSession(config=config)
# with tf.Graph().as_default():
tf.global_variables_initializer().run()

best_r = 0
best_epoch = -1
stop_train = False
for epoch in range( epochs ): 
    if stop_train:
        break
    print( epoch )
    one_epoch_loss = 0.0
    one_epoch_batchnum = 0.0
    user_input, item_input, labels = get_train_instances(train_data, num_negs, adj_UB)
    # user_input, item_input, labels = shuffle(user_input, item_input, labels)
    print( "Start Traing...." )
    for index in range( len(user_input) // batch_size + 1 ):
        batch_u =  user_input[index * batch_size:(index + 1) * batch_size]
        batch_i =  item_input[index * batch_size:(index + 1) * batch_size]
        batch_labels =  labels[index * batch_size:(index + 1) * batch_size]
        _, loss_val, pred_value,w1_ob,w2_ob,w3_ob = sess.run(
            [train_step, gmf_loss, pred_val,w1,w2,w3],
            feed_dict={U_feature_input: batch_u, I_feature_input: batch_i, true_rating: batch_labels})
        output = np.array(pred_value).reshape((-1))
        print(output)
        # print(batch_labels)
        # print(loss_val)
        
        one_epoch_loss += loss_val
        one_epoch_batchnum += 1.0
        if index % 100 == 0:
            format_str = '# %s: Progress %.2f %%, Loss = %.4f'
            print (format_str % ( datetime.now(), index /( len(user_input) // batch_size ) * 100 , one_epoch_loss / (index+1) ) )
    
        if index == len(user_input) // batch_size:
                format_str = '# ****%s: %d epoch, iteration averge loss = %.4f '
                print (format_str % (datetime.now(), epoch, one_epoch_loss / one_epoch_batchnum))
                eval_p = []
                eval_r = []
                eval_ndcg = []
                for data in test_data:
                    user = data['user_id']
                    pos_n = len(data['pos_business_id'])
                    neg_n = len(data['neg_business_id'])

                    items = []
                    for item in data['pos_business_id']:
                        items.append([item])
                    for item in data['neg_business_id']:
                        items.append([item])
                    user = [[user]] * (pos_n + neg_n)
                    print(user, items)
                    pred_value = sess.run([pred_val], feed_dict={U_feature_input: user, I_feature_input: items})
                    output = np.array(pred_value).reshape((-1))
                    
                    pred_items, indexes = torch.topk(torch.tensor(output), K)
                    indexes = indexes.tolist()
                    gt_items = list(range(pos_n))
                    print(output)
                    print(indexes, gt_items)
                    p_at_k = getP(indexes, gt_items)
                    r_at_k = getR(indexes, gt_items)
                    ndcg_at_k = getNDCG(indexes, gt_items)
                    print(p_at_k, r_at_k, ndcg_at_k)

                    eval_p.append(p_at_k)
                    eval_r.append(r_at_k)
                    eval_ndcg.append(ndcg_at_k)
                    '''
                    # Get prediction scores
                    map_item_score = {}
                    for i in range(len(items)):
                        item = items[i]
                        map_item_score[item] = pre_real_val[i]

                    # Evaluate top rank list
                    ranklist = heapq.nlargest(20, map_item_score, key=map_item_score.get)
            
                    hr_list_5.append(getHitRatio(ranklist[:5], gtItem))
                    ndcg_list_5.append(getNDCG(ranklist[:5], gtItem))
            
                    hr_list_10.append(getHitRatio(ranklist[:10], gtItem))
                    ndcg_list_10.append(getNDCG(ranklist[:10], gtItem))
                    
                    hr_list_15.append(getHitRatio(ranklist[:15], gtItem))
                    ndcg_list_15.append(getNDCG(ranklist[:15], gtItem))
                    
                    hr_list_20.append(getHitRatio(ranklist[:20], gtItem))
                    ndcg_list_20.append(getNDCG(ranklist[:20], gtItem))
                    '''
                '''
                final_hr_metric_list_5.append( np.array(hr_list_5).mean() )
                final_ndcg_metric_list_5.append( np.array(ndcg_list_5).mean() )
                
                final_hr_metric_list_10.append( np.array(hr_list_10).mean() )
                final_ndcg_metric_list_10.append( np.array(ndcg_list_10).mean() )
                
                final_hr_metric_list_15.append( np.array(hr_list_15).mean() )
                final_ndcg_metric_list_15.append( np.array(ndcg_list_15).mean() )
                
                final_hr_metric_list_20.append( np.array(hr_list_20).mean() )
                final_ndcg_metric_list_20.append( np.array(ndcg_list_20).mean() )
                
                
                print( "***Result HR@5,:NDCG@5: ", final_hr_metric_list_5[-1], final_ndcg_metric_list_5[-1] )
                print( "***Result HR@10,:NDCG@10: ", final_hr_metric_list_10[-1], final_ndcg_metric_list_10[-1] )
                print( "***Result HR@15,:NDCG@15: ", final_hr_metric_list_15[-1], final_ndcg_metric_list_15[-1] )
                print( "***Result HR@20,:NDCG@20: ", final_hr_metric_list_20[-1], final_ndcg_metric_list_20[-1] )
                '''
                mean_p = np.mean(eval_p)
                mean_r = np.mean(eval_r)
                mean_ndcg = np.mean(eval_ndcg)
                print('Valid:\tprecision@', K, ':', mean_p, ', recall@', K, ':', mean_r, ', ndcg@', K, ':', mean_ndcg)
                
                if mean_r > best_r:
                    best_r = mean_r
                    best_epoch = epoch
                    saver = tf.train.Saver()
                    saver.save(sess, 'model/neuacf-model')
                    print('Model save for better valid recall:', best_r)
                
                if epoch - best_epoch >= args.patience:
                    print("Stop training at epoch", epoch)
                    stop_train = True

sess = tf.Session()
saver = tf.train.import_meta_graph('model/neuacf-model.meta')
saver.restore(sess, tf.train.latest_checkpoint("model"))

print("Valid:")
valid_data_path = data_path + 'rates/valid_with_neg'
with open(valid_data_path, 'rb') as f:
    valid_data = pickle.load(f)

eval_p = []
eval_r = []
eval_ndcg = []
for data in valid_data:
    user = data['user_id']
    pos_n = len(data['pos_business_id'])
    neg_n = len(data['neg_business_id'])

    items = []
    for item in data['pos_business_id']:
        items.append([item])
    for item in data['neg_business_id']:
        items.append([item])
    user = [[user]] * (pos_n + neg_n)
    pred_value = sess.run([pred_val], feed_dict={U_feature_input: user, I_feature_input: items})
    output = np.array(pred_value).reshape((-1))
            
    pred_items, indexes = torch.topk(torch.tensor(output), K)
    indexes = indexes.tolist()
    gt_items = range(pos_n)
    p_at_k = getP(indexes, gt_items)
    r_at_k = getR(indexes, gt_items)
    ndcg_at_k = getNDCG(indexes, gt_items)
    
    eval_p.append(p_at_k)
    eval_r.append(r_at_k)
    eval_ndcg.append(ndcg_at_k)
print('Valid:\tprecision@', K, ':', mean_p, ', recall@', K, ':', mean_r, ', ndcg@', K, ':', mean_ndcg)
