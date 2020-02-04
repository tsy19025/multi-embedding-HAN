import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import randint
import sys
import time

def one_hot(index, n):
    tmp = torch.zeros(n, dtype = torch.int)
    tmp[int(index)] = 1
    return tmp

class Path_Embedding(nn.Module):
    def __init__(self, in_dim, out_dim, n_type, device, kernel_size = 2):
        super(Path_Embedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_type = n_type
        self.device = device

        self.conv1d = nn.Conv1d(in_dim, out_dim, kernel_size)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
    def forward(self, path_input, path_num, timestamp, path_type, type_embedding):
        batch_size = len(path_input)
        # path_input: batch_size * path_num * timestamp
        
        outputs = []
        for i in range(path_num):
            paths = path_input[:, i, :].squeeze(1)
            path = []
            for j in range(len(path_type)):
                path.append(type_embedding[path_type[j]](paths[:, j]))
            input = torch.cat(path).view(timestamp, batch_size, self.in_dim).permute([1, 2, 0])

            path = self.conv1d(input)
            # assert(input == path)
            # print(path.shape)
            path = F.max_pool2d(path, kernel_size = (1, path.shape[-1])).squeeze(-1)
            output = path
            assert(list(output.shape) == [batch_size, self.in_dim])
            outputs.append(output)
        outputs = torch.cat(outputs, -1).view(path_num, batch_size, self.out_dim).permute([1, 2, 0])
        outputs = F.max_pool2d(outputs, kernel_size = (1, outputs.shape[-1])).squeeze(-1)
        assert(list(output.shape) == [batch_size, self.out_dim])
        return outputs

class AttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AttentionLayer, self).__init__()
        self.layer = nn.Linear(in_dim * 2, out_dim)
        self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.zeros(out_dim))
    def forward(self, latent, path_output):
        inputs = torch.cat([latent, path_output], -1)
        # print(inputs.shape)
        output = self.relu(self.layer(inputs) + self.bias)
        attention = F.softmax(output, -1)
        output = latent * attention
        return output
        
class MetapathAttentionLayer(nn.Module):
    def __init__(self, dim, hiddens):
        super(MetapathAttentionLayer, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(dim, hiddens)
        self.layer2 = nn.Linear(hiddens, 1)
        self.relu = nn.ReLU()
    def forward(self, user_latent, item_latent, metapath_latent):
        # metapath_latent: batch_size * paths * dim
        paths, batch_size, dim = metapath_latent.shape

        outputs = []
        for metapath in metapath_latent:
            inputs = torch.cat([user_latent, item_latent, metapath], dim = -1)
            # print(inputs.shape)
            output = self.relu(self.layer2(self.relu(self.layer1(inputs))))
            outputs.append(output.squeeze(-1))
        outputs = torch.cat(outputs, 0).view(paths, batch_size)
        attention = F.softmax(outputs, -1)
        return sum(metapath_latent * attention.unsqueeze(-1), 1)

class MCRec(nn.Module):
    def __init__(self, n_type, path_nums, timestamps, latent_dim, path_type, device):
        super(MCRec, self).__init__()

        # print("latent_dim: ", latent_dim)
        self.users = n_type[0]
        self.items = n_type[1]
        self.path_nums = path_nums
        self.timestamps = timestamps
        self.device = device
        self.latent_dim = latent_dim
        self.path_type = path_type

        #self.type_embedding = [nn.Embedding(n_type[i], feature_dim).to(device) for i in range(len(n_type))]

        self.user_embedding = nn.Embedding(self.users, latent_dim).to(device)
        self.item_embedding = nn.Embedding(self.items, latent_dim).to(device)
        #self.user_embedding = self.type_embedding[0]
        #self.item_embedding = self.type_embedding[1]
        #self.path_embedding = [Path_Embedding(feature_dim, latent_dim, n_type, device).to(device) for i in range(len(path_nums))]

        # self.user_attention = AttentionLayer(latent_dim, latent_dim).to(device)
        # self.item_attention = AttentionLayer(latent_dim, latent_dim).to(device)
        # self.metapath_attention = MetapathAttentionLayer(3 * latent_dim, latent_dim).to(device)
        self.prediction_layer = nn.Linear(latent_dim*2//(2**3), 1).to(device)

        MLP_modules = []
        for i in range(3):
            input_size = latent_dim*2 // (2 ** i)
            MLP_modules.append(nn.Dropout(0.0))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

    def forward(self, user_input, item_input):
        # user_input: batch_size * 1(one_hot)
        # item_input: batch_size * 1(one_hot)
        # path_intpus: paths * batch_size * path_num * timestamp * length

        # paths = len(path_inputs)
        batch_size = user_input.shape[0]

        user_latent = torch.cat([self.user_embedding(user) for user in user_input], -1)
        user_latent = user_latent.view(batch_size, -1).to(self.device)
        # print("user_latent: ", user_latent)

        # print(user_latent.shape)

        item_latent = torch.cat([self.item_embedding(item) for item in item_input], -1)
        item_latent = item_latent.view(batch_size, -1).to(self.device)
        # print("item_latent: ", item_latent)
        # print(item_latent.shape)
        
        # path_latent = [self.path_embedding[i](path_inputs[i], self.path_nums[i], self.timestamps[i], self.path_type[i], self.type_embedding) for i in range(paths)]
        # path_latent = torch.cat(path_latent, -1).view(paths, batch_size, self.latent_dim).to(self.device)
        # print(path_latent.shape)

        # path_attention = self.metapath_attention(user_latent, item_latent, path_latent)
        # user_attention = self.user_attention(user_latent, item_latent)
        # item_attention = self.item_attention(item_latent, user_latent)

        output = torch.cat([user_latent, item_latent], -1)
        # print(output)
        # print("output: ", output.shape)
        prediction = self.MLP_layers(output)
        prediction = self.prediction_layer(prediction)
        return torch.sigmoid(prediction)

