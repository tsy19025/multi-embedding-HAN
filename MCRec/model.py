import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import randint
import sys

class Path_Embedding(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size = 1):
        super(Path_Embedding, self).__init__()
        self.out_dim = out_dim

        self.conv1d = nn.Conv1d(in_dim, out_dim, kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, path_input, path_num, timestamp, user_latent, item_latent, index):
        batch_size = len(path_input)
        # path_input: batch_size * path_num * timestamp * in_dim
        
        outputs = []
        for i in range(path_num):
            path = self.conv1d(path_input[:, i, :, :].squeeze(1).permute([0, 2, 1]))
            path = F.max_pool2d(path, kernel_size = (1, path.shape[-1])).squeeze(-1)
            output = self.dropout(path)
            outputs.append(output)
        outputs = torch.cat(outputs, -1).view(batch_size, path_num, self.out_dim).permute([0, 2, 1])
        outputs = F.max_pool2d(outputs, kernel_size = (1, outputs.shape[-1])).squeeze(-1)
        return outputs

class AttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AttentionLayer, self).__init__()
        self.layer = nn.Linear(in_dim * 2, out_dim)
        self.relu = nn.ReLU()
    def forward(self, latent, path_output):
        inputs = torch.cat([latent, path_output], -1)
        # print(inputs.shape)
        output = self.relu(self.layer(inputs))
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
    def __init__(self, users, items, path_nums, timestamps, feature_dim, latent_dim, device):
        super(MCRec, self).__init__()

        # print("latent_dim: ", latent_dim)
        self.users = users
        self.items = items
        self.path_nums = path_nums
        self.timestamps = timestamps
        self.device = device
        self.latent_dim = latent_dim

        self.user_embedding = nn.Embedding(users, latent_dim).to(device)
        self.item_embedding = nn.Embedding(items, latent_dim).to(device)
        self.path_embedding = [Path_Embedding(feature_dim, latent_dim).to(device) for i in range(len(path_nums))]

        self.user_attention = AttentionLayer(latent_dim, latent_dim).to(device)
        self.item_attention = AttentionLayer(latent_dim, latent_dim).to(device)
        self.metapath_attention = MetapathAttentionLayer(3 * latent_dim, latent_dim).to(device)
        self.prediction_layer = nn.Linear(3 * latent_dim, 1).to(device)
    def forward(self, user_input, item_input, path_inputs):
        # user_input: batch_size * 1(one_hot)
        # item_input: batch_size * 1(one_hot)
        # path_intpus: paths * batch_size * path_num * timestamp * length

        paths = len(path_inputs)
        batch_size = user_input.shape[0]

        user_latent = torch.cat([self.user_embedding(user) for user in user_input], -1)
        user_latent = user_latent.view(batch_size, -1).to(self.device)
        # print(user_latent.shape)

        item_latent = torch.cat([self.item_embedding(item) for item in item_input], -1)
        item_latent = item_latent.view(batch_size, -1).to(self.device)
        # print(item_latent.shape)
        
        path_latent = [self.path_embedding[i](path_inputs[i], self.path_nums[i], self.timestamps[i], user_latent, item_latent, i) for i in range(paths)]
        path_latent = torch.cat(path_latent, -1).view(paths, batch_size, self.latent_dim).to(self.device)
        # print(path_latent.shape)

        path_attention = self.metapath_attention(user_latent, item_latent, path_latent)
        user_attention = self.user_attention(user_latent, path_attention)
        item_attention = self.item_attention(item_latent, path_attention)

        output = torch.cat([user_attention, path_attention, item_attention], -1)
        # print("output: ", output.shape)
        prediction = self.prediction_layer(output)
        return torch.sigmoid(prediction)

