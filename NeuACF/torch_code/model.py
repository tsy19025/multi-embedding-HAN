import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    def __init__(self, last_layer_size):
        super(Attention, self).__init__()

        self.layer1 = nn.Linear(last_layer_size, 64)
        self.layer2 = nn.Linear(64, 1)
    def forward(self, input):
        net_1 = torch.sigmoid(self.layer1(input))
        net_2 = torch.sigmoid(self.layer2(net_1))
        return net_2

class HIN(nn.Module):
    def __init__(self, nlayer, u_feature_num, i_feature_num, hidden_size, last_layer_size):
        super(HIN, self).__init__()

        self.nlayer = nlayer
        self.ulayer0 = nn.Linear(u_feature_num, hidden_size)
        self.ulayer1 = nn.Linear(hidden_size, last_layer_size)
        self.ulayer2 = nn.Linear(last_layer_size, last_layer_size)
        self.ulayer3 = nn.Linear(last_layer_size, last_layer_size)

        self.ilayer0 = nn.Linear(i_feature_num, hidden_size)
        self.ilayer1 = nn.Linear(hidden_size, last_layer_size)
        self.ilayer2 = nn.Linear(last_layer_size, last_layer_size)
        self.ilayer3 = nn.Linear(hidden_size, last_layer_size)

        self.relu = nn.ReLU()

    def forward(self, u_embedding, i_embedding):
        net_u_2 = self.relu(self.ulayer0(u_embedding))
        net_v_2 = self.relu(self.ilayer0(i_embedding))
        if self.nlayer == 1: return net_u_2, net_v_2

        net_u_2 = self.ulayer1(net_u_2)
        net_v_2 = self.ilayer1(net_v_2)
        if self.nlayer == 2: return net_u_2, net_v_2

        net_u_2 = self.ulayer2(net_u_2) + self.ub2
        net_v_2 = self.ilayer2(net_v_2) + self.ib2
        if self.nlayer == 3: return net_u_2, net_v_2

        net_u_2 = self.ulayer3(net_u_2) + self.ub3
        net_v_2 = self.ilayer3(net_v_2) + self.ib3
        return net_u_2, net_v_2

class NeuACF(nn.Module):
    def __init__(self, u_feature_num, i_feature_num, hidden_size, last_layer_size, n_mat, nlayer, merge, device):
        super(NeuACF, self).__init__()

        self.n_mat = n_mat
        self.merge = merge
        self.u_feature_num = u_feature_num
        self.i_feature_num = i_feature_num

        self.hin0 = HIN(nlayer, u_feature_num[0], i_feature_num[0], hidden_size, last_layer_size).to(device)
        self.hin1 = HIN(nlayer, u_feature_num[1], i_feature_num[1], hidden_size, last_layer_size).to(device)
        self.hin2 = HIN(nlayer, u_feature_num[2], i_feature_num[2], hidden_size, last_layer_size).to(device)
        self.hin3 = HIN(nlayer, u_feature_num[3], i_feature_num[3], hidden_size, last_layer_size).to(device)

        self.attention0 = Attention(last_layer_size).to(device)
        self.attention1 = Attention(last_layer_size).to(device)
        self.attention2 = Attention(last_layer_size).to(device)
        self.attention3 = Attention(last_layer_size).to(device)

    def forward(self, u_input, i_input):
        u_input = u_input.permute(1, 0, 2)
        i_input = i_input.permute(1, 0, 2)
        U_embedding = u_input[0]
        U_embedding2 = u_input[1]
        U_embedding3 = u_input[2]
        U_embedding4 = u_input[3]
        I_embedding = i_input[0]
        I_embedding2 = i_input[1]
        I_embedding3 = i_input[2]
        I_embedding4 = i_input[3]

        U1, I1 = self.hin0(U_embedding, I_embedding)
        U2, I2 = self.hin1(U_embedding2, I_embedding2)
        U3, I3 = self.hin2(U_embedding3, I_embedding3)
        U4, I4 = self.hin3(U_embedding4, I_embedding4)
        
        w1 = torch.exp(self.attention0(U1))
        w2 = torch.exp(self.attention1(U2))
        w3 = torch.exp(self.attention2(U3))
        w4 = torch.exp(self.attention3(U4))

        if self.n_mat == 8:
            if self.merge == "attention":
                U = w1/(w1+w2+w3+w4)*U1 + w2/(w1+w2+w3+w4)*U2 + w3/(w1+w2+w3+w4)*U3 + w4/(w1+w2+w3+w4)*U4       
                I = w1/(w1+w2+w3+w4)*I1 + w2/(w1+w2+w3+w4)*I2 + w3/(w1+w2+w3+w4)*I3 + w4/(w1+w2+w3+w4)*I4
            if self.merge == "avg":
                U = 1/4*U1 + 1/4*U2 + 1/4*U3 + 1/4*U4
                I = 1/4*I1 + 1/4*I2 + 1/4*I3 + 1/4*I4
        if self.n_mat == 6:
            if self.merge == "attention":
                U = w1/(w1+w2+w3)*U1 + w2/(w1+w2+w3)*U2 + w3/(w1+w2+w3)*U3
                I = w1/(w1+w2+w3)*I1 + w2/(w1+w2+w3)*I2 + w3/(w1+w2+w3)*I3
            if self.merge == "avg":
                U = 1/3*U1 + 1/3*U2 + 1/3*U3
                I = 1/3*I1 + 1/3*I2 + 1/3*I3
        if self.n_mat == 2:
            U = w1/(w1+w2+w3)*U1 + w2/(w1+w2+w3)*U2 + w3/(w1+w2+w3)*U3
            I = w1/(w1+w2+w3)*I1 + w2/(w1+w2+w3)*I2 + w3/(w1+w2+w3)*I3
        
        def cosineSim(U, I):
            fen_zhi = torch.sum(U*I, 1).view(-1, 1)
            pred_zhi = torch.sigmoid(fen_zhi)
            return pred_zhi

        pred_val1 = cosineSim( U1, I1 )
        pred_val2 = cosineSim( U2, I2 )
        pred_val3 = cosineSim( U3, I3 )
        pred_val4 = cosineSim( U4, I4 )
        pred_val5 = cosineSim( U, I )

        if self.n_mat == 2: pred_val = pred_val1
        else: pred_val = pred_val5

        return pred_val

