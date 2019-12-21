import torch
import torch.nn as nn

def Path_Embedding(nn.Module):
    def init(self, in_dim, out_dim, kernel_size = 4):
        super(Path_Embedding, self).__init__()
        self.out_dim = out_dim

        self.conv1d = nn.Conv1D(in_dim, out_dim, kernel_size)
        self.relu = nn.ReLU()
        self.maxpool1d = nn.MaxPool1d()
        self.dropout = nn.Dropout(0.5)
    def forward(self, path_input, path_num, timestamp, user_latent, item_latent):
        outputs = []
        for i in range(path_num):
            path = path_input[:, i, :, :]
            output = self.dropout(self.maxpool1d(self.conv1d(path)))
            outputs = torch.concat([outputs, output])
        outputs = outputs.view(path_num, self.out_dim)
        return self.maxpool1d(outputs)

def AttentionLayer(nn.Module):
    def init(self, in_dim, out_dim):
        super(AttentionLayer, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
    def forward(self, latent, path_output):
        inputs = torch.concat([latent, path_output])
        output = self.relu(self.layer(inputs))
        attention = F.softmax(output)
        output = latent * attention
        return output
        
def MetapathAttentionLayer(nn.Module):
    def init(self, dim, hiddens):
        super(MetapathAttentionLayer, self).__init__()

        self.layer1 = nn.Linear(dim, hiddens)
        self.layer2 = nn.Linear(hiddens, 1)
        self.relu = nn.ReLU()
    def forward(self, user_latent, item_latent, metapath_latent):
        latent_size = user_latent.shape[1]
        _, path_num, path_latent_size = metapath_latent.shape

        outputs = []
        for i in range(path_num):
            metapath = metapath_latent[:, i, :]
            inputs = torch.concat([user_latent, item_latent, metapath])
            output = self.relu(self.layer2(self.relu(self.layer1(inputs))))
            outputs = torch.concat([outputs, output])
        attention = F.softmax(outputs)
        return sum(metapath_latent * torch.unsqueeze(attention, -1), 1)


def MCRec(nn.Module):
    def init(self, users, items, paths, path_nums, path_times, ):
        super(MCRec, self).__init__()

        self.users = users
        self.items = items

        self.user_embedding = nn.Embedding(users, latent_dim)
        self.item_embedding = nn.Embedding(items, latent_dim)
        self.path_embedding = [Path_Embedding(timestamps[i], out_dim) for i in range(paths)]

        self.user_attention = AttentionLayer(latent_dim, latent_dim)
        self.item_attention = AttentionLayer(latent_dim, latent_dim)
        self.matapath_attention = MetapathAttention(3 * latent_dim, latent_dim)
        self.prediction_layer = nn.Linear(3 * latent_dim, 1)
    def forward(self, user_input, item_input, path_inputs):
        user_input: batch_size * 1(one_hot)
        item_input: batch_size * 1(one_hot)
        path_intpus: batch_size * paths * path_num * timestamp * length

        paths = path_inputs.shape[0]
        user_latent = [self.user_embedding(user) for user in user_input]
        item_latent = [self.item_embedding(item) for item in item_input]
        path_latent = [self.path_embedding(path_inputs[:, i, :, :], self.path_num[i], self.timestamps[i], user_latent, item_latent) for i in range(paths)]
        path_attention = self.metapath_attention(user_latent, item_latent, path_latent)
        user_attention = self.user_attention(user_latent, path_attention)
        item_attention = self.item_attention(item_latent, path_attention)

        output = torch.cat([user_attention, path_attention, item_attention])
        prediction = self.prediction_layer(output)
        return F.sigmoid(prediction)

