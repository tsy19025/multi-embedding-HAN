import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, n_facet, emb_dim, lstm_layers):
        super(LSTM, self).__init__()
        self.n_facet = n_facet
        self.emb_dim = emb_dim
        self.layer = nn.LSTM(n_facet * emb_dim, n_facet * emb_dim, lstm_layers) 

    def forward(self, node_feature, node_neighbor_feature):
        # node_neighbor_feature: [batch_size, neighbor_size, n_facet, emb_dim]
        batch_size = node_feature.shape[0]
        neighbor_size = node_neighbor_feature.shape[1]
        u = node_feature.view(batch_size, self.n_facet * self.emb_dim)
        v = node_neighbor_feature.view(batch_size, neighbor_size, self.n_facet * self.emb_dim)

        input = torch.cat((u.unsqueeze(1), v), dim = 1)
        input = input.permute(1, 0, 2)
        output, _ = self.layer(input)
        return output[-1].view(batch_size, -1)

class 
        '''
        w = torch.sum(v * u.view(batch_size, 1, n_facet, emb_dim), dim = 3)

        x = v.permute(1, 0, 2, 3)
        x = x.view(neighbor_size, batch_size, -1)
        x = self.layer(x).view(batch_size, neighbor_size, self.n_facet, self.emb_dim)

        u = torch.sum(x * w.view(batch_size, neighbor_size, self.n_facet, 1), dim = 1)
        return u.view(batch_size, self.n_facet * self.emb_dim)
        '''

if __name__ == "__main__":
    batch_size = 3
    n_facet = 4
    emb_dim = 5
    neighbor_size = 7
    nlayers = 2

    node_feature = torch.rand(batch_size, n_facet * emb_dim)
    node_neighbor_feature = torch.rand(batch_size, neighbor_size, n_facet * emb_dim)

    model = LSTM(n_facet, emb_dim, nlayers)
    print(model(node_feature, node_neighbor_feature).shape)
