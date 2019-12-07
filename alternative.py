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
        # output: [nlayers, batch_size, output_size]
        return output[-1]

class mean_pooling(nn.Module):
    def __init__(self):
        super(mean_pooling, self).__init__()
    def forward(self, node_feature, node_neighbor_feature):
        n = node_neighbor_feature.shape[1]
        x = torch.cat((node_feature.unsqueeze(1), node_neighbor_feature), dim = 1)
        x = torch.sum(x, dim = 1).squeeze() / (n + 1)
        return x

if __name__ == "__main__":
    batch_size = 3
    n_facet = 4
    emb_dim = 5
    neighbor_size = 7
    nlayers = 2

    node_feature = torch.rand(batch_size, n_facet * emb_dim)
    node_neighbor_feature = torch.rand(batch_size, neighbor_size, n_facet * emb_dim)

    model = mean_pooling()
    print(model(node_feature, node_neighbor_feature))
