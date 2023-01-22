import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class GATLayer(nn.Module):
    """GAT layer"""

    def __init__(self, n_feat, n_hidden, alpha=0.05):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(n_feat, n_hidden, bias=False)
        self.a = nn.Linear(2 * n_hidden, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):

        z = self.fc(x)
        indices = adj.coalesce().indices()
        h = torch.cat((z[indices[0, :], :], z[indices[1, :], :]), dim=1)
        h = self.a(h)
        h = self.leakyrelu(h)

        h = torch.exp(h.squeeze())
        unique = torch.unique(indices[0, :])
        t = torch.zeros(unique.size(0), device=x.device)
        h_sum = t.scatter_add(0, indices[0, :], h)
        h_norm = torch.gather(h_sum, 0, indices[0, :])
        alpha = torch.div(h, h_norm)
        adj_att = torch.sparse.FloatTensor(
            indices, alpha, torch.Size([x.size(0), x.size(0)])
        ).to(x.device)

        out = torch.sparse.mm(adj_att, z)

        return out


class GNN(nn.Module):
    """
    Graph Neural Network either using Graph Convolutional Layers or
    Graph Attention Layers

    Arguments
    ---------
        input_dim: int
            Dimension of the vector embeddings
        hidden_dim: int
            Dimension of the hidden dimensions of the model
        dropout: float
            Percentage of dropout applied
        n_class: int
            Number of classes in y
        type_model: str ["Gat","GConv"]
            Select the convolution or the attention layers
    """

    def __init__(self, input_dim, hidden_dim, dropout, n_class, type_model="GAT"):
        super(GNN, self).__init__()

        if type_model == "GAT":
            self.mp1 = GATLayer(input_dim, hidden_dim)
            self.mp2 = GATLayer(hidden_dim, hidden_dim)
            self.mp3 = GATLayer(hidden_dim, hidden_dim)
        elif type_model == "GConv":
            self.mp1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
            self.mp2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
            self.mp3 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        else:
            raise ValueError("type_model must be either GAT or GCONV")
        self.fc = nn.Linear(hidden_dim, n_class)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.05)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, adj, idx):
        # first message passing layer
        x = self.mp1(x_in, adj)
        x = self.leakyrelu(x)
        x = self.dropout(x)

        # second message passing layer
        x = self.mp2(x, adj)
        x = self.leakyrelu(x)
        x = self.dropout(x)

        # third message passing layer
        x = self.mp3(x, adj)
        x = self.leakyrelu(x)

        # sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx) + 1, x.size(1)).to(x_in.device)
        out = out.scatter_add_(0, idx, x)

        # batch normalization layer
        out = self.bn(out)

        # mlp to produce output
        out = self.fc(out)
        return F.log_softmax(out, dim=1)
