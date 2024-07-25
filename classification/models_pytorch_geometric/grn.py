import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FAConv

from .norm import Norm

class GInitResN(nn.Module):
    def __init__(self, in_feats, num_classes, num_layers, drop_rate, norm_type, enable_background=False):
        super(GInitResN, self).__init__()

        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(FAConv(in_feats, eps=1))
        self.conv_layers.append(nn.ELU())
        self.conv_layers.append(nn.Dropout(drop_rate)) # Feature map dropout
        self.conv_layers.append(Norm(norm_type=norm_type, hidden_dim=in_feats))

        for l in range(1,num_layers):
            self.conv_layers.append(FAConv(in_feats, eps=1))
            self.conv_layers.append(nn.ELU())
            self.conv_layers.append(nn.Dropout(drop_rate))
            self.conv_layers.append(Norm(norm_type=norm_type, hidden_dim=in_feats))

        self.conv_layers.append(nn.Linear(in_feats, num_classes))


    def forward(self, x, edge_index):
        h = x
        h0 = x
        for i, layer in enumerate(self.conv_layers):
            if i == len(self.conv_layers) - 1:          # last layer
                h = layer(h)
            else:
                if i % 4 == 1 or i % 4 == 2:
                    h = layer(h)              # ELU and dropout
                else:
                    h = layer(h, h0, edge_index)  # Other layers
        return h


class GResN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers, drop_rate, norm_type, enable_background=False):
        super(GResN, self).__init__()

        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(nn.Linear(in_feats, h_feats))
        self.conv_layers.append(nn.ELU())
        self.conv_layers.append(nn.Dropout(drop_rate))  # Feature map dropout
        self.conv_layers.append(Norm(norm_type=norm_type, hidden_dim=h_feats))

        self.conv_layers.append(FAConv(h_feats, eps=1))
        self.conv_layers.append(nn.ELU())
        self.conv_layers.append(nn.Dropout(drop_rate)) # Feature map dropout
        self.conv_layers.append(Norm(norm_type=norm_type, hidden_dim=h_feats))

        for l in range(1,num_layers):
            self.conv_layers.append(FAConv(h_feats, eps=1))
            self.conv_layers.append(nn.ELU())
            self.conv_layers.append(nn.Dropout(drop_rate))
            self.conv_layers.append(Norm(norm_type=norm_type, hidden_dim=h_feats))

        self.conv_layers.append(nn.Linear(h_feats, num_classes))


    def forward(self, x, edge_index):
        h = x
        h0 = x
        for i, layer in enumerate(self.conv_layers):
            if i == 0:                                  # first layer (linear)
                h = layer(h)
                h0 = h
            elif i == len(self.conv_layers) - 1:          # last layer
                h = layer(h)
            else:
                if i % 4 == 1 or i % 4 == 2:
                    h = layer(h)              # ELU and dropout
                else:
                    h = layer(h, h0, edge_index)  # Other layers
        return h