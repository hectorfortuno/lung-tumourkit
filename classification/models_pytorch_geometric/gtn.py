import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from .norm import Norm

class GTN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, heads, num_layers, drop_rate, norm_type, enable_background=False):
        super(GTN, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(TransformerConv(in_feats, h_feats, heads))
        self.conv_layers.append(nn.ELU())
        self.conv_layers.append(nn.Dropout(drop_rate)) # Feature map dropout
        self.conv_layers.append(Norm(norm_type=norm_type, hidden_dim=h_feats))

        for l in range(1,num_layers):
            self.conv_layers.append(TransformerConv(h_feats, h_feats, heads))
            self.conv_layers.append(nn.ELU())
            self.conv_layers.append(nn.Dropout(drop_rate))
            self.conv_layers.append(Norm(norm_type=norm_type, hidden_dim=h_feats))

        self.conv_layers.append(TransformerConv(h_feats, num_classes, heads))


    def forward(self, x, edge_index, return_attention_weights=None):
        h = x
        attention_edges, attention_weights = [], []
        for i, layer in enumerate(self.conv_layers):

            if i % 4 == 0:  # transformer layer
                h = layer(h, edge_index, return_attention_weights=return_attention_weights)
                if return_attention_weights is not None:
                    h, (att_edge_index, att_weights) = h
                    attention_edges.append(att_edge_index)
                    attention_weights.append(att_weights)
            else:
                if i % 4 == 1 or i % 4 == 2:
                    h = layer(h)  # ELU and dropout
                else:
                    h = layer(h, edge_index)  # Other layers

            """
            if i == len(self.conv_layers) - 1:          # last layer
                h = layer(h, edge_index)
            else:
                if i % 4 == 1 or i % 4 == 2:
                    h = layer(h)              # ELU and dropout
                else:
                    h = layer(h, edge_index)  # Other layers
            """

        if return_attention_weights is not None:
            return h, attention_edges, attention_weights

        return h