import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv    


class HANModel(nn.Module):
    def __init__(self, in_channels_dict, out_channels, metadata, hidden_channels=64, heads=8):
        super(HANModel, self).__init__()
        self.metadata = metadata
        common_dim = hidden_channels

        self.lin_dict = nn.ModuleDict()
        for node_type, in_dim in in_channels_dict.items():
            self.lin_dict[node_type] = nn.Linear(in_dim, common_dim)

        # Remove metapaths. HANConv will infer them based on the metadata
        self.conv = HANConv(
            in_channels=common_dim,
            out_channels=common_dim,
            metadata=self.metadata,
            heads=heads
        )

        self.out_lin = nn.Linear(heads * common_dim, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: self.lin_dict[k](x) for k, x in x_dict.items()}
        x_dict = self.conv(x_dict, edge_index_dict)
        # print("After HANConv, document shape:", x_dict['document'].shape)
        out = self.out_lin(x_dict['document'])
        return out
    

