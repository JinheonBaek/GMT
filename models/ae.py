import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp

from torch_geometric.utils import to_dense_batch

from models.layers import SAB, ISAB, PMA

from math import ceil

class GraphMultisetTransformer(torch.nn.Module):
    def __init__(self, args, n_nodes):
        super(GraphMultisetTransformer, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.num_hidden
        self.ln = args.ln

        # Encoder
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        # Pooling
        pool_nodes = ceil(n_nodes * args.pooling_ratio)
        self.pool = PMA(self.nhid, 1, pool_nodes, ln=self.ln, cluster=True, mab_conv="GCN")
        self.lin = nn.Linear(self.nhid, self.nhid)
        
        # Decoder
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.num_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Pooling
        batch_x, mask = to_dense_batch(x, batch)
        extended_attention_mask = mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
        
        x, attn = self.pool(batch_x, extended_attention_mask, graph=(x, edge_index, batch), return_attn=True)
        x = self.lin(x)
        k = x.shape[1]

        # Upsampling
        x_out = torch.bmm(attn.transpose(1, 2), x)
        x_out = x_out.squeeze(0)

        x_out = x_out[mask]
        x = F.relu(self.conv3(x_out, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = self.conv5(x, edge_index)

        return x
