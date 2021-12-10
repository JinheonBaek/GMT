import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp

from torch_geometric.utils import to_dense_batch

from models.layers import SAB, ISAB, PMA
from models.layers import GCNConv_for_OGB, GINConv_for_OGB

from ogb.graphproppred.mol_encoder import AtomEncoder

from math import ceil

class GraphRepresentation(torch.nn.Module):

    def __init__(self, args):

        super(GraphRepresentation, self).__init__()

        self.args = args
        self.num_features = args.num_features
        self.nhid = args.num_hidden
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout

    def get_convs(self):

        convs = nn.ModuleList()

        _input_dim = self.num_features
        _output_dim = self.nhid

        for _ in range(self.args.num_convs):

            if self.args.conv == 'GCN':
            
                conv = GCNConv(_input_dim, _output_dim)

            elif self.args.conv == 'GIN':

                conv = GINConv(
                    nn.Sequential(
                        nn.Linear(_input_dim, _output_dim),
                        nn.ReLU(),
                        nn.Linear(_output_dim, _output_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(_output_dim),
                ), train_eps=False)

            convs.append(conv)

            _input_dim = _output_dim
            _output_dim = _output_dim

        return convs

    def get_pools(self):

        pools = nn.ModuleList([gap])

        return pools

    def get_classifier(self):

        return nn.Sequential(
            nn.Linear(self.nhid, self.nhid),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.num_classes)
        )

class GraphMultisetTransformer(GraphRepresentation):

    def __init__(self, args):

        super(GraphMultisetTransformer, self).__init__(args)

        self.ln = args.ln
        self.num_heads = args.num_heads
        self.cluster = args.cluster

        self.model_sequence = args.model_string.split('-')

        self.convs = self.get_convs()
        self.pools = self.get_pools()
        self.classifier = self.get_classifier()

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # For Graph Convolution Network
        xs = []

        for _ in range(self.args.num_convs):

            x = F.relu(self.convs[_](x, edge_index))
            xs.append(x)

        # For jumping knowledge scheme
        x = torch.cat(xs, dim=1)

        # For Graph Multiset Transformer
        for _index, _model_str in enumerate(self.model_sequence):

            if _index == 0:

                batch_x, mask = to_dense_batch(x, batch)

                extended_attention_mask = mask.unsqueeze(1)
                extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

            if _model_str == 'GMPool_G':

                batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask, graph=(x, edge_index, batch))

            else:

                batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask)

            extended_attention_mask = None

        batch_x = self.pools[len(self.model_sequence)](batch_x)
        x = batch_x.squeeze(1)

        # For Classification
        x = self.classifier(x)

        return F.log_softmax(x, dim=-1)

    def get_pools(self, _input_dim=None, reconstruction=False):

        pools = nn.ModuleList()

        _input_dim = self.nhid * self.args.num_convs if _input_dim is None else _input_dim
        _output_dim = self.nhid
        _num_nodes = ceil(self.pooling_ratio * self.args.avg_num_nodes)

        for _index, _model_str in enumerate(self.model_sequence):

            if (_index == len(self.model_sequence) - 1) and (reconstruction == False):
                
                _num_nodes = 1

            if _model_str == 'GMPool_G':

                pools.append(
                    PMA(_input_dim, self.num_heads, _num_nodes, ln=self.ln, cluster=self.cluster, mab_conv=self.args.mab_conv)
                )

                _num_nodes = ceil(self.pooling_ratio * _num_nodes)

            elif _model_str == 'GMPool_I':

                pools.append(
                    PMA(_input_dim, self.num_heads, _num_nodes, ln=self.ln, cluster=self.cluster, mab_conv=None)
                )

                _num_nodes = ceil(self.pooling_ratio * _num_nodes)

            elif _model_str == 'SelfAtt':

                pools.append(
                    SAB(_input_dim, _output_dim, self.num_heads, ln=self.ln, cluster=self.cluster)
                )

                _input_dim = _output_dim
                _output_dim = _output_dim

            else:

                raise ValueError("Model Name in Model String <{}> is Unknown".format(_model_str))

        pools.append(nn.Linear(_input_dim, self.nhid))

        return pools

class GraphMultisetTransformer_for_OGB(GraphMultisetTransformer):

    def __init__(self, args):

        super(GraphMultisetTransformer_for_OGB, self).__init__(args)

        self.atom_encoder = AtomEncoder(self.nhid)
        self.convs = self.get_convs()

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.atom_encoder(x)

        # For Graph Convolution Network
        xs = []

        for _ in range(self.args.num_convs):

            x = F.relu(self.convs[_](x, edge_index, edge_attr))
            xs.append(x)

        # For jumping knowledge scheme
        x = torch.cat(xs, dim=1)

        # For Graph Multiset Transformer
        for _index, _model_str in enumerate(self.model_sequence):

            if _index == 0:

                batch_x, mask = to_dense_batch(x, batch)

                extended_attention_mask = mask.unsqueeze(1)
                extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

            if _model_str == 'GMPool_G':

                batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask, graph=(x, edge_index, batch))

            else:

                batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask)

            extended_attention_mask = None

        batch_x = self.pools[len(self.model_sequence)](batch_x)
        x = batch_x.squeeze(1)

        # For Classification
        x = self.classifier(x)

        return x

    def get_convs(self):

        convs = nn.ModuleList()

        for _ in range(self.args.num_convs):

            if self.args.conv == 'GCN':
            
                conv = GCNConv_for_OGB(self.nhid)

            elif self.args.conv == 'GIN':

                conv = GINConv_for_OGB(self.nhid)

            convs.append(conv)

        return convs

class GraphMultisetTransformer_for_Recon(GraphMultisetTransformer):

    def __init__(self, args):

        super(GraphMultisetTransformer_for_Recon, self).__init__(args)

        self.pools = self.get_pools(_input_dim=self.nhid, reconstruction=True)
        self.unconvs = self.get_unconvs()
        
    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encoder
        for _ in range(self.args.num_convs):

            x = F.relu(self.convs[_](x, edge_index))
            
        # Pooling
        for _index, _model_str in enumerate(self.model_sequence):

            if _index == 0:

                batch_x, mask = to_dense_batch(x, batch)

                extended_attention_mask = mask.unsqueeze(1)
                extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

            if _model_str == 'GMPool_G':

                batch_x, attn = self.pools[_index](batch_x, attention_mask=extended_attention_mask, graph=(x, edge_index, batch), return_attn=True)

            else:

                batch_x, attn = self.pools[_index](batch_x, attention_mask=extended_attention_mask, return_attn=True)

            extended_attention_mask = None

        # Decoder
        batch_x = self.pools[len(self.model_sequence)](batch_x)
        x = torch.bmm(attn.transpose(1, 2), batch_x)
        
        x = x[mask]

        for _ in range(self.args.num_unconvs):

            x = self.unconvs[_](x, edge_index)

            if _ < (self.args.num_unconvs - 1):
                x = F.relu(x)

        return x

    def get_unconvs(self):

        unconvs = nn.ModuleList()

        _input_dim = self.nhid
        _output_dim = self.nhid

        for _ in range(self.args.num_unconvs):

            if _ == (self.args.num_unconvs - 1):

                _output_dim = self.num_features

            if self.args.conv == 'GCN':
            
                conv = GCNConv(_input_dim, _output_dim)

            elif self.args.conv == 'GIN':

                conv = GINConv(
                    nn.Sequential(
                        nn.Linear(_input_dim, _input_dim),
                        nn.ReLU(),
                        nn.Linear(_input_dim, _output_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(_output_dim),
                ), train_eps=False)

            unconvs.append(conv)

        return unconvs