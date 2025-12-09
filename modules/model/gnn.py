import torch.nn as nn
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, Subset
from torch_geometric.nn import GCNConv, GATConv

from typing import Dict

from .alnn.graphormer.model import Graphormer
from .alnn.gnn import GNN_Blocks


# use H C N O F S Cl Br I
_USED_ATOMIC_NUM = [1, 6, 7, 8, 9, 16, 17, 35, 53]


class Node_Embedding(nn.Module):
    # one hot encoding of atom
    def __init__(self, 
                 num_classes: int, 
                 embed_dim: int):
        super().__init__()
        self.embed = nn.Linear(num_classes, embed_dim)
    
    def forward(self, x):
        # TODO: check if one_hot is necessary
        return one_hot(x, self.num_classes).float()


class AL_GNN_Classifier(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 gnn_module_name: str,
                 gnn_configs: Dict,
                 use_acid_q: bool,
                 cat_aggr: bool,
                 gnn_output_dim: int,
                 ffn_hidden_dim: int):
        super().__init__()

        self.num_classes = num_classes
        self.gnn = None
        if gnn_module_name == 'Graphormer':
            self.gnn = Graphormer(**gnn_configs)
        elif self.gnn == 'GCN':
            self.gnn = GNN_Blocks(gnn_conv_type='GCN', **gnn_configs)
        elif self.gnn == 'GAT':
            self.gnn = GNN_Blocks(gnn_conv_type='GAT', **gnn_configs)
        else:
            raise ValueError(f"Invalid GNN module name: {gnn_module_name}")
        
        self.acid_q_embed = None
        if use_acid_q:
            self.acid_q_embed = nn.Sequential(
                nn.Linear(1, gnn_output_dim),
                nn.LeakyReLU()
            )
        
        self.ffn = nn.Sequential(
            nn.Linear(gnn_output_dim, ffn_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(ffn_hidden_dim, num_classes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, acid_x, acid_edge_index, ligand_x, ligand_edge_index, acid_q=None):
        # TODO: implement the forward pass
        return
    