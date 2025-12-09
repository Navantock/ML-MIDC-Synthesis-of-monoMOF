import torch
from torch_geometric.data import Data


class SynMOF_GraphData(Data):
    def __init__(self, 
                 acid_x=None,
                 acid_edge_x=None,
                 acid_edge_index=None,
                 ligand_x=None,
                 ligand_edge_x=None,
                 ligand_edge_index=None,
                 y=None,
                 **kwargs):
        
        super(SynMOF_GraphData, self).__init__(acid_x=acid_x, acid_ex=acid_edge_x, acid_edge_index=acid_edge_index, ligand_x=ligand_x, ligand_ex=ligand_edge_x, ligand_edge_index=ligand_edge_index, y=y, **kwargs)

    @property
    def x(self) -> torch.Tensor:
        return torch.cat([self.acid_x, self.ligand_x], dim=0)
    
    @property
    def edge_x(self) -> torch.Tensor:
        return torch.cat([self.acid_edge_x, self.ligand_edge_x], dim=0)
    
    @property
    def edge_index(self) -> torch.Tensor:
        return torch.cat([self.acid_edge_index, self.ligand_edge_index + self.acid_x.size(0)], dim=1)
    
    @property
    def tag(self) -> torch.Tensor:
        return torch.cat([torch.zeros_like(self.acid_x[:,0]), torch.ones_like(self.ligand_x[:,0])], dim=0)

    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.x.size(0)
        else:
            return super(SynMOF_GraphData, self).__inc__(key, value)
        
class SynMOF_PathGraphData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, norm=None, face=None, **kwargs):
        super(SynMOF_PathGraphData, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, norm=norm, face=face, **kwargs)

    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.x.size(0)
        else:
            return super(SynMOF_PathGraphData, self).__inc__(key, value)