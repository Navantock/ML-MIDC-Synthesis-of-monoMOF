from ._base_calculator import BaseDescriptorCalculator
from .rdkit_desc import Selected_Acid_RDKit2D_DescriptorCalculator, Selected_Ligand_RDKit2D_DescriptorCalculator, Default_RDKit2D_Descriptors, Selected_Acid_RDKit2D_Descriptors, Selected_Ligand_RDKit2D_Descriptors
from .geom_desc import GeomDescCalculator, LigandGeomDescCalculator
from .topo_desc import TopoDescCalculator, DEFAULT_TOPO_USED_INDICES
from .qc_desc import AcidGaussianDescCalculator, LigandGaussianDescCalculator

from rdkit.Chem import Mol

from typing import List


USE_SELECTED = True # Set to True to use selected descriptors, False to use all available descriptors

ACID_RDKIT_DESCS = Selected_Acid_RDKit2D_Descriptors if USE_SELECTED else Default_RDKit2D_Descriptors
ACID_TOPO_SELECTED_INDICES = [2, 10, 12, 13, 14, 25] if USE_SELECTED else DEFAULT_TOPO_USED_INDICES
ACID_GS_SELECTED_INDICES = [3, 6, 9, 12, 14] if USE_SELECTED else None

LIGAND_RDKIT_DESCS = Selected_Ligand_RDKit2D_Descriptors if USE_SELECTED else Default_RDKit2D_Descriptors
LIGAND_GEOM_SELECTED_INDICES = [0, 1, 2, 3] if USE_SELECTED else None
LIGAND_TOPO_SELECTED_INDICES = [5, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26] if USE_SELECTED else DEFAULT_TOPO_USED_INDICES
LIGAND_GS_SELECTED_INDICES = [0, 4, 7, 9, 13, 16, 19] if USE_SELECTED else None


class HybridDescriptorCalculator(BaseDescriptorCalculator):
    def __init__(self, **kwargs):
        super().__init__()
        self.calculators : List

    def calculate(self, mol: Mol, **kwargs):
        desc = []
        for calculator in self.calculators:
            desc.extend(calculator.calculate(mol))
        return desc

    @property
    def descriptor_summaries(self):
        summaries = []
        for calculator in self.calculators:
            summaries.extend(calculator.descriptor_summaries)
        return summaries

    @property
    def descriptor_names(self):
        names = []
        for calculator in self.calculators:
            names.extend(calculator.descriptor_names)
        return names
    
    @property
    def descriptor_count(self):
        return sum([calculator.descriptor_count for calculator in self.calculators])


class AcidHybridDescriptorCalculator(HybridDescriptorCalculator):
    def __init__(self, acid_sp_dir: str, acid_opt_dir: str, **kwargs):
        super().__init__()
        self.calculators = [
            Selected_Acid_RDKit2D_DescriptorCalculator(rdkit_desc_list=ACID_RDKIT_DESCS),
            #GeomDescCalculator(),
            TopoDescCalculator(used_indices=ACID_TOPO_SELECTED_INDICES),
            AcidGaussianDescCalculator(acid_sp_dir, acid_opt_dir, used_indices=ACID_GS_SELECTED_INDICES)
        ]


class LigandHybridDescriptorCalculator(HybridDescriptorCalculator):
    def __init__(self, ligand_sp_dir: str, ligand_opt_dir: str, **kwargs):
        super().__init__()
        self.calculators = [
            Selected_Ligand_RDKit2D_DescriptorCalculator(rdkit_desc_list=LIGAND_RDKIT_DESCS),
            TopoDescCalculator(used_indices=LIGAND_TOPO_SELECTED_INDICES),
            LigandGeomDescCalculator(used_indices=LIGAND_GEOM_SELECTED_INDICES, ligand_opt_dir=ligand_opt_dir),
            LigandGaussianDescCalculator(ligand_sp_dir, ligand_opt_dir, used_indices=LIGAND_GS_SELECTED_INDICES, disable_calc_charge2=USE_SELECTED)
        ]
