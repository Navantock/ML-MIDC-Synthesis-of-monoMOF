from ._base_calculator import BaseDescriptorCalculator

from rdkit.Chem import Mol
from rdkit.Chem.Descriptors import _descList
from rdkit.ML.Descriptors import MoleculeDescriptors

from typing import Sequence


Default_RDKit2D_Descriptors = ['MolLogP', 'TPSA', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n', 'Kappa1', 'Kappa2', 'Kappa3']
Selected_Acid_RDKit2D_Descriptors = ['MolLogP', 'Kappa3']
#['MaxAbsPartialCharge', 'MaxPartialCharge', 'PEOE_VSA6', 'VSA_EState6', 'FpDensityMorgan1', 'MaxEStateIndex', 'ExactMolWt', 'BCUT2D_LOGPHI', 'SMR_VSA1', 'PEOE_VSA7']
Selected_Ligand_RDKit2D_Descriptors = ['MolLogP', 'Chi4v', 'Kappa3']
#['MolLogP', 'EState_VSA9', 'NumAromaticCarbocycles', 'Chi4n', 'VSA_EState5', 'Chi1v', 'VSA_EState4', 'VSA_EState3', 'MaxAbsEStateIndex', 'SMR_VSA7']


class RDKit2D_DescriptorCalculator(BaseDescriptorCalculator):
    def __init__(self, **kwargs):
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in _descList])
        super().__init__()
    
    def calculate(self, mol: Mol, **kwargs) -> Sequence[float|int]:
        descriptor = list(self.calculator.CalcDescriptors(mol))
        return descriptor
    
    @property
    def descriptor_names(self) -> Sequence[str]:
        return [desc[0] for desc in _descList]
    
    @property
    def descriptor_summaries(self) -> Sequence[str]:
        return self.calculator.GetDescriptorSummaries()


class Selected_Acid_RDKit2D_DescriptorCalculator(BaseDescriptorCalculator):
    def __init__(self, rdkit_desc_list: Sequence[str] = Selected_Acid_RDKit2D_Descriptors, **kwargs):
        self.rdkit_desc_list = rdkit_desc_list
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(rdkit_desc_list)
        super().__init__()
    
    def calculate(self, mol: Mol, **kwargs) -> Sequence[float|int]:
        descriptor = list(self.calculator.CalcDescriptors(mol))
        return descriptor
    
    @property
    def descriptor_names(self) -> Sequence[str]:
        return self.rdkit_desc_list
    
    @property
    def descriptor_summaries(self) -> Sequence[str]:
        return self.calculator.GetDescriptorSummaries()

class Selected_Ligand_RDKit2D_DescriptorCalculator(BaseDescriptorCalculator):
    def __init__(self, rdkit_desc_list: Sequence[str] = Selected_Ligand_RDKit2D_Descriptors, **kwargs):
        self.rdkit_desc_list = rdkit_desc_list
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(rdkit_desc_list)
        super().__init__()
    
    def calculate(self, mol: Mol, **kwargs) -> Sequence[float|int]:
        descriptor = list(self.calculator.CalcDescriptors(mol))
        return descriptor
    
    @property
    def descriptor_names(self) -> Sequence[str]:
        return self.rdkit_desc_list
    
    @property
    def descriptor_summaries(self) -> Sequence[str]:
        return self.calculator.GetDescriptorSummaries()