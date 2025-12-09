from .rdkit_desc import RDKit2D_DescriptorCalculator
from .hybrid import AcidHybridDescriptorCalculator, LigandHybridDescriptorCalculator


Desc_Calculators_Dict = {
    "2drdkit": RDKit2D_DescriptorCalculator,
    "acid_hybrid": AcidHybridDescriptorCalculator,
    "ligand_hybrid": LigandHybridDescriptorCalculator
}
