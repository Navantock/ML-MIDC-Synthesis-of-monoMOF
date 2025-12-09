from ._base_calculator import BaseDescriptorCalculator
from rdkit.Chem import Mol, AllChem
from rdkit import Chem
from rdkit.Chem import Fragments, Descriptors, Lipinski

from typing import Sequence


Element_Dict = { 1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I' }
Metal_Set = {'Na', 'Mg', 'K', 'Ca', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Pd', 'Ir', 'Pt'}
DEFAULT_TOPO_USED_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


def _find_COOH(mol: Mol):
    matches_COOH = mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)O'))
    return matches_COOH


class TopoDescCalculator(BaseDescriptorCalculator):
    def __init__(self, used_indices: Sequence[int] = DEFAULT_TOPO_USED_INDICES, **kwargs):
        super().__init__()
        self.used_indices = used_indices
        self.total_length = 30

    def calculate(self, mol: Mol, **kwargs):
        mol = Chem.AddHs(mol, addCoords=True)
        matches_COOH = _find_COOH(mol)
        C_indices = [a_idx for match in matches_COOH for a_idx in match if mol.GetAtomWithIdx(a_idx).GetSymbol() == 'C']
        O_indices = [a_idx for match in matches_COOH for a_idx in match if mol.GetAtomWithIdx(a_idx).GetSymbol() == 'O']

        desc = [0] * self.total_length

        # MolWeight
        desc[0] = Descriptors.MolWt(mol)
        # Total Formal Charge
        desc[1] = Chem.GetFormalCharge(mol)
        # Number of Atoms
        desc[2] = mol.GetNumAtoms()
        # Number of Bonds
        desc[3] = mol.GetNumBonds()
        # Number of Rings
        desc[4] = Chem.rdMolDescriptors.CalcNumRings(mol)
        # Number of Aromatic Rings
        desc[5] = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
        # Atoms in Rings
        desc[6] = len([1 for atom in mol.GetAtoms() if atom.IsInRing()])
        # Atoms in Aromatic
        desc[7] = len([1 for atom in mol.GetAtoms() if atom.GetIsAromatic()])
        # Bonds Coponjugated
        desc[8] = len([1 for bond in mol.GetBonds() if bond.GetIsConjugated()])
        # Bonds is Aromatic
        desc[9] = len([1 for bond in mol.GetBonds() if bond.GetIsAromatic()])
        # Number of Rotatable Bonds
        desc[10] = Lipinski.NumRotatableBonds(mol)
        # Number of H
        desc[11] = len([1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'H'])
        # Number of C
        desc[12] = len([1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])
        # Number of O
        desc[13] = len([1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O'])
        # Number of N
        desc[14] = len([1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N'])
        # Number of S
        desc[15] = len([1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S'])
        # Number of Halogens
        desc[16] = len([1 for atom in mol.GetAtoms() if atom.GetSymbol() in ['F', 'Cl', 'Br', 'I']])
        # Number of Metal Ions
        desc[17] = len([1 for atom in mol.GetAtoms() if atom.GetSymbol() in Metal_Set])
        # C sp2 ratio
        desc[18] = len([1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2]) / desc[12]
        # C sp3 ratio
        desc[19] = len([1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3]) / desc[12]
        # Number of Ar-OH
        desc[20] = Fragments.fr_Ar_OH(mol)
        # Number of Ar-N
        desc[21] = Fragments.fr_Ar_N(mol)
        # Number of Ar-NH
        desc[22] = Fragments.fr_Ar_NH(mol)
        # Number of Al-OH
        desc[23] = Fragments.fr_Al_OH(mol)
        # Number of Al-NH
        desc[24] = Fragments.fr_NH2(mol)
        # Number of Hydrogen Bond Donors
        desc[25] = Lipinski.NumHDonors(mol)
        # Number of Hydrogen Bond Acceptors
        desc[26] = Lipinski.NumHAcceptors(mol)

        # Calculate COOH Desc
        AllChem.ComputeGasteigerCharges(mol)
        # COOH C Gasteiger Charge
        desc[27] = float(mol.GetAtomWithIdx(C_indices[0]).GetProp('_GasteigerCharge'))
        # COOH O (=O) Gasteiger Charge and COOH O (-OH) Gasteiger Charge
        for O_idx in O_indices[:2]:
            if mol.GetAtomWithIdx(O_idx).GetTotalNumHs(includeNeighbors=True) == 0:
                desc[28] = float(mol.GetAtomWithIdx(O_idx).GetProp('_GasteigerCharge'))
            else:
                desc[29] = float(mol.GetAtomWithIdx(O_idx).GetProp('_GasteigerCharge'))

        return [desc[i] for i in self.used_indices]

    @property
    def descriptor_summaries(self):
        summaries = [
            'MolWeight', # 0
            'Total Formal Charge', # 1
            'Number of Atoms', # 2
            'Number of Bonds', # 3
            'Number of Rings', # 4
            'Number of Aromatic Rings', # 5
            'Atoms in Rings', # 6
            'Atoms in Aromatic', # 7
            'Bonds Conjugated', # 8
            'Bonds is Aromatic', # 9
            'Number of Rotatable Bonds', # 10
            'Number of H', # 11
            'Number of C', # 12
            'Number of O', # 13
            'Number of N', # 14
            'Number of S', # 15
            'Number of Halogens', # 16
            'Number of Metal Ions', # 17
            'C sp2 ratio', # 18
            'C sp3 ratio', # 19
            'Number of Ar-OH', # 20
            'Number of Ar-N', # 21
            'Number of Ar-NH', # 22
            'Number of Al-OH', # 23
            'Number of NH2', # 24
            'Number of Hydrogen Bond Donors', # 25
            'Number of Hydrogen Bond Acceptors', # 26
            'COOH C Gasteiger Charge', # 27
            'COOH O (=O) Gasteiger Charge', # 28
            'COOH O (-OH) Gasteiger Charge', # 29
        ]
        return [summaries[i] for i in self.used_indices]

    @property
    def descriptor_names(self):
        return self.descriptor_summaries

    @property
    def descriptor_count(self):
        return len(self.descriptor_names)