import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol, AllChem, rdMolTransforms, Descriptors
import ase.io

import logging
import os
from typing import Optional, Sequence

from ._base_calculator import BaseDescriptorCalculator
from .topo_desc import _find_COOH

H_BAR = 1.380649e-23



def point_distance_line(points: np.ndarray[np.ndarray], line_point1: np.ndarray, line_point2: np.ndarray):
    """
    Calculate the distance between some points and a line defined by two points
    :param points: np.array (N, 3)
    :param line_point1: np.array
    :param line_point2: np.array
    :return: float
    """
    vec1 = line_point2 - points
    vec2 = line_point1 - points
    return np.linalg.norm(np.cross(vec1, vec2), axis=1) / np.linalg.norm(line_point2 - line_point1)

def get_gyration_radius(coords: np.ndarray[np.ndarray], masses: np.ndarray, rot_ax_point1: np.ndarray, rot_ax_point2: np.ndarray):
    """
    Calculate the gyration radius of a set of points around a line defined by two points
    :param coords: np.array (N, 3)
    :param masses: np.array (N,)
    :param rot_ax_point1: np.array (3,)
    :param rot_ax_point2: np.array (3,)
    :return: float
    """
    distance_to_rot_ax = np.array([point_distance_line(coords, rot_ax_point1, rot_ax_point2)])
    return np.sqrt(np.sum(masses * distance_to_rot_ax ** 2) / np.sum(masses))


class GeomDescCalculator(BaseDescriptorCalculator):
    def __init__(self, **kwargs):
        super().__init__()

    def calculate(self, mol: Mol, embed_molecule: bool = True, **kwargs):
        mol = Chem.AddHs(mol, addCoords=True)
        if embed_molecule:
            try:
                AllChem.EmbedMolecule(mol)
            except:
                logging.warning("Embedding failed, using 2D coordinates")
        AllChem.MMFFOptimizeMolecule(mol)

        # Get the coordinates of the atoms
        principal_axes, principal_moments = rdMolTransforms.ComputePrincipalAxesAndMoments(mol.GetConformer(), ignoreHs=False)
        mass = Descriptors.MolWt(mol)
        desc = [np.sqrt(principal_moment / mass) for principal_moment in principal_moments]
        return desc

    @property
    def descriptor_summaries(self):
        return [
            "Principal Axis Gyration Radius X", 
            "Principal Axis Gyration Radius Y", 
            "Principal Axis Gyration Radius Z",
        ]

    @property
    def descriptor_names(self):
        return self.descriptor_summaries

    @property
    def descriptor_count(self):
        return len(self.descriptor_names)

class LigandGeomDescCalculator(GeomDescCalculator):
    def __init__(self, used_indices: Optional[Sequence[int]] = None, **kwargs):
        if "ligand_opt_dir" in kwargs:
            self.ligand_opt_dir = kwargs["ligand_opt_dir"]
        
        super().__init__()
        self.total_length = 8
        self.used_indices = list(range(self.total_length)) if used_indices is None else used_indices

    def calculate(self, mol: Mol, **kwargs):
        embed_mol = False if mol.HasProp("Mult") else True
        base_calculator = GeomDescCalculator()
        desc = [0.] * (self.total_length - base_calculator.descriptor_count)
        base_desc = base_calculator.calculate(mol, embed_molecule=embed_mol)
        desc.extend(base_desc)

        mol = Chem.AddHs(mol, addCoords=True)
        matches_COOH = _find_COOH(mol)
        C_indices = [a_idx for match in matches_COOH for a_idx in match if mol.GetAtomWithIdx(a_idx).GetSymbol() == 'C']
        O_indices = [a_idx for match in matches_COOH for a_idx in match if mol.GetAtomWithIdx(a_idx).GetSymbol() == 'O']

        # Calculate the distance between the two C atoms in the COOH group
        assert len(C_indices) >= 2, "2 COOH group not found"

        coords = None
        if getattr(self, "ligand_opt_dir", None) is not None:
            # disable ase log
            logging.getLogger('ase').setLevel(logging.ERROR)
            coords = ase.io.read(os.path.join(self.ligand_opt_dir, str(mol.GetProp("Name")), "output", f"{mol.GetProp("Name")}_gas_0.out")).get_positions()
        else:
            coords = mol.GetConformer().GetPositions()
            
        C_coords = np.array([coords[C_idx] for C_idx in C_indices])
        dist_CC = np.linalg.norm(np.linalg.norm(C_coords[0] - C_coords[1]))
        desc[0] = dist_CC

        center = np.mean(coords, axis=0)
        # dis center 2 CC
        desc[1] = point_distance_line(center.reshape(1, 3), C_coords[0], C_coords[1])[0]
        # dis center to C-C vertical plane
        center_CC = (C_coords[0] + C_coords[1]) / 2
        desc[2] = np.sqrt(np.linalg.norm(center - center_CC) ** 2 - desc[1] ** 2)

        # Get all atoms weight (including H)
        atoms_masses = np.array([Chem.GetPeriodicTable().GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetAtoms()])
        # delete C in COOH group
        coords = np.delete(coords, C_indices, axis=0)
        atoms_masses = np.delete(atoms_masses, C_indices)
        desc[3] = get_gyration_radius(coords, atoms_masses, C_coords[0], C_coords[1])

        # COOH COOH Dihedral
        dihedral = rdMolTransforms.GetDihedralRad(mol.GetConformer(), O_indices[0], C_indices[0], C_indices[1], O_indices[1])
        if dihedral > np.pi / 2:
            dihedral = np.pi - dihedral
        desc[4] = np.cos(dihedral)

        return [desc[i] for i in self.used_indices]

    @property
    def descriptor_summaries(self):
        summaries = [
            "Distance COOH C-C", # 0
            "Distance C-C to Center", # 1
            "Distance Center to C-C Center Plane", # 2
            "COOH C-C Gyration Radius", # 3
            "COOH-COOH Dihedral Cosine", # 4
            ] + super().descriptor_summaries
        return [summaries[i] for i in self.used_indices]

    @property
    def descriptor_names(self):
        return self.descriptor_summaries

    @property
    def descriptor_count(self):
        return len(self.descriptor_names)