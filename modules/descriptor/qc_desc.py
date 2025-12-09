from ._base_calculator import BaseDescriptorCalculator
from .topo_desc import _find_COOH

from rdkit.Chem import Mol
import numpy as np
import pandas as pd
from typing import Optional, Sequence

import ase.io

import os

eV2kcal_mol = 23.0605
kcal_mol2J_mol = 4183.9954
eV2J_mol = 96485.3
hartree2kcal_mol = 627.5094740631
G_PROTON_GAS = -6.28 # kcal/mol
DG_PROTON_SOLV_WATER = -264.61 # kcal/mol
DG_PROTON_SOLV_DMF = -249.75 # kcal/mol -10.83eV from ref
IDEAL_GAS_CONSTANT = 8.31446261815324

REF_ACID = 'BA'
ACID_NOT_REF_SET = {}
ACID_SPEC_REF_SET = {}
REF_ACID_pKa = {
    'BA': [4.20, 12.28],
}
#'BA-246-3F': [2.28, 11.26]
ACID_ANNOTATE_INFO_PATH = "./dataset/annotate/Acid_Info.csv"

REF_LIGAND = '45'
LIGAND_NOT_REF_SET = {}
LIGAND_SPEC_REF_SET = {
    '12': '27',
    '17': '27',
    '27': '27',
    '28': '27',
    '29': '27',
}
REF_LIGAND_pKa = {
    '45': [3.80, 11.46],
    '27': [2.42, 8.54]
}
LIGAND_ANNOTATE_INFO_PATH = "./dataset/annotate/Ligand_Info.csv"

def read_free_energy_from_gs_output(output_file: str):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'Sum of electronic and thermal Free Energies=' in line:
            return float(lines[i].strip().split()[-1]) * hartree2kcal_mol # hartree 2 kcal/mol
    raise ValueError("Free energy not found in output file")

def read_free_energy_from_cbs_gs_output(output_file: str):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'CBS-QB3 Free Energy=' in line:
            return float(lines[i].strip().split()[-1]) * hartree2kcal_mol # hartree 2 kcal/mol
    raise ValueError("Free energy not found in output file")

def read_free_energy_correction_from_gs_output(output_file: str):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'Thermal correction to Gibbs Free Energy=' in line:
            return float(lines[i].strip().split()[-1]) * hartree2kcal_mol # hartree 2 kcal/mol
    raise ValueError("Free energy correction not found in output file")

def _get_free_energy(str_charge: str, mol_name: str, sp_dir: str, opt_dir: str):
    if os.path.exists(os.path.join(sp_dir, mol_name, 'output', '{}_gas_{}_gc.out'.format(mol_name, str_charge))):
        return read_free_energy_from_cbs_gs_output(os.path.join(sp_dir, mol_name, 'output', '{}_gas_{}_gc.out'.format(mol_name, str_charge)))
    else:
        # read correction from opt
        fe_correction = read_free_energy_correction_from_gs_output(os.path.join(opt_dir, mol_name, 'output', '{}_gas_{}.out'.format(mol_name, str_charge)))
        e_basic = ase.io.read(os.path.join(sp_dir, mol_name, 'output', '{}_gas_{}_acc.out'.format(mol_name, str_charge)), index=-1).get_potential_energy() * eV2kcal_mol
        return e_basic + fe_correction

def _get_ref_dG(sp_dir: str, opt_dir: str, ref_mol_name: str):
    atoms = ase.io.read(os.path.join(sp_dir, ref_mol_name, 'output', '{}_gas_0.out'.format(ref_mol_name)), index=-1)
    e_gas_0 = atoms.get_potential_energy() * eV2kcal_mol
    atoms = ase.io.read(os.path.join(sp_dir, ref_mol_name, 'output', '{}_gas_1.out'.format(ref_mol_name)), index=-1)
    e_gas_1 = atoms.get_potential_energy() * eV2kcal_mol
    atoms = ase.io.read(os.path.join(sp_dir, ref_mol_name, 'output', '{}_water_0.out'.format(ref_mol_name)), index=-1)
    e_water_0 = atoms.get_potential_energy() * eV2kcal_mol
    atoms = ase.io.read(os.path.join(sp_dir, ref_mol_name, 'output', '{}_water_1.out'.format(ref_mol_name)), index=-1)
    e_water_1 = atoms.get_potential_energy() * eV2kcal_mol
    atoms = ase.io.read(os.path.join(sp_dir, ref_mol_name, 'output', '{}_dmf_0.out'.format(ref_mol_name)), index=-1)
    e_dmf_0 = atoms.get_potential_energy() * eV2kcal_mol
    atoms = ase.io.read(os.path.join(sp_dir, ref_mol_name, 'output', '{}_dmf_1.out'.format(ref_mol_name)), index=-1)
    e_dmf_1 = atoms.get_potential_energy() * eV2kcal_mol

    ref_g0_fe = _get_free_energy('0', ref_mol_name, sp_dir, opt_dir)
    ref_g1_fe = _get_free_energy('1', ref_mol_name, sp_dir, opt_dir)

    ref_d_G_water = ref_g0_fe - ref_g1_fe - e_water_1 + e_gas_1 - e_gas_0 + e_water_0
    ref_d_G_dmf = ref_g0_fe - ref_g1_fe - e_dmf_1 + e_gas_1 - e_gas_0 + e_dmf_0

    return ref_d_G_water, ref_d_G_dmf

def read_enthalpy_from_cbs_gs_output(output_file: str):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'CBS-QB3 Enthalpy=' in line:
            return float(lines[i].strip().split()[2]) * hartree2kcal_mol # hartree 2 kcal/mol
    raise ValueError("Free energy not found in output file")

def read_enthalpy_correction_from_gs_output(output_file: str):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'Thermal correction to Enthalpy=' in line:
            return float(lines[i].strip().split()[-1]) * hartree2kcal_mol # hartree 2 kcal/mol
    raise ValueError("Free energy correction not found in output file")

def _get_enthalpy(str_charge: str, mol_name: str, sp_dir: str, opt_dir: str):
    if os.path.exists(os.path.join(sp_dir, mol_name, 'output', '{}_gas_{}_gc.out'.format(mol_name, str_charge))):
        return read_enthalpy_from_cbs_gs_output(os.path.join(sp_dir, mol_name, 'output', '{}_gas_{}_gc.out'.format(mol_name, str_charge)))
    else:
        # read correction from opt
        fe_correction = read_enthalpy_correction_from_gs_output(os.path.join(opt_dir, mol_name, 'output', '{}_gas_{}.out'.format(mol_name, str_charge)))
        e_basic = ase.io.read(os.path.join(sp_dir, mol_name, 'output', '{}_gas_{}_acc.out'.format(mol_name, str_charge)), index=-1).get_potential_energy() * eV2kcal_mol
        return e_basic + fe_correction

def read_entropy_from_gs_output(output_file: str):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'E (Thermal)' in line:
            s_total = float(lines[i+2].strip().split()[-1])
            s_elec = float(lines[i+3].strip().split()[-1])
            s_trans = float(lines[i+4].strip().split()[-1])
            s_rot = float(lines[i+5].strip().split()[-1])
            s_vib = float(lines[i+6].strip().split()[-1])
            return s_total, s_elec, s_trans, s_rot, s_vib
    raise ValueError("Entropy not found in output file")

def read_dipole_from_gs_output(output_file: str):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    dipole = None
    for i, line in enumerate(lines):
        if 'Dipole moment' in line:
            dipole = float(lines[i+1].strip().split()[-1])
    assert dipole is not None, "Dipole moment not found in output file"
    return dipole

def _get_dipole(str_charge: str, mol_name: str, sp_dir: str):
    if os.path.exists(os.path.join(sp_dir, mol_name, 'output', '{}_gas_{}_gc.out'.format(mol_name, str_charge))):
        return read_dipole_from_gs_output(os.path.join(sp_dir, mol_name, 'output', '{}_gas_{}_gc.out'.format(mol_name, str_charge)))
    else:
        return read_dipole_from_gs_output(os.path.join(sp_dir, mol_name, 'output', '{}_gas_{}_acc.out'.format(mol_name, str_charge)))

class AcidGaussianDescCalculator(BaseDescriptorCalculator):
    def __init__(self, acid_sp_dir: str, acid_opt_dir: str, used_indices: Optional[Sequence[int]] = None, use_ref: bool = True, use_annotate: bool = False, **kwargs):
        self.acid_sp_dir = acid_sp_dir
        self.acid_opt_dir = acid_opt_dir
        super().__init__()
        self.total_length = 15
        self.used_indices = list(range(self.total_length)) if used_indices is None else used_indices
        self.use_ref = use_ref
        self.use_annotate = use_annotate

        if use_ref:
            self.ref_d_G_water, self.ref_d_G_dmf = _get_ref_dG(acid_sp_dir, acid_opt_dir, REF_ACID)
        else:
            self.ref_d_G_water, self.ref_d_G_dmf = None, None

        if use_annotate:
            self.annotate_info = pd.read_csv(ACID_ANNOTATE_INFO_PATH, index_col=0)

    def calculate(self, mol: Mol, **kwargs):
        mol_name = mol.GetProp('Name')
        matches_COOH = _find_COOH(mol)
        O_indices = [a_idx for match in matches_COOH for a_idx in match if mol.GetAtomWithIdx(a_idx).GetSymbol() == 'O']
        assert len(O_indices) == 2, "Required 2 O atoms in COOH group, but found {}".format(len(O_indices))
        acid_join_path = os.path.join(self.acid_sp_dir, mol_name, 'output')
        desc = [0] * self.total_length
        # Gas output
        atoms = ase.io.read(os.path.join(acid_join_path, '{}_gas_0.out'.format(mol_name)), index=-1)
        """ charges = atoms.get_charges()
        for O_idx in O_indices[:2]:
            if mol.GetAtomWithIdx(O_idx).GetTotalNumHs(includeNeighbors=True) == 0:
                desc[0] = charges[O_idx]
            else:
                desc[1] = charges[O_idx] """
        e_gas_0 = atoms.get_potential_energy() * eV2kcal_mol
        #desc[3] = atoms.get_potential_energy() * eV2kcal_mol
        atoms = ase.io.read(os.path.join(acid_join_path, '{}_gas_1.out'.format(mol_name)), index=-1)
        e_gas_1 = atoms.get_potential_energy() * eV2kcal_mol
        #desc[4] = atoms.get_potential_energy() * eV2kcal_mol
        desc[0] = atoms.get_charges()[O_indices[0]]
        # Solvation output
        atoms = ase.io.read(os.path.join(acid_join_path, '{}_water_0.out'.format(mol_name)), index=-1)
        desc[1] = atoms.get_potential_energy() * eV2kcal_mol- e_gas_0
        atoms = ase.io.read(os.path.join(acid_join_path, '{}_water_1.out'.format(mol_name)), index=-1)
        desc[2] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_1
        atoms = ase.io.read(os.path.join(acid_join_path, '{}_dmf_0.out'.format(mol_name)), index=-1)
        desc[3] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_0
        atoms = ase.io.read(os.path.join(acid_join_path, '{}_dmf_1.out'.format(mol_name)), index=-1)
        desc[4] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_1
        # pKa
        g0_fe = _get_free_energy('0', mol_name, self.acid_sp_dir, self.acid_opt_dir)
        g1_fe = _get_free_energy('1', mol_name, self.acid_sp_dir, self.acid_opt_dir)
        if not self.use_ref or mol_name in ACID_NOT_REF_SET:
            d_G = g1_fe + G_PROTON_GAS - g0_fe + DG_PROTON_SOLV_WATER + desc[2] - desc[1] + 1.89
            desc[5] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10))
            d_G = g1_fe + G_PROTON_GAS - g0_fe + DG_PROTON_SOLV_DMF + desc[4] - desc[3] + 1.89
            desc[6] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10))
        elif mol_name in ACID_SPEC_REF_SET:
            d_G_water, d_G_dmf = _get_ref_dG(self.acid_sp_dir, self.acid_opt_dir, ACID_SPEC_REF_SET[mol_name])
            d_G = g1_fe - g0_fe + d_G_water + desc[2] - desc[1]
            desc[5] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_ACID_pKa[ACID_SPEC_REF_SET[mol_name]][0]
            d_G = g1_fe - g0_fe + d_G_dmf + desc[4] - desc[3]
            desc[6] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_ACID_pKa[ACID_SPEC_REF_SET[mol_name]][1]
        else:
            d_G = g1_fe - g0_fe + self.ref_d_G_water + desc[2] - desc[1]
            desc[5] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_ACID_pKa[REF_ACID][0]
            d_G = g1_fe - g0_fe + self.ref_d_G_dmf + desc[4] - desc[3]
            desc[6] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_ACID_pKa[REF_ACID][1]
        
        if self.use_annotate and mol_name in self.annotate_info.index:
            if 'pKa(Water)' in self.annotate_info.columns and not np.isnan(self.annotate_info.loc[mol_name, 'pKa(Water)']):
                desc[5] = self.annotate_info.loc[mol_name, 'pKa(Water)']
            if 'pKa(DMF)' in self.annotate_info.columns and not np.isnan(self.annotate_info.loc[mol_name, 'pKa(DMF)']):
                desc[6] = self.annotate_info.loc[mol_name, 'pKa(DMF)']

        desc[7], desc[8], desc[9], desc[10], desc[11] = read_entropy_from_gs_output(os.path.join(self.acid_opt_dir, mol_name, 'output', '{}_dmf_0.out'.format(mol_name)))
        
        desc[12] = _get_enthalpy('0', mol_name, self.acid_sp_dir, self.acid_opt_dir) + desc[3] - 1.89
        desc[13] = _get_enthalpy('1', mol_name, self.acid_sp_dir, self.acid_opt_dir) + desc[4] - 1.89

        desc[14] = _get_dipole('0', mol_name, self.acid_sp_dir)

        return [desc[i] for i in self.used_indices]

    @property
    def descriptor_summaries(self):
        summaries = [
            "COO- O gas Mulliken charge", # 0
            "SolFE water 0 eV", # 1
            "SolFE water -1 eV", # 2
            "SolFE dmf 0 eV", # 3
            "SolFE dmf -1 eV", # 4
            "pKa(Water)", # 5
            "pKa(DMF)", # 6
            "S(Total)", # 7
            "S(Elec)", # 8
            "S(Trans)", # 9
            "S(Rot)", # 10
            "S(Vib)", # 11
            "H(0)", # 12
            "H(-1)", # 13
            "Dipole Moment", # 14
        ]
        return [summaries[i] for i in self.used_indices]
    
    @property
    def descriptor_names(self):
        return self.descriptor_summaries

    @property
    def descriptor_count(self):
        return len(self.descriptor_names)


class LigandGaussianDescCalculator(BaseDescriptorCalculator):
    def __init__(self, ligand_sp_dir: str, ligand_opt_dir: str, used_indices: Optional[Sequence[int]] = None, use_ref: bool = True, use_annotate: bool = True, disable_calc_charge2: bool = True,**kwargs):
        self.ligand_sp_dir = ligand_sp_dir
        self.ligand_opt_dir = ligand_opt_dir
        super().__init__()
        self.total_length = 20
        self.used_indices = list(range(self.total_length)) if used_indices is None else used_indices
        self.use_ref = use_ref
        self.disable_calc_charge2 = disable_calc_charge2
        self.use_annotate = use_annotate

        if use_ref:
            self.ref_d_G_water, self.ref_d_G_dmf = _get_ref_dG(ligand_sp_dir, ligand_opt_dir, REF_LIGAND)
        else:
            self.ref_d_G_water, self.ref_d_G_dmf  = None, None
        
        if use_annotate:
            self.annotate_info = pd.read_csv(LIGAND_ANNOTATE_INFO_PATH, index_col=0)
        
    def calculate(self, mol: Mol, **kwargs):
        if self.disable_calc_charge2:
            mol_name = mol.GetProp('Name')
            matches_COOH = _find_COOH(mol)
            C_indices = [a_idx for match in matches_COOH for a_idx in match if mol.GetAtomWithIdx(a_idx).GetSymbol() == 'C']
            O_indices = [a_idx for match in matches_COOH for a_idx in match if mol.GetAtomWithIdx(a_idx).GetSymbol() == 'O']
            assert len(C_indices) == 2, "Required 2 C atom in COOH group, but found {}".format(len(C_indices))
            assert len(O_indices) == 4, "Required 4 O atoms in COOH group, but found {}".format(len(O_indices))
            ligand_join_path = os.path.join(self.ligand_sp_dir, mol_name, 'output')
            desc = [0] * self.total_length
            # Gas output
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_gas_0.out'.format(mol_name)), index=-1)
            e_gas_0 = atoms.get_potential_energy() * eV2kcal_mol
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_gas_1.out'.format(mol_name)), index=-1)
            e_gas_1 = atoms.get_potential_energy() * eV2kcal_mol
            desc[0] = np.mean(atoms.get_charges()[O_indices])
            # Solvation output
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_water_0.out'.format(mol_name)), index=-1)
            desc[1] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_0
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_water_1.out'.format(mol_name)), index=-1)
            desc[2] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_1
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_dmf_0.out'.format(mol_name)), index=-1)
            desc[4] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_0
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_dmf_1.out'.format(mol_name)), index=-1)
            desc[5] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_1
            # pKa
            g0_fe = _get_free_energy('0', mol_name, self.ligand_sp_dir, self.ligand_opt_dir)
            g1_fe = _get_free_energy('1', mol_name, self.ligand_sp_dir, self.ligand_opt_dir)
            if not self.use_ref or mol_name in LIGAND_NOT_REF_SET:
                d_G = g1_fe + G_PROTON_GAS - g0_fe + DG_PROTON_SOLV_WATER + desc[2] - desc[1] + 1.89
                desc[7] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10))

                d_G = g1_fe + G_PROTON_GAS - g0_fe + DG_PROTON_SOLV_DMF + desc[5] - desc[4] + 1.89
                desc[9] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10))
            elif mol_name in LIGAND_SPEC_REF_SET:
                d_G_water, d_G_dmf = _get_ref_dG(self.ligand_sp_dir, self.ligand_opt_dir, LIGAND_SPEC_REF_SET[mol_name])

                d_G = g1_fe - g0_fe + d_G_water + desc[2] - desc[1]
                desc[7] = d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_LIGAND_pKa[LIGAND_SPEC_REF_SET[mol_name]][0]

                d_G = g1_fe - g0_fe + d_G_dmf + desc[5] - desc[4]
                desc[9] = d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_LIGAND_pKa[LIGAND_SPEC_REF_SET[mol_name]][1]
            else:
                d_G = g1_fe - g0_fe + self.ref_d_G_water + desc[2] - desc[1]
                desc[7] = d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_LIGAND_pKa[REF_LIGAND][0]
                
                d_G = g1_fe - g0_fe + self.ref_d_G_dmf + desc[5] - desc[4]
                desc[9] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_LIGAND_pKa[REF_LIGAND][1]

            if self.use_annotate and mol_name in self.annotate_info.index:
                if 'pKa(Water)-1' in self.annotate_info.columns and "pKa(Water)-2" in self.annotate_info.columns and not np.isnan(self.annotate_info.loc[mol_name, 'pKa(Water)-1']) and not np.isnan(self.annotate_info.loc[mol_name, 'pKa(Water)-2']):
                    desc[7] = min(self.annotate_info.loc[mol_name, 'pKa(Water)-1'], self.annotate_info.loc[mol_name, 'pKa(Water)-2'])
                if 'pKa(DMF)-1' in self.annotate_info.columns and "pKa(DMF)-2" in self.annotate_info.columns and not np.isnan(self.annotate_info.loc[mol_name, 'pKa(DMF)-1']) and not np.isnan(self.annotate_info.loc[mol_name, 'pKa(DMF)-2']):
                    desc[9] = min(self.annotate_info.loc[mol_name, 'pKa(DMF)-1'], self.annotate_info.loc[mol_name, 'pKa(DMF)-2'])
            
            desc[11], desc[12], desc[13], desc[14], desc[15] = read_entropy_from_gs_output(os.path.join(self.ligand_opt_dir, mol_name, 'output', '{}_dmf_0.out'.format(mol_name)))

            desc[16] = _get_enthalpy('0', mol_name, self.ligand_sp_dir, self.ligand_opt_dir) + desc[4] - 1.89
            desc[17] = _get_enthalpy('1', mol_name, self.ligand_sp_dir, self.ligand_opt_dir) + desc[5] - 1.89

            desc[19] = _get_dipole('0', mol_name, self.ligand_sp_dir)
        else:
            mol_name = mol.GetProp('Name')
            matches_COOH = _find_COOH(mol)
            C_indices = [a_idx for match in matches_COOH for a_idx in match if mol.GetAtomWithIdx(a_idx).GetSymbol() == 'C']
            O_indices = [a_idx for match in matches_COOH for a_idx in match if mol.GetAtomWithIdx(a_idx).GetSymbol() == 'O']
            assert len(C_indices) == 2, "Required 2 C atom in COOH group, but found {}".format(len(C_indices))
            assert len(O_indices) == 4, "Required 4 O atoms in COOH group, but found {}".format(len(O_indices))
            ligand_join_path = os.path.join(self.ligand_sp_dir, mol_name, 'output')
            desc = [0] * self.total_length
            # Gas output
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_gas_0.out'.format(mol_name)), index=-1)
            e_gas_0 = atoms.get_potential_energy() * eV2kcal_mol
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_gas_1.out'.format(mol_name)), index=-1)
            e_gas_1 = atoms.get_potential_energy() * eV2kcal_mol
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_gas_2.out'.format(mol_name)), index=-1)
            e_gas_2 = atoms.get_potential_energy() * eV2kcal_mol
            desc[0] = np.mean(atoms.get_charges()[O_indices])
            # Solvation output
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_water_0.out'.format(mol_name)), index=-1)
            desc[1] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_0
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_water_1.out'.format(mol_name)), index=-1)
            desc[2] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_1
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_water_2.out'.format(mol_name)), index=-1)
            desc[3] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_2
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_dmf_0.out'.format(mol_name)), index=-1)
            desc[4] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_0
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_dmf_1.out'.format(mol_name)), index=-1)
            desc[5] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_1
            atoms = ase.io.read(os.path.join(ligand_join_path, '{}_dmf_2.out'.format(mol_name)), index=-1)
            desc[6] = atoms.get_potential_energy() * eV2kcal_mol - e_gas_2
            # pKa
            g0_fe = _get_free_energy('0', mol_name, self.ligand_sp_dir, self.ligand_opt_dir)
            g1_fe = _get_free_energy('1', mol_name, self.ligand_sp_dir, self.ligand_opt_dir)
            g2_fe = _get_free_energy('2', mol_name, self.ligand_sp_dir, self.ligand_opt_dir)
            if not self.use_ref or mol_name in LIGAND_NOT_REF_SET:
                d_G = g1_fe + G_PROTON_GAS - g0_fe + DG_PROTON_SOLV_WATER + desc[2] - desc[1] + 1.89
                desc[7] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10))
                d_G = g2_fe + G_PROTON_GAS - g1_fe + DG_PROTON_SOLV_WATER + desc[3] - desc[2] + 1.89
                desc[8] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10))

                d_G = g1_fe + G_PROTON_GAS - g0_fe + DG_PROTON_SOLV_DMF + desc[5] - desc[4] + 1.89
                desc[9] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10))
                d_G = g2_fe + G_PROTON_GAS - g1_fe + DG_PROTON_SOLV_DMF + desc[6] - desc[5] + 1.89
                desc[10] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10))

            elif mol_name in LIGAND_SPEC_REF_SET:
                d_G_water, d_G_dmf = _get_ref_dG(self.ligand_sp_dir, self.ligand_opt_dir, LIGAND_SPEC_REF_SET[mol_name])

                d_G = g1_fe - g0_fe + d_G_water + desc[2] - desc[1]
                desc[7] = d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_LIGAND_pKa[LIGAND_SPEC_REF_SET[mol_name]][0]
                d_G = g2_fe - g1_fe + d_G_water + desc[3] - desc[2]
                desc[8] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_LIGAND_pKa[LIGAND_SPEC_REF_SET[mol_name]][1]

                d_G = g1_fe - g0_fe + d_G_dmf + desc[5] - desc[4]
                desc[9] = d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_LIGAND_pKa[LIGAND_SPEC_REF_SET[mol_name]][1]
                d_G = g2_fe - g1_fe + d_G_dmf + desc[6] - desc[5]
                desc[10] = d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_LIGAND_pKa[LIGAND_SPEC_REF_SET[mol_name]][1]

            else:
                d_G = g1_fe - g0_fe + self.ref_d_G_water + desc[2] - desc[1]
                desc[7] = d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_LIGAND_pKa[REF_LIGAND][0]
                d_G = g2_fe - g1_fe + self.ref_d_G_water + desc[3] - desc[2]
                desc[8] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_LIGAND_pKa[REF_LIGAND][1]
                
                d_G = g1_fe - g0_fe + self.ref_d_G_dmf + desc[5] - desc[4]
                desc[9] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_LIGAND_pKa[REF_LIGAND][1]
                d_G = g2_fe - g1_fe + self.ref_d_G_dmf + desc[6] - desc[5]
                desc[10] =  d_G * kcal_mol2J_mol / (IDEAL_GAS_CONSTANT * 298.15 * np.log(10)) + REF_LIGAND_pKa[REF_LIGAND][1]

            if self.use_annotate and mol_name in self.annotate_info.index:
                if 'pKa(Water)-1' in self.annotate_info.columns and "pKa(Water)-2" in self.annotate_info.columns and not np.isnan(self.annotate_info.loc[mol_name, 'pKa(Water)-1']) and not np.isnan(self.annotate_info.loc[mol_name, 'pKa(Water)-2']):
                    desc[7] = min(self.annotate_info.loc[mol_name, 'pKa(Water)-1'], self.annotate_info.loc[mol_name, 'pKa(Water)-2'])
                if 'pKa(DMF)-1' in self.annotate_info.columns and "pKa(DMF)-2" in self.annotate_info.columns and not np.isnan(self.annotate_info.loc[mol_name, 'pKa(DMF)-1']) and not np.isnan(self.annotate_info.loc[mol_name, 'pKa(DMF)-2']):
                    desc[9] = min(self.annotate_info.loc[mol_name, 'pKa(DMF)-1'], self.annotate_info.loc[mol_name, 'pKa(DMF)-2'])
            
            desc[11], desc[12], desc[13], desc[14], desc[15] = read_entropy_from_gs_output(os.path.join(self.ligand_opt_dir, mol_name, 'output', '{}_dmf_0.out'.format(mol_name)))

            desc[16] = _get_enthalpy('0', mol_name, self.ligand_sp_dir, self.ligand_opt_dir) + desc[4] - 1.89
            desc[17] = _get_enthalpy('1', mol_name, self.ligand_sp_dir, self.ligand_opt_dir) + desc[5] - 1.89
            desc[18] = _get_enthalpy('2', mol_name, self.ligand_sp_dir, self.ligand_opt_dir) + desc[6] - 1.89

            desc[19] = _get_dipole('0', mol_name, self.ligand_sp_dir)

        return [desc[i] for i in self.used_indices]

    @property
    def descriptor_summaries(self):
        summaries = [
            "COO- O gas Mulliken charge", # 0
            "SolFE water 0 eV", # 1
            "SolFE water -1 eV", # 2
            "SolFE water -2 eV", # 3
            "SolFE dmf 0 eV", # 4
            "SolFE dmf -1 eV", # 5
            "SolFE dmf -2 eV", # 6
            "pKa1(Water)", # 7
            "pKa2(Water)", # 8
            "pKa1(DMF)", # 9
            "pKa2(DMF)", # 10
            "S(Total)", # 11
            "S(Elec)", # 12
            "S(Trans)", # 13
            "S(Rot)", # 14
            "S(Vib)", # 15
            "H(0)", # 16
            "H(-1)", # 17
            "H(-2)", # 18
            "Dipole Moment", # 19
        ]
        return [summaries[i] for i in self.used_indices]
                
    @property
    def descriptor_names(self):
        return self.descriptor_summaries

    @property
    def descriptor_count(self):
        return len(self.descriptor_names)