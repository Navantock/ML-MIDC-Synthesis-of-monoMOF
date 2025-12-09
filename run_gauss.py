from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem
import ase.io
import numpy as np

import os
import logging
import time
from tqdm import tqdm

SOL2SOL_NAME = {
    'water': 'water',
    'n,n-DiMethylFormamide': 'dmf',
}

logging.getLogger('ase').setLevel(logging.ERROR)

def is_number(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False


def write_gaussian_opt_input(save_dir: str, 
                             mol: Mol, 
                             mol_name: str, 
                             mute_indices: list = [],
                             charge: int = 0, 
                             multiplicity: int = 1, 
                             nproc: int = 16,
                             mem: str = '32GB',
                             functional: str = 'B3LYP',
                             main_basis_set: str = '6-31G*',
                             sub_basis_set: str = 'LANL2DZ',
                             iodine_basis_set: str = 'def2tzvp',
                             solvent: str = None,
                             imp_solvent_method: str = 'IEFPCM',
                             opt_cartesian: bool = False,
                             **kwargs):

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    main_elem, sub_elem = set(), set()
    iodine_flag = False
    for atom in mol.GetAtoms():
        if atom.GetIdx() in mute_indices:
            continue
        if atom.GetAtomicNum() < 20:
            main_elem.add(atom.GetSymbol())
        elif atom.GetAtomicNum() == 53:
            iodine_flag = True
        else:
            sub_elem.add(atom.GetSymbol())
    main_elem_str = ' '.join(main_elem)
    sub_elem_str = ' '.join(sub_elem)

    conformer = mol.GetConformer()

    if not os.path.exists(os.path.join(save_dir, 'input')):
        os.makedirs(os.path.join(save_dir, 'input'), exist_ok=True)

    with open(os.path.join(save_dir, f'input/{mol_name}.gjf'), 'w') as fd:
        fd.write(f'%chk={os.path.join(save_dir, f'chk/{mol_name}.chk')}\n')
        fd.write(f'%NoSave\n')
        fd.write(f'%nprocshared={nproc}\n')
        fd.write(f'%mem={mem}\n')
        opt_str = 'opt=cartesian' if opt_cartesian else 'opt'
        if solvent:
            fd.write(f'#p {opt_str} freq {functional}/gen scrf=({imp_solvent_method}, solvent={solvent}, read)\n\n')
        else:
            fd.write(f'#p {opt_str} freq {functional}/gen\n\n')
        fd.write(f'{mol_name} Optimization\n\n')
        fd.write(f'{charge} {multiplicity}\n')
        for atom in mol.GetAtoms():
            if atom.GetIdx() in mute_indices:
                continue
            pos = conformer.GetAtomPosition(atom.GetIdx())
            fd.write(' {}                  {:.8f}    {:.8f}   {:.8f}\n'.format(atom.GetSymbol(), pos.x, pos.y, pos.z))
        fd.write('\n')
        fd.write(f'{main_elem_str} 0\n')
        fd.write(main_basis_set + '\n')
        fd.write(f'****\n')
        if iodine_flag:
            fd.write(f'I 0\n')
            fd.write(iodine_basis_set + '\n')
            fd.write(f'****\n')
        if sub_elem:
            fd.write(f'{sub_elem_str} 0\n')
            fd.write(sub_basis_set + '\n')
            fd.write(f'****\n')
        fd.write('\n\n')

def write_sp_gjf_from_src_opt_gjf(src_gjf_file: str, 
                                  save_file_name: str, 
                                  save_dir: str, 
                                  use_GC: bool = False, 
                                  acc_basis: bool = False,
                                  **kwargs):
    lines = None
    with open(src_gjf_file, 'r') as fd:
        lines = fd.readlines()
    
    use_solvent = False if save_file_name.split('_')[1] == 'gas' else True
    
    with open(os.path.join(save_dir, f'input/{save_file_name}.gjf'), 'w') as fd:
        new_lines = []
        for i, line in enumerate(lines):
            if f"%chk" in line:
                new_lines.append(f"%chk={os.path.join(save_dir, f'chk/{save_file_name}.chk')}\n")
            elif f"#p opt" in line:
                opt_str = line.split()[1]
                functional = line.split()[3].split('/')[0]
                new_functional = 'M062X'
                if functional != 'B3LYP':
                    new_functional = 'PBE1PBE EmpiricalDispersion=GD3BJ'
                if use_solvent:
                    new_line = line.replace("{} freq {}/gen scrf=(IEFPCM".format(opt_str, functional), "{}/gen scrf=(SMD".format(new_functional))
                else:
                    if use_GC:
                        new_line = line.replace("{} freq {}/gen".format(opt_str, functional), "CBS-QB3")
                    else:
                        new_line = line.replace("{} freq {}/gen".format(opt_str, functional), "{}/gen".format(new_functional))
                new_lines.append(new_line)
            elif f"Optimization" in line:
                new_line = line.replace("Optimization", "Single Point")
                new_lines.append(new_line)
            elif f"****" in line and use_GC:
                new_lines = new_lines[:-2]
            elif f"6-31G*" in line and acc_basis:
                new_line = line.replace("6-31G*", "def2tzvpp")
                new_lines.append(new_line)
            elif f"LANL2DZ" in line and acc_basis:
                new_line = line.replace("LANL2DZ", "SDD")
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        fd.writelines(new_lines)
        
def replace_gjf_coordinates(gjf_file: str, new_coords: np.ndarray):
    with open(gjf_file, 'r') as fd:
        lines = fd.readlines()
    with open(gjf_file, 'w') as fd:
        write_coordinates_from_idx = -1
        for i, line in enumerate(lines):
            ls_line = list(line.strip().split())
            if write_coordinates_from_idx > 0 and i - write_coordinates_from_idx < new_coords.shape[0]:
                write_idx = i - write_coordinates_from_idx
                element_name = line.split()[0]
                fd.write(' {}                  {:.8f}    {:.8f}   {:.8f}\n'.format(element_name, new_coords[write_idx][0], new_coords[write_idx][1], new_coords[write_idx][2]))
                continue
            if len(ls_line) == 2 and is_number(ls_line[0]) and is_number(ls_line[1]):
                write_coordinates_from_idx = i + 1
            fd.write(line)
                

def generate_opt_gjf_from_rdkit_mol(gjf_dir: str, 
                                    mol: Mol, 
                                    is_acid: bool, 
                                    addHs: bool = True, 
                                    multiplicity: int = 1, 
                                    solvent_list: list = ['water', 'n,n-DiMethylFormamide'],
                                    pre_build: bool = True,
                                    pre_opt: bool = True, 
                                    opt_cartesian: bool = False,
                                    has_metal: bool = False,
                                    seed: int = 42):
    """Convert RDKit molecule to Gaussian input file.

    Args:
        gjf_path (str): Path to save Gaussian input file
        mol (Mol): RDKit molecule.
        is_acid (bool, optional): Show the molecule is acid or ligand (acid has only 1 COOH group).
        addHs (bool, optional): Add explicit hydrogens to the molecule. Defaults to True.
        multiplicity (int, optional): The multiplicity of the molecule. Defaults to 1.

    """
    if addHs:
        mol = Chem.AddHs(mol, addCoords=True)

    charge = Chem.GetFormalCharge(mol)

    # Find COOH
    matches_COOH = mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)O'))
    if is_acid:
        assert len(matches_COOH) == 1, 'COOH not found or more than 1 COOH group found.'
    else:
        assert len(matches_COOH) == 2, 'COOH not found or more than 2 COOH groups found.'
    
    C_indices = [a_idx for match in matches_COOH for a_idx in match if mol.GetAtomWithIdx(a_idx).GetSymbol() == 'C']
    O_indices = [a_idx for match in matches_COOH for a_idx in match if mol.GetAtomWithIdx(a_idx).GetSymbol() == 'O']
    H_indices = [a.GetIdx() for o_idx in O_indices for a in mol.GetAtomWithIdx(o_idx).GetNeighbors() if a.GetSymbol() == 'H']

    # Generate a initial 3D structure
    if pre_build:
        AllChem.EmbedMolecule(mol, randomSeed=seed)
    if pre_opt:
        AllChem.MMFFOptimizeMolecule(mol)

    # Write Gaussian input file
    functional = 'TPSSh' if has_metal else 'B3LYP'
    if not os.path.exists(os.path.dirname(gjf_dir)):
        os.makedirs(os.path.dirname(gjf_dir), exist_ok=True)

    write_gaussian_opt_input(save_dir=gjf_dir,
                             mol=mol,
                             mol_name=mol.GetProp('Name') + '_gas_0',
                             charge=charge,
                             multiplicity=multiplicity,
                             opt_cartesian=opt_cartesian,
                             functional=functional)
    #add solvent
    for solvent in solvent_list:
        write_gaussian_opt_input(save_dir=gjf_dir,
                                 mol=mol,
                                 mol_name=mol.GetProp('Name') + '_{}_0'.format(SOL2SOL_NAME[solvent]),
                                 charge=charge,
                                 multiplicity=multiplicity,
                                 solvent=solvent,
                                 opt_cartesian=opt_cartesian,
                                 functional=functional)
    
    mute_indices = []
    
    for i, h_idx in enumerate(H_indices):
        mute_indices.append(h_idx)
        write_gaussian_opt_input(save_dir=gjf_dir,
                                 mol=mol,
                                 mol_name=mol.GetProp('Name') + '_gas_{}'.format(i+1),
                                 mute_indices=mute_indices,
                                 charge=charge - i - 1,
                                 multiplicity=multiplicity,
                                 opt_cartesian=opt_cartesian,
                                 functional=functional)
        # add solvent
        for solvent in solvent_list:
            write_gaussian_opt_input(save_dir=gjf_dir,
                                     mol=mol,
                                     mol_name=mol.GetProp('Name') + '_{}_{}'.format(SOL2SOL_NAME[solvent], i+1),
                                     mute_indices=mute_indices,
                                     charge=charge - i - 1,
                                     multiplicity=multiplicity,
                                     solvent=solvent,
                                     opt_cartesian=opt_cartesian,
                                     functional=functional)                                 


def check_gs_output(output_file: str):
    with open(output_file, 'r') as fd:
        lines = fd.readlines()
        if len(lines) > 2 and 'Normal termination of Gaussian' in lines[-1]:
            return True
    return False

def check_err_gs_output(output_file: str):
    with open(output_file, 'r') as fd:
        lines = fd.readlines()
        if len(lines) > 4 and 'Error termination' in lines[-4]:
            return True
    return False

def get_coordinates_from_gsout(output_file: str) -> np.ndarray:
    return ase.io.read(output_file).get_positions()

def run_opt_task(parent_task_dir: str, mol_dict: dict = None):
    print('***  Run Gaussian calculation ***')
    #for task_dir in tqdm(sorted(os.listdir(parent_task_dir))):
    for task_dir in tqdm(['103']):
        metal_flag = False
        if mol_dict is not None and mol_dict[task_dir].HasProp('Mult'):
            metal_flag = True
        if not os.path.exists(os.path.join(parent_task_dir, task_dir, 'output')):
            os.makedirs(os.path.join(parent_task_dir, task_dir, 'output'), exist_ok=True)
        if not os.path.exists(os.path.join(parent_task_dir, task_dir, 'chk')):
            os.makedirs(os.path.join(parent_task_dir, task_dir, 'chk'), exist_ok=True)
        print('---  Calculating molecule systems in {} ---'.format(task_dir))
        for task_file in os.listdir(os.path.join(parent_task_dir, task_dir, 'input')):
            if IGNORE_METAL_SOLVENT and metal_flag and "_gas_" not in task_file:
                print(f'Skip the solvent calculation for {task_file} in {task_dir}.')
                continue
            out_filename = task_file.split('.')[0] + '.out'
            out_filepath = os.path.join(parent_task_dir, task_dir, "output", out_filename)
            input_filepath = os.path.join(parent_task_dir, task_dir, "input", task_file)
            if os.path.exists(out_filepath):
                if check_gs_output(out_filepath):
                    print(f'{out_filepath} already exists and calculated successfully.')
                elif RERUN_MODE and check_err_gs_output(out_filepath):
                    print(f'{out_filepath} already exists but failed to calculate, rerun the calculation.')
                    # Try to find a successful calculation as initial coordinates
                    str_charge = task_file.split('.')[0].split('_')[-1]
                    for ref_file in os.listdir(os.path.join(parent_task_dir, task_dir, 'output')):
                        if ref_file.split('.')[0].split('_')[-1] == str_charge and check_gs_output(os.path.join(parent_task_dir, task_dir, 'output', ref_file)):
                            ref_coords = get_coordinates_from_gsout(os.path.join(parent_task_dir, task_dir, 'output', ref_file))
                            replace_gjf_coordinates(input_filepath, ref_coords)
                            print(f'- Rewrite the initial coordinates from {ref_file}.')
                            break
                    else:
                        print(f'- Failed to find a successful calculation as initial coordinates. Use the original coordinates.')
                    os.system(f'g16 < {input_filepath} > {out_filepath}')
                    time.sleep(0.1)
                    if not check_gs_output(out_filepath):
                        print(f'- Failed to calculate, please check the output file.', flush=True)
                    else:
                        print(f'- Calculated successfully.', flush=True)
                elif RERUN_MODE == 0:
                    print(f'Not RERUN_MODE, skip {out_filepath}.')
                else:
                    print(f'{out_filepath} may be calculating, skip the calculation.', flush=True)
                continue
            elif RERUN_MODE:
                print(f'{out_filepath} not exists, skip the calculation.', flush=True)
                continue

            print('Calculating: ', out_filepath)
            os.system(f'g16 < {input_filepath} > {out_filepath}')
            time.sleep(0.1)
            if not check_gs_output(out_filepath):
                print(f'Failed to calculate, please check the output file.', flush=True)
            else:
                print(f'Calculated successfully.', flush=True)


def run_sp_task(parent_sp_task_dir: str, parent_opt_task_dir: str, use_GC: bool = False, loose_gas_from_sol: str = None):
    print('***  Run SP Gaussian calculation  ***')
    #for task_dir in tqdm(sorted(os.listdir(parent_opt_task_dir))):
    for task_dir in tqdm(['103']):
        print('---  Calculating molecule systems in {} ---'.format(task_dir), flush=True)
        # Generate Directory
        if not os.path.exists(os.path.join(parent_sp_task_dir, task_dir, 'input')):
            os.makedirs(os.path.join(parent_sp_task_dir, task_dir, 'input'), exist_ok=True)
        if not os.path.exists(os.path.join(parent_sp_task_dir, task_dir, 'output')):
            os.makedirs(os.path.join(parent_sp_task_dir, task_dir, 'output'), exist_ok=True)
        if not os.path.exists(os.path.join(parent_sp_task_dir, task_dir, 'chk')):
            os.makedirs(os.path.join(parent_sp_task_dir, task_dir, 'chk'), exist_ok=True)
        # Write and Run SP input
        for input_file in os.listdir(os.path.join(parent_opt_task_dir, task_dir, 'input')):
            src_file_name = input_file.split('.')[0]
            out_coords_src_path = os.path.join(parent_opt_task_dir, task_dir, 'output', f'{src_file_name}.out')
            if not os.path.exists(out_coords_src_path) or not check_gs_output(out_coords_src_path):
                if loose_gas_from_sol is not None and src_file_name.split('_')[1] == 'gas':
                    out_coords_src_path = os.path.join(parent_opt_task_dir, task_dir, 'output', f'{src_file_name.split("_")[0]}_{loose_gas_from_sol}_{src_file_name.split("_")[-1]}.out')
                print(f'{src_file_name}.out not exists or failed, use coordinates from the gas phase calculation.', flush=True)
                out_coords_src_path = os.path.join(parent_opt_task_dir, task_dir, 'output', f'{src_file_name.split("_")[0]}_gas_{src_file_name.split("_")[-1]}.out')
                if not os.path.exists(out_coords_src_path) or not check_gs_output(out_coords_src_path):
                    print(f'Gas calculation for {src_file_name.split("_")[0]}_gas_{src_file_name.split("_")[-1]}.out not exists or failed, skip the calculation.', flush=True)
                    continue
            
            write_sp_gjf_from_src_opt_gjf(src_gjf_file=os.path.join(parent_opt_task_dir, task_dir, 'input', input_file),
                                          save_file_name=src_file_name,
                                          save_dir=os.path.join(parent_sp_task_dir, task_dir))
            replace_gjf_coordinates(os.path.join(parent_sp_task_dir, task_dir, 'input', f'{src_file_name}.gjf'),
                                    get_coordinates_from_gsout(out_coords_src_path))
            
            if not os.path.exists(os.path.join(parent_sp_task_dir, task_dir, 'output', f'{src_file_name}.out')) or not check_gs_output(os.path.join(parent_sp_task_dir, task_dir, 'output', f'{src_file_name}.out')):
                print(f'Calculating: {src_file_name}.out', flush=True)
                os.system(f'g16 < {os.path.join(parent_sp_task_dir, task_dir, "input", f"{src_file_name}.gjf")} > {os.path.join(parent_sp_task_dir, task_dir, "output", f"{src_file_name}.out")}')
            else:
                print(f'{src_file_name}.out already exists and calculated successfully.', flush=True)
            
            if src_file_name.split('_')[1] == 'gas':
                gas_fe_prefix = "_gc" if use_GC else "_acc"
                write_sp_gjf_from_src_opt_gjf(src_gjf_file=os.path.join(parent_opt_task_dir, task_dir, 'input', input_file),
                                              save_file_name=src_file_name + gas_fe_prefix,
                                              save_dir=os.path.join(parent_sp_task_dir, task_dir),
                                              use_GC=use_GC,
                                              acc_basis= not use_GC)
                replace_gjf_coordinates(os.path.join(parent_sp_task_dir, task_dir, 'input', f'{src_file_name + gas_fe_prefix}.gjf'),
                                        get_coordinates_from_gsout(out_coords_src_path))
                
                if not os.path.exists(os.path.join(parent_sp_task_dir, task_dir, 'output', f'{src_file_name + gas_fe_prefix}.out')) or not check_gs_output(os.path.join(parent_sp_task_dir, task_dir, 'output', f'{src_file_name + gas_fe_prefix}.out')):
                    print(f'Calculating: {src_file_name + gas_fe_prefix}.out', flush=True)
                    os.system(f'g16 < {os.path.join(parent_sp_task_dir, task_dir, "input", f"{src_file_name + gas_fe_prefix}.gjf")} > {os.path.join(parent_sp_task_dir, task_dir, "output", f"{src_file_name + gas_fe_prefix}.out")}')
                else:
                    print(f'{src_file_name + gas_fe_prefix}.out already exists and calculated successfully.', flush=True)

GEN_GJF = [0, 1]
# 0: modulator(acid) 1: ligand
RUN_OPT = [0, 1]
RUN_SP = [0, 1]
# rerun the failed calculation
# 0: Not Rerun 1: Rerun the failed calculation
RERUN_MODE = 0
# Use solvent conformation when gas optimization failed. None for strict gas calculation.
LOOSE_GAS_OPT_SOL = None
# Ignore metal solvent calculation
# 0: Not Ignore 1: Ignore
IGNORE_METAL_SOLVENT = 1

if __name__ == "__main__":
    qc_result_dir = "/mnt/data8T/wuhf/monoMOF-SynPred/qc_calc/"
    ligands_file = "./dataset/ligands/monoMOF-ligands.sdf"
    acids_file = "./dataset/acids/monoMOF-acids.sdf"
    acid_task_dir = os.path.join(qc_result_dir, 'acids_opt')
    ligand_task_dir = os.path.join(qc_result_dir, 'ligands_opt')
    acids_sp_task_dir = os.path.join(qc_result_dir, 'acids_sp')
    ligands_sp_task_dir = os.path.join(qc_result_dir, 'ligands_sp')

    ligands_sdf = Chem.SDMolSupplier(ligands_file)
    acids_sdf = Chem.SDMolSupplier(acids_file)

    ligands_dict = {lig.GetProp('Name'): lig for lig in ligands_sdf}
    acids_dict = {acid.GetProp('Name'): acid for acid in acids_sdf}

    if 0 in GEN_GJF:
        print('***  Generate Gaussian input files for acids  ***')
        for acid in tqdm(acids_sdf):
            save_gjf_dir = os.path.join(qc_result_dir, 'acids_opt/{}'.format(acid.GetProp('Name')))
            generate_opt_gjf_from_rdkit_mol(gjf_dir=save_gjf_dir,
                                            mol=acid,
                                            is_acid=True,
                                            addHs=True,
                                            multiplicity=1,
                                            pre_build=True,
                                            pre_opt=True)
    if 1 in GEN_GJF:
        print('***  Generate Gaussian input files for ligands  ***')
        for ligand in tqdm(ligands_sdf):
            save_gjf_dir = os.path.join(qc_result_dir, 'ligands_opt/{}'.format(ligand.GetProp('Name')))
            use_pre_build = False if ligand.HasProp('Mult') else True
            mult = ligand.GetProp('Mult') if ligand.HasProp('Mult') else 1
            generate_opt_gjf_from_rdkit_mol(gjf_dir=save_gjf_dir,
                                            mol=ligand,
                                            is_acid=False,
                                            addHs=True,
                                            multiplicity=mult,
                                            pre_build=use_pre_build,
                                            pre_opt=True,
                                            opt_cartesian=True if ligand.HasProp('Mult') else False,
                                            has_metal=True if ligand.HasProp('Mult') else False)

    # Run Gaussian Optimization calculation
    if 0 in RUN_OPT:
        run_opt_task(acid_task_dir)
                
    if 1 in RUN_OPT:
        run_opt_task(ligand_task_dir, ligands_dict)

    # Run Gaussian Single Point calculation
    if 0 in RUN_SP:
        run_sp_task(parent_sp_task_dir=acids_sp_task_dir,
                    parent_opt_task_dir=acid_task_dir,
                    use_GC=True,
                    loose_gas_from_sol=LOOSE_GAS_OPT_SOL)
    
    if 1 in RUN_SP:
        run_sp_task(parent_sp_task_dir=ligands_sp_task_dir,
                    parent_opt_task_dir=ligand_task_dir,
                    use_GC=False,
                    loose_gas_from_sol=LOOSE_GAS_OPT_SOL)
    