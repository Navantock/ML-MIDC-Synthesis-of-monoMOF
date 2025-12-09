import torch
from torch.utils.data import Dataset, Subset, random_split

from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

import os
from os import PathLike
from abc import abstractmethod
from typing import Sequence, Dict, Optional, Union

from .descriptor import Desc_Calculators_Dict

#_ignore_acid_set = {'BA-4-I'}
#_ignore_ligand_set = {'82', '91', '92', '93', '94', '95'}


def _load_exp_data(exp_data_path: str):
    exp_data = None
    if exp_data_path.endswith('.csv'):
        exp_data = pd.read_csv(exp_data_path)
    else:
        exp_data = pd.read_excel(exp_data_path)
    assert exp_data is not None, "Fail: Experimental data could not be loaded"
    return exp_data

def _random_split_datase_from_frac_ls(dataset, frac_list):
    train_dataset, valid_dataset, test_dataset = None, None, None
    if len(frac_list) == 1:
        frac_list = [frac_list[0], 1 - frac_list[0]]
        train_dataset, test_dataset = random_split(dataset, frac_list)
    elif len(frac_list) == 2:
        train_dataset, test_dataset = random_split(dataset, frac_list)
    elif len(frac_list) == 3:
        train_dataset, valid_dataset, test_dataset = random_split(dataset, frac_list)
    else:
        raise ValueError('frac_list is used to split the dataset into train, validation, and test sets. It must have 1, 2, or 3 elements.')
    return train_dataset, valid_dataset, test_dataset

class SynMOF_Dataset:
    def __init__(self, 
                 frac_list: Sequence[float|int],
                 exp_data_path, 
                 acids_path, 
                 ligands_path, 
                 acid_column_name: str = "Acid Name", 
                 ligand_column_name: str = "Ligand Name", 
                 acid_q_column_name: str = "Acid Q (mmol)", 
                 y_column_name: str = "y"):
        
        self.frac_list = frac_list
        self.data_x: torch.Tensor
        self.data_y: torch.Tensor
        self.y_class_num: int
        
        if not exp_data_path.endswith('.csv') and not exp_data_path.endswith('.xlsx') and exp_data_path is not None:
            raise ValueError('Experimental data file must be in csv or xlsx format')
        self.exp_data_path = exp_data_path
        if not acids_path.endswith('.sdf'):
            raise ValueError('Acids file must be in sdf format')
        if not ligands_path.endswith('.sdf'):
            raise ValueError('Ligands file must be in sdf format')
        self.acids_path = acids_path
        self.ligands_path = ligands_path
        self.acid_column_name = acid_column_name
        self.ligand_column_name = ligand_column_name
        self.acid_q_column_name = acid_q_column_name
        self.y_column_name = y_column_name

    @property
    def raw_file_names(self):
        return [self.exp_data_path, self.acids_path, self.ligands_path]

    @property
    def processed_file_names(self):
        return ['SynMOF_data.pt', 'SynMOF_acid_descs.csv', 'SynMOF_ligand_descs.csv', 'SynMOF_al_names.csv']
    
    @abstractmethod
    def get_predict_data(self, predict_mol_filepath: str, **kwargs):
        pass


class SynMOF_DescriptorDataset(SynMOF_Dataset, Dataset):
    def __init__(self,
                 acid_descriptor_method: str, 
                 ligand_descriptor_method: str,
                 frac_list: Sequence[float|int],
                 exp_data_path, 
                 acids_path, 
                 ligands_path, 
                 root: str,
                 acid_descriptor_kwargs: dict = None,
                 ligand_descriptor_kwargs: dict = None,
                 select_k: int = None,
                 normalize: bool = True,
                 acid_column_name: str = "Acid Name", 
                 ligand_column_name: str = "Ligand Name", 
                 acid_q_column_name: str = "Acid Q (mmol)", 
                 tag_column_name: str = None,
                 tag_frac_ls: Sequence = [],
                 tag_train: Sequence = [0],
                 tag_valid: Sequence = [],
                 tag_test: Sequence = [1],
                 y_column_name: str = "y"):
        
        super(SynMOF_DescriptorDataset, self).__init__(frac_list, 
                                                       exp_data_path, 
                                                       acids_path, 
                                                       ligands_path, 
                                                       acid_column_name, 
                                                       ligand_column_name, 
                                                       acid_q_column_name, 
                                                       y_column_name)
        
        if acid_descriptor_method not in Desc_Calculators_Dict:
            raise ValueError('Acid Descriptor method not recognized or implemented: {}.'.format(acid_descriptor_method))
        if acid_descriptor_kwargs is None:
            acid_descriptor_kwargs = {}
        self.acid_desc_calculator = Desc_Calculators_Dict[acid_descriptor_method](**acid_descriptor_kwargs)
        if ligand_descriptor_method not in Desc_Calculators_Dict:
            raise ValueError('Ligand Descriptor method not recognized or implemented: {}.'.format(ligand_descriptor_method))
        if ligand_descriptor_kwargs is None:
            ligand_descriptor_kwargs = {}
        self.ligand_desc_calculator = Desc_Calculators_Dict[ligand_descriptor_method](**ligand_descriptor_kwargs)
        
        if self.exp_data_path is not None:
            self.root = root
            self.process()
            self.y_class_num = len(np.unique(self.data_y))
        
            self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None
            if tag_column_name is not None:
                tag_data = _load_exp_data(self.exp_data_path)[tag_column_name]
                if tag_frac_ls:
                    all_indices = [idx for idx, tag in enumerate(tag_data) if tag in tag_frac_ls]
                    self.train_dataset, self.valid_dataset, self.test_dataset = _random_split_datase_from_frac_ls(Subset(self, all_indices), frac_list)
                else:
                    train_indices = [idx for idx, tag in enumerate(tag_data) if tag in tag_train]
                    valid_indices = [idx for idx, tag in enumerate(tag_data) if tag in tag_valid]
                    test_indices = [idx for idx, tag in enumerate(tag_data) if tag in tag_test]
                    self.train_dataset, self.valid_dataset, self.test_dataset = Subset(self, train_indices), Subset(self, valid_indices), Subset(self, test_indices)
            else:
                self.train_dataset, self.valid_dataset, self.test_dataset = _random_split_datase_from_frac_ls(self, frac_list)
        
            self.select_k = select_k

            # Drop features with NaN
            self.acid_desc_nan_mask = ~torch.any(torch.isnan(self.data_acid_desc), dim=0)
            self.ligand_desc_nan_mask = ~torch.any(torch.isnan(self.data_ligand_desc), dim=0)

            self.data_acid_desc = self.data_acid_desc[:, self.acid_desc_nan_mask]
            self.data_ligand_desc = self.data_ligand_desc[:, self.ligand_desc_nan_mask]
            self.acid_desc_names = [desc for idx, desc in enumerate(self.acid_desc_names) if self.acid_desc_nan_mask[idx]]
            self.ligand_desc_names = [desc for idx, desc in enumerate(self.ligand_desc_names) if self.ligand_desc_nan_mask[idx]]
            self.other_desc_names = ["Acid Q (mmol)"] if acid_q_column_name is not None else []

            if self.select_k is not None:
                self.pre_select_features(select_k)
            
            if self.acid_q_column_name is not None:
                self.data_x = torch.cat((self.data_acid_desc, self.data_ligand_desc, self.data_acid_q), dim=1).float()
            else:
                self.data_x = torch.cat((self.data_acid_desc, self.data_ligand_desc), dim=1).float()

            self.normalize = normalize
            self.normalizer = None
            if self.normalize:
                self.normalizer = StandardScaler()
                self.normalizer.fit(self.data_x[self.train_dataset.indices])
                self.data_x = torch.tensor(self.normalizer.transform(self.data_x))
            # Check nan
            if torch.isnan(self.data_acid_desc).sum() > 0:
                raise ValueError('Acid descriptors contain NaN values')
            if torch.isnan(self.data_ligand_desc).sum() > 0:
                raise ValueError('Ligand descriptors contain NaN values')
            if torch.isnan(self.data_acid_q).sum() > 0:
                raise ValueError('Acid q values contain NaN values')

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.data_acid_desc[idx], self.data_ligand_desc[idx], self.data_acid_q[idx], self.data_y[idx]
    
    def pre_select_features(self, select_k: int):
        acid_ligand_features = torch.cat((self.data_acid_desc, self.data_ligand_desc), dim=1)
        acid_feat_len = self.data_acid_desc.shape[1]
        ligand_feat_len = self.data_ligand_desc.shape[1]

        var_selector = VarianceThreshold(0.0)
        var_selector.fit(acid_ligand_features[self.train_dataset.indices])
        mask_varsele = var_selector.get_support()

        self.data_acid_desc = self.data_acid_desc[:, mask_varsele[:acid_feat_len]]
        self.data_ligand_desc = self.data_ligand_desc[:, mask_varsele[acid_feat_len:]]
        self.acid_desc_names = [desc for idx, desc in enumerate(self.acid_desc_names) if mask_varsele[idx]]
        self.ligand_desc_names = [desc for idx, desc in enumerate(self.ligand_desc_names) if mask_varsele[idx + acid_feat_len]]
        acid_ligand_features = acid_ligand_features[:, mask_varsele]
        acid_feat_len = self.data_acid_desc.shape[1]
        ligand_feat_len = self.data_ligand_desc.shape[1]

        if 0 < select_k < mask_varsele.sum():
            feature_selector = SelectKBest(f_classif, k=self.select_k)
            feature_selector.fit(acid_ligand_features[self.train_dataset.indices], self.data_y[self.train_dataset.indices].reshape(-1))
            mask_featsele = feature_selector.get_support()
            self.data_acid_desc = self.data_acid_desc[:, mask_featsele[:acid_feat_len]]
            self.data_ligand_desc = self.data_ligand_desc[:, mask_featsele[acid_feat_len:]]
            self.acid_desc_names = [desc for idx, desc in enumerate(self.acid_desc_names) if mask_featsele[idx]]
            self.ligand_desc_names = [desc for idx, desc in enumerate(self.ligand_desc_names) if mask_featsele[idx + acid_feat_len]]

        if self.root is not None:
            with open(os.path.join(self.root, 'selected_features.txt'), 'w') as f:
                f.write('Acid Count: {}\n'.format(len(self.acid_desc_names)))
                f.write('Ligand Count: {}\n'.format(len(self.ligand_desc_names)))
                f.write('Acid descriptors:\n')
                f.write(','.join(self.acid_desc_names) + '\n')
                f.write('Ligand descriptors:\n')
                f.write(','.join(self.ligand_desc_names) + '\n')
        else:
            print('Select {} features for acid descriptors'.format(self.select_k))

    @property
    def acid_desc_names_nosele(self):
        return pd.read_csv(os.path.join(self.root, self.processed_file_names[1]), index_col=0).index.to_list()
    
    @property
    def ligand_desc_names_nosele(self):
        return pd.read_csv(os.path.join(self.root, self.processed_file_names[2]), index_col=0).index.to_list()

    def process(self):
        if self.root is not None and all(os.path.exists(os.path.join(self.root, pf)) for pf in self.processed_file_names):
            print(f"Directory {self.root} already exists. Data will be loaded from the existing files.")
            self.data_acid_desc, self.data_ligand_desc, self.data_acid_q, self.data_y = torch.load(os.path.join(self.root, self.processed_file_names[0]), weights_only=False)
            self.acid_desc_names = self.acid_desc_names_nosele
            self.ligand_desc_names = self.ligand_desc_names_nosele
            self.acid_dict = pd.read_csv(os.path.join(self.root, self.processed_file_names[1]), index_col=0).to_dict(orient='list')
            self.acid_dict = {k: np.asarray(v) for k, v in self.acid_dict.items()}
            self.ligand_dict = pd.read_csv(os.path.join(self.root, self.processed_file_names[2]), index_col=0).to_dict(orient='list')
            self.ligand_dict = {k: np.asarray(v) for k, v in self.ligand_dict.items()}
            self.len = len(self.data_y)
            return
        
        acids = Chem.SDMolSupplier(self.acids_path)
        ligands = Chem.SDMolSupplier(self.ligands_path)
        print("Calculating descriptors for acids and ligands...")
        self.acid_dict = {acid.GetProp("Name"): np.asarray(self.acid_desc_calculator.calculate(acid)) for acid in tqdm(acids) if acid.HasProp("Name") and acid.GetProp("Name")}
        self.ligand_dict = {ligand.GetProp("Name"): np.asarray(self.ligand_desc_calculator.calculate(ligand)) for ligand in tqdm(ligands) if ligand.HasProp("Name") and ligand.GetProp("Name")}

        exp_data = _load_exp_data(self.exp_data_path)

        data_acid_desc = []
        data_ligand_desc = []
        data_acid_q = []
        data_y = []
        data_al = {"Acid": [], "Ligand": []}
        print("Processing experimental data...")
        for index, row in tqdm(exp_data.iterrows()):
            acid_name = str(row[self.acid_column_name])
            ligand_name = str(row[self.ligand_column_name])
            data_acid_desc.append(self.acid_dict[acid_name])
            data_ligand_desc.append(self.ligand_dict[ligand_name])
            data_acid_q.append([row[self.acid_q_column_name]]) if self.acid_q_column_name is not None else data_acid_q.append([0.])
            data_y.append([row[self.y_column_name]])
            data_al["Acid"].append(acid_name)
            data_al["Ligand"].append(ligand_name)

        self.data_acid_desc = torch.tensor(np.asarray(data_acid_desc), dtype=torch.float64)
        self.acid_desc_names = self.acid_desc_calculator.descriptor_names
        self.data_ligand_desc = torch.tensor(np.asarray(data_ligand_desc), dtype=torch.float64)
        self.ligand_desc_names = self.ligand_desc_calculator.descriptor_names
        self.data_acid_q = torch.tensor(data_acid_q, dtype=torch.float64)
        self.data_y = torch.tensor(data_y, dtype=torch.float64)
        self.len = len(data_y)

        if self.root is not None:
            os.makedirs(self.root, exist_ok=True)
            torch.save((self.data_acid_desc, self.data_ligand_desc, self.data_acid_q, self.data_y), os.path.join(self.root, self.processed_file_names[0]))
            df = pd.DataFrame(self.acid_dict, index=self.acid_desc_names)
            df.to_csv(os.path.join(self.root, self.processed_file_names[1]))
            df = pd.DataFrame(self.ligand_dict, index=self.ligand_desc_names)
            df.to_csv(os.path.join(self.root, self.processed_file_names[2]))
            df = pd.DataFrame(data_al)
            df.to_csv(os.path.join(self.root, self.processed_file_names[3]), index=False)
    
    @property
    def data_al(self) -> Dict:
        return pd.read_csv(os.path.join(self.root, self.processed_file_names[3])).to_dict(orient='list')

    @property
    def acid_desc_mask(self):
        acid_desc_mask = np.zeros(len(self.acid_desc_calculator.descriptor_names), dtype=bool)
        for desc_name in self.acid_desc_names:
            j = 0
            while desc_name != self.acid_desc_calculator.descriptor_names[j]:
                j += 1
            acid_desc_mask[j] = True
        return acid_desc_mask
    
    @property
    def ligand_desc_mask(self):
        ligand_desc_mask = np.zeros(len(self.ligand_desc_calculator.descriptor_names), dtype=bool)
        for desc_name in self.ligand_desc_names:
            j = 0
            while desc_name != self.ligand_desc_calculator.descriptor_names[j]:
                j += 1
            ligand_desc_mask[j] = True
        return ligand_desc_mask
    
    @property
    def data_x_mask(self):
        data_x_mask = np.concatenate((self.acid_desc_mask, self.ligand_desc_mask), axis=0)
        if len(self.other_desc_names) > 0:
            data_x_mask = np.concatenate((data_x_mask, np.array([True] * len(self.other_desc_names))), axis=0)
        return data_x_mask
        
    def get_predict_data(self, predict_mol_input: Union[PathLike, Dict], mol_type: Optional[str] = None, acid_q_interpolation_num: Optional[int] = 5, sort_mol_input: bool = False):
        acid_dict, ligand_dict = None, None
        if isinstance(predict_mol_input, Dict):
            if 'acid' not in predict_mol_input or predict_mol_input['acid'] is None or predict_mol_input['acid'] == []:
                print("No specified acid in the yaml file. Use all acids in the database.")
                acid_dict = {k: v[self.acid_desc_mask] for k, v in self.acid_dict.items()}
            else:
                acid_dict = {k: self.acid_dict[k][self.acid_desc_mask] for k in predict_mol_input['acid'] if k in self.acid_dict}
            if 'ligand' not in predict_mol_input or predict_mol_input['ligand'] is None or predict_mol_input['ligand'] == []:
                print("No specified ligand in the yaml file. Use all ligands in the database.")
                ligand_dict = {k: v[self.ligand_desc_mask] for k, v in self.ligand_dict.items()}
            else:
                ligand_dict = {k: self.ligand_dict[k][self.ligand_desc_mask] for k in predict_mol_input['ligand'] if k in self.ligand_dict}
        else:    
            if not os.path.exists(predict_mol_input):
                raise ValueError('Predict molecule file does not exist: {}'.format(predict_mol_input))
            if predict_mol_input.endswith('.sdf'):
                if mol_type == 'acid':
                    acids = Chem.SDMolSupplier(predict_mol_input)
                    acid_dict = {acid.GetProp("Name") if acid.HasProp("Name") else acid.GetProp("_Name"): self.acid_desc_calculator.calculate(acid)[self.acid_desc_mask] for acid in tqdm(acids)}
                    ligand_dict = {k: v[self.ligand_desc_mask] for k, v in self.ligand_dict.items()}
                elif mol_type == 'ligand':
                    ligands = Chem.SDMolSupplier(predict_mol_input)
                    ligand_dict = {ligand.GetProp("Name") if ligand.HasProp("Name") else ligand.GetProp("_Name"): self.ligand_desc_calculator.calculate(ligand)[self.ligand_desc_mask] for ligand in tqdm(ligands)}
                    acid_dict = {k: v[self.acid_desc_mask] for k, v in self.acid_dict.items()}
                else:
                    if mol_type is None:
                        print("Read molecule type from sdf file.")
                    mols = Chem.SDMolSupplier(predict_mol_input)
                    acid_dict = {mol.GetProp("Name") if mol.HasProp("Name") else mol.GetProp("_Name"): self.acid_desc_calculator.calculate(mol)[self.acid_desc_mask] for mol in tqdm(mols) if mol.HasProp("Type") and mol.GetProp("Type").lower() == "a"}
                    assert len(acid_dict) > 0, "No acid found in the sdf file. Please check the sdf file."
                    ligand_dict = {mol.GetProp("Name") if mol.HasProp("Name") else mol.GetProp("_Name"): self.ligand_desc_calculator.calculate(mol)[self.ligand_desc_mask] for mol in tqdm(mols) if mol.HasProp("Type") and mol.GetProp("Type").lower() == "l"}
                    assert len(ligand_dict) > 0, "No ligand found in the sdf file. Please check the sdf file."
            elif predict_mol_input.endswith('.yaml'):
                with open (predict_mol_input, 'r') as f:
                    mol_info = yaml.safe_load(f)
                if 'acid' not in mol_info['acid'] or mol_info['acid'] is None or mol_info['acid'] == []:
                    print("No specified acid in the yaml file. Use all acids in the database.")
                    acid_dict = {k: v[self.acid_desc_mask] for k, v in self.acid_dict.items()}
                else:
                    acid_dict = {k: self.acid_dict[k][self.acid_desc_mask] for k in mol_info['acid'] if k in self.acid_dict}
                if 'ligand' not in mol_info['ligand'] or mol_info['ligand'] is None or mol_info['ligand'] == []:
                    print("No specified ligand in the yaml file. Use all ligands in the database.")
                    ligand_dict = {k: v[self.ligand_desc_mask] for k, v in self.ligand_dict.items()}
                else:
                    ligand_dict = {k: self.ligand_dict[k][self.ligand_desc_mask] for k in mol_info['ligand'] if k in self.ligand_dict}

        pred_data_x, ligand_list, acid_list, acid_q_list = [], list(ligand_dict.keys()), list(acid_dict.keys()), None
        if sort_mol_input:
            ligand_list.sort()
            acid_list.sort()
        # Order: for ligand: for acid: for acid q (optional)
        if self.acid_q_column_name is not None:
            min_acid_q, max_acid_q = self.data_acid_q.min(), self.data_acid_q.max()
            acid_q_list = list(np.linspace(min_acid_q, max_acid_q, acid_q_interpolation_num))
            for ligand_name in ligand_list:
                ligand_desc = ligand_dict[ligand_name]
                for acid_name in acid_list:
                    acid_desc = acid_dict[acid_name]
                    for acid_q in acid_q_list:
                        pred_data_x.append(np.concatenate((acid_desc, ligand_desc, [acid_q]), axis=0))
        else:
            for ligand_name in ligand_list:
                ligand_desc = ligand_dict[ligand_name]
                for acid_name in acid_list:
                    acid_desc = acid_dict[acid_name]
                    pred_data_x.append(np.concatenate((acid_desc, ligand_desc), axis=0))

        pred_data_x = np.array(pred_data_x)
        pred_data_x = torch.tensor(pred_data_x, dtype=torch.float64)
        # normalize
        if self.normalize:
            pred_data_x = torch.tensor(self.normalizer.transform(pred_data_x), dtype=torch.float64)
        # Check nan
        if torch.isnan(pred_data_x).sum() > 0:
            raise ValueError('Predicted descriptors contain NaN values')

        return pred_data_x, ligand_list, acid_list, acid_q_list

            
class SynMOF_DescriptorDataset_Predict(SynMOF_Dataset, Dataset):
    def __init__(self,
                 acid_desc_csv: PathLike = None,
                 ligand_desc_csv: PathLike = None,
                 acid_descriptor_method: str = None,
                 ligand_descriptor_method: str = None,
                 acid_descriptor_kwargs: dict = None,
                 ligand_descriptor_kwargs: dict = None,
                 normalize: bool = True):
        
        self.normalize = normalize
        self.normalizer = None

        self.acid_dict = {}
        self.ligand_dict = {}

        if acid_desc_csv is not None and ligand_desc_csv is not None:
            self.acid_dict = pd.read_csv(acid_desc_csv, index_col=0).to_dict(orient='list')
            self.acid_dict = {k: np.asarray(v) for k, v in self.acid_dict.items()}
            self.ligand_dict = pd.read_csv(ligand_desc_csv, index_col=0).to_dict(orient='list')
            self.ligand_dict = {k: np.asarray(v) for k, v in self.ligand_dict.items()}
        
        if acid_descriptor_method not in Desc_Calculators_Dict and acid_desc_csv is None:
            raise ValueError('Acid Descriptor method not recognized or implemented: {}.'.format(acid_descriptor_method))
        if acid_descriptor_kwargs is None:
            acid_descriptor_kwargs = {}
        self.acid_desc_calculator = Desc_Calculators_Dict[acid_descriptor_method](**acid_descriptor_kwargs) if acid_descriptor_method is not None else None
        if ligand_descriptor_method not in Desc_Calculators_Dict and ligand_desc_csv is None:
            raise ValueError('Ligand Descriptor method not recognized or implemented: {}.'.format(ligand_descriptor_method))
        if ligand_descriptor_kwargs is None:
            ligand_descriptor_kwargs = {}
        self.ligand_desc_calculator = Desc_Calculators_Dict[ligand_descriptor_method](**ligand_descriptor_kwargs) if ligand_descriptor_method is not None else None

    def get_predict_data(self, predict_ligand_input: PathLike, predict_acid_input: PathLike, sort_mol_input: bool = False):
        acid_dict, ligand_dict = None, None
        assert os.path.exists(predict_ligand_input), 'Predict ligand file does not exist: {}'.format(predict_ligand_input)
        assert predict_ligand_input.endswith('.sdf'), 'Predict ligand file must be in sdf format: {}'.format(predict_ligand_input)
        assert os.path.exists(predict_acid_input), 'Predict acid file does not exist: {}'.format(predict_acid_input)
        assert predict_acid_input.endswith('.sdf'), 'Predict acid file must be in sdf format: {}'.format(predict_acid_input)
        ligands = Chem.SDMolSupplier(predict_ligand_input)
        acids = Chem.SDMolSupplier(predict_acid_input)
        ligand_dict = {ligand.GetProp("Name") if ligand.HasProp("Name") else ligand.GetProp("_Name"): ligand for ligand in tqdm(ligands)}
        for ligand_name in ligand_dict:
            ligand_dict[ligand_name] = self.ligand_dict[ligand_name] if ligand_name in self.ligand_dict else self.ligand_desc_calculator.calculate(ligand_dict[ligand_name])
        acid_dict = {acid.GetProp("Name") if acid.HasProp("Name") else acid.GetProp("_Name"): acid for acid in tqdm(acids)}
        for acid_name in acid_dict:
            acid_dict[acid_name] = self.acid_dict[acid_name] if acid_name in self.acid_dict else self.acid_desc_calculator.calculate(acid_dict[acid_name])
        pred_data_x, ligand_list, acid_list = [], list(ligand_dict.keys()), list(acid_dict.keys())
        if sort_mol_input:
            ligand_list.sort()
            acid_list.sort()
        # Order: for ligand: for acid: for acid q (optional)

        for ligand_name in ligand_list:
            ligand_desc = ligand_dict[ligand_name]
            for acid_name in acid_list:
                acid_desc = acid_dict[acid_name]
                pred_data_x.append(np.concatenate((acid_desc, ligand_desc), axis=0))

        pred_data_x = np.array(pred_data_x)
        pred_data_x = torch.tensor(pred_data_x, dtype=torch.float64)
        # normalize
        if self.normalize:
            pred_data_x = torch.tensor(self.normalizer.transform(pred_data_x), dtype=torch.float64)
        # Check nan
        if torch.isnan(pred_data_x).sum() > 0:
            raise ValueError('Predicted descriptors contain NaN values')

        return pred_data_x, ligand_list, acid_list, None