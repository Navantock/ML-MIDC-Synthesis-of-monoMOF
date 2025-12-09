# **ML-MIDC Synthesis of monoMOF**
##  🐤 1 Introduction

This project contains serveral methods to predict the synthesis result of $\text{Hf}_{12}$-monoMOFs with different modulator, linkers and conditions. MIDC(Modulator-Induced Dimensionality Control) is the core strategy of the synthesis. This repository primarily provides predictions of the success rate for Linker-Modulator combinations and the recommended addition amount of Modulators, with the goal of obtaining monoMOF.

##  📦 2 Dependencies

Install dependencies by conda and pip.

```bash
# [OPTIONAL][RECOMMEND] create conda environment
[Optional] conda create -n monomof python=3.12
[Optional] conda activate monomof

# [MANDATORY] Install dependencies
# [RECOMMEND] Install from a new environment

conda install numpy
conda install pandas
conda install rdkit -c conda-forge
pip install PyYAML
# In our project, scikit-learn=1.5.2
conda install scikit-learn
pip install xgboost
# Require torch functions. Torch installation refers to https://pytorch.org/get-started/locally/. In our project, torch=2.5.1 CUDA=12.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# Importance analysis provided by SHAP
conda install -c conda-forge shap
# Use joblib to save or load model
pip install joblib

# [OPTIONAL][RECOMMEND] install part of dependencies for data preprocessing or analysing
pip install openpyxl
conda install matplotlib
```

## 🚀 3 Run Prediction

By utilizing our pre-trained models, you can replicate our prediction results or conduct synthetic design based on modulators/linkers with pre-computed descriptors.

### 3.1 Prepare Modulators & Linkers SDF File

The synthesis prediction requires at least one potential substrate for both linker and modulator. We recommend using ChemDraw software within ChemOffice to draw your Linker and Modulator (referred to as Ligand and Acid in filenames or codes), and save each as a separate sdf file.  Then name each molecule by add a attribution **Name** so that the code can find it correctly.

Here, we provide the modulator (`./dataset/acids/acids.sdf`) and linker (`./dataset/ligands/ligands.sdf`) files involved in our experiments for reference or to reproduce our predictions.

### 3.2 Prepare descriptors

Using our trained model requires specific descriptors. Specifically, you need to prepare two CSV files corresponding to the calculated linker/modulator descriptors. The first row of each CSV file should list the linker/modulator names from the SDF file, and the first column should list the names of all descriptors.

To use our model, the specific descriptors must be included. For the descriptors of our linkers or modulators as an example, please refer to `./dataset/acids/acids_descs.csv` or `./dataset/ligands/ligands_descs.csv`.  The meanings of these descriptors can be found in our supplementary materials.

### 3.3 Predict L-M Combination / Modulator Concentration

Use a dataset configuration file to link the molecular files and descriptor files you prepared earlier. The file is located at `./configs/dataset_configs/predict_only.yaml` and has the following form

```
dataset_class: descriptors_predict

dataset_kwargs:
  acid_desc_csv: ./dataset/acids/acid_descs.csv
  ligand_desc_csv: ./dataset/ligands/ligand_descs.csv
  acid_descriptor_method: acid_hybrid
  ligand_descriptor_method: ligand_hybrid
```

Please modify the **acid_desc_csv** and **ligand_desc_csv** as your own file path in Sec 3.2 if you have independently prepared the descriptors for your molecules. 

For a given linker-modulator combination, the classification model determines whether the resulting product is a monoMOF/3D-MOF, or a non-crystalline product. These classifiers are located at `./models/classifier` corresponding to the three stages in our experiment.

```
|- models
	|- classfier
		|- s1_svm
		|- s2_svm
		|- all_random_svm
```

The prefix 'all_random' refers to the model trained on all data across 3 stages with a train/test split ratio 4:1.

Similarly, if you want to predict the modulator concentration (actually the molar quantity in 1 ml of DMF solvent), please refer to the `./models/regressor`.

```
|- models
	|- regressor
		|- s1_svm
		|- s2_svm
		|- all_random_svm
```

Now you need to specify the path to use the model in `./configs/train_configs/predict_only.yaml`, where the variable **model_load_path** is the corresponding parameters, like

```yaml
model_load_path: ./models/classifier/all_random_svm/model.pkl
predict_kwargs:
  predict_ligand_input: ./dataset/ligands/ligands.sdf
  predict_acid_input: ./dataset/acids/acids.sdf
  sort_mol_input: False
```

If you have prepared your own sdf file in Sec 3.1, please modify the variables **predict_ligand_input** and  **predict_acid_input** to your own file path.

The running config file refers to `./configs/predict_only.yaml`, where the variable **result_dir** is the directory you want to save the prediction results. Please modify it.

After this, all you need to do is to run

```bash
python main.py -c config predict_only.yaml
```

The prediction results will output to CSV files in matrix format, with row labels representing linker names and column labels representing modulator names.

## 🔍 4 Detailed Usage

If you need to use the code in this repository for descriptor computation or to train your own models, please refer to the detailed workflow descriptions in the respective subsections of this chapter.
### 4.1 Run Gaussian Calculation (QC Descriptors Required)

If you are using our model or attempting to train a model incorporating quantum chemistry descriptors, you must first perform quantum chemistry calculations. For your convenience, we provide a calculation script `run_gauss.py`, based on the Gaussian16 quantum chemistry software. Once you have confirmed that Gaussian16 is installed and the relevant environment variables are configured, please refer to the following steps

1. Set the variable **ligands_file**/**acids_file** in `run_gauss.py` as your own linker/modulator file path in section 3.1
2. Set the variable **qc_result_dir** in `run_gauss.py` as your quantum chemistry save directory.
3. Simply run `python run_gauss.py`

The file tree structure obtained after correct execution is as follows

```
<your_qc_results_dir>
|- acids_opt
  |- <acid_name_1>
  	|- chk
  	|- input
  	|- output
  	  |- <acid_name_1>_dmf_0.out
  	  |- <acid_name_1>_dmf_1.out
  	  |- <acid_name_1>_gas_0.out
  	  |- <acid_name_1>_gas_1.out
  	  |- <acid_name_1>_water_0.out
  	  |- <acid_name_1>_water_1.out
  |- ...
|- acids_sp
  |- <acid_name_1>
  	|- chk
  	|- input
  	|- output
  	  |- <acid_name_1>_dmf_0.out
  	  |- <acid_name_1>_dmf_1.out
  	  |- <acid_name_1>_gas_0_gc.out
  	  |- <acid_name_1>_gas_0.out
  	  |- <acid_name_1>_gas_0_gc.out
  	  |- <acid_name_1>_gas_1.out
  	  |- <acid_name_1>_water_0.out
  	  |- <acid_name_1>_water_1.out
  |- ...
|- ligands_opt
  |- <ligand_name_1>
  	|- chk
  	|- input
  	|- output
  	  |- <ligand_name_1>_dmf_0.out
  	  |- <ligand_name_1>_dmf_1.out
  	  |- <ligand_name_1>_dmf_2.out
  	  |- <ligand_name_1>_gas_0.out
  	  |- <ligand_name_1>_gas_1.out
  	  |- <ligand_name_1>_gas_2.out
  	  |- <ligand_name_1>_water_0.out
  	  |- <ligand_name_1>_water_1.out
  	  |- <ligand_name_1>_water_2.out
  |- ...
|- ligands_sp
  |- <ligand_name_1>
    |- chk
  	|- input
  	|- output
  	  |- <ligand_name_1>_dmf_0.out
  	  |- <ligand_name_1>_dmf_1.out
  	  |- <ligand_name_1>_dmf_2.out
  	  |- <ligand_name_1>_gas_0_acc.out
  	  |- <ligand_name_1>_gas_0.out
  	  |- <ligand_name_1>_gas_1_acc.out
  	  |- <ligand_name_1>_gas_1.out
  	  |- <ligand_name_1>_gas_2_acc.out
  	  |- <ligand_name_1>_gas_2.out
  	  |- <ligand_name_1>_water_0.out
  	  |- <ligand_name_1>_water_1.out
  	  |- <ligand_name_1>_water_2.out
  |- ...

```

This script provides the superset of calculations required to meet any demand. In practice, if the final selected descriptor isn't needed for all files, you can modify the script in our Supplementary Materials to reduce computational load.

If you decide not to use quantum chemistry descriptors, please set the variables **ACID_GS_SELECTED_INDICES** and **LIGAND_GS_SELECTED_INDICES** in `modules/descriptor/hybrid.py` as empty list.

### 4.2 CSV Dataset
Training a model requires not only the molecular structures and descriptors but also a dataset. For classification models, the dataset is a CSV file containing at least the following elements:

| Ligand Name | Acid Name | Tag  | y    |
| ----------- | --------- | ---- | ---- |
| L-XX        | M-XX      | 0    | 1    |
| ...         | ...       | ...  | ...  |

where $y$ denotes the experiment results (0 for non-crystalline product, 1 for monoMOF and 2 for 3D-MOF). Tag is deployed to specify the train/test split or for other usage if random split. 

For regressors, the $y$ denotes the molar quantity in 1 ml of DMF solvent.

| Ligand Name | Acid Name | Tag  | y    |
| ----------- | --------- | ---- | ---- |
| L-XX        | M-XX      | 0    | 0.40 |
| ...         | ...       | ...  | ...  |

These are corresponding to the dataset configuration for training as

```yaml
dataset_class: descriptors

dataset_kwargs:
  acid_descriptor_method: acid_hybrid
  ligand_descriptor_method: ligand_hybrid
  acid_descriptor_kwargs:
    acid_sp_dir: <your_qc_results_dir>/acids_sp
    acid_opt_dir: <your_qc_results_dir>/acids_opt
  ligand_descriptor_kwargs:
    ligand_sp_dir: <your_qc_results_dir>/ligands_sp
    ligand_opt_dir: <your_qc_results_dir>/ligands_opt
  frac_list:
    - 0.8
    - 0.2
  exp_data_path: <your_csv_experiment_data_path>
  acids_path: <your_acids_sdf>
  ligands_path: <your_ligands_sdf>
  root: <a_directory_to_save_dataset>
  select_k: 0 # default as 0 to use all descriptors
  normalize: True # normalize the features
  acid_column_name: Acid Name # column name in <your_csv_experiment_data_path>
  ligand_column_name: Ligand Name # column name in <your_csv_experiment_data_path>
  tag_column_name: Tag
  tag_train: 
  	- 0	# if specified, use the data with Tag in tag_train as training data
  tag_test:
    - 1 # if specified, use the data with Tag in tag_train as training data
  y_column_name: y
```

### 4.3 Model Configurations

We have provided all model configurations in our experiments at `./configs/model_configs`, containing the hyperparameters set to search.



