import torch
import numpy as np
import pandas as pd

from modules import *

import os
import time
import yaml
import copy
import joblib
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

from configs import get_args


def global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset(dataset_args):
    if dataset_args["dataset_class"] == "descriptors":
        dataset = SynMOF_DescriptorDataset(**dataset_args["dataset_kwargs"])
    elif dataset_args["dataset_class"] == "descriptors_predict":
        dataset = SynMOF_DescriptorDataset_Predict(**dataset_args["dataset_kwargs"])
    else:
        raise ValueError("Fail: Dataset class {} not recognized or implemented.".format(dataset_args["dataset_class"]))
    return dataset

def prepare_model(args, model_args, train_args, dataset: SynMOF_Dataset):
    if train_args.get("model_load_path", None) is not None:
        assert os.path.exists(train_args["model_load_path"]), "Fail: Model load path {} does not exist.".format(train_args["model_load_path"])
        # Load model from file
        model = joblib.load(train_args["model_load_path"])
        normalizer = joblib.load(os.path.join(os.path.dirname(train_args["model_load_path"]), "normalizer.pkl"))
        if hasattr(dataset, "data_x") and dataset.get("normalizer", None) is not None:
            dataset.data_x = normalizer.transform(dataset.normalizer.inverse_transform(dataset.data_x))
        dataset.normalizer = joblib.load(os.path.join(os.path.dirname(train_args["model_load_path"]), "normalizer.pkl"))
        return model

    model = None
    if model_args["model_class_type"] == "Classifier":
        model = Classifier_Dict[model_args["model_class"]](**model_args["model_kwargs"])
    elif model_args["model_class_type"] == "Regressor":
        model = Regressor_Dict[model_args["model_class"]](**model_args["model_kwargs"])
    else:
        raise ValueError("Fail: Model class type {} not recognized.".format(model_args["model_class_type"]))
    if args.get("train", False):
        assert isinstance(dataset, SynMOF_DescriptorDataset), "Fail: Dataset class {} not suitable for training model class {}.".format(dataset.__class__.__name__, model_args["model_class"])
        if train_args["kfold"]:
            train_x = dataset.normalizer.inverse_transform(dataset.data_x[dataset.train_dataset.indices])
            train_y = dataset.data_y[dataset.train_dataset.indices]
            kf_indices = []
            if train_args["kfold_by_ligand"]:
                # KFold by ligand type
                kf= KFold(n_splits=train_args["kfold"], shuffle=True, random_state=args["seed"])
                train_ligands = [dataset.data_al["Ligand"][idx] for idx in dataset.train_dataset.indices]
                train_ligands_set = list(set(train_ligands))
                for fold, (train_index, val_index) in enumerate(kf.split(train_ligands_set)):
                    curr_lg = [train_ligands_set[i] for i in train_index]
                    lg_isin = np.isin(train_ligands, curr_lg)
                    kf_train_indices = np.where(lg_isin)[0]
                    curr_lg = [train_ligands_set[i] for i in val_index]
                    lg_isin = np.isin(train_ligands, curr_lg)
                    kf_val_indices = np.where(lg_isin)[0]
                    kf_indices.append((kf_train_indices, kf_val_indices))
            else:
                kf = StratifiedKFold(n_splits=train_args["kfold"], shuffle=True, random_state=args["seed"])
                kf_indices = list(kf.split(train_x, train_y))

            if train_args["search_hyperparams"] and "model_search_params" in model_args:
                print("------ Searching hyperparameters ------")
                clf = Pipeline([("scaler", StandardScaler()), ("model", model.model)])
                if isinstance(model.model, TransformedTargetRegressor):
                    search_params = {f"model__regressor__{k}": v for k, v in model_args["model_search_params"].items()}
                else:
                    search_params = {f"model__{k}": v for k, v in model_args["model_search_params"].items()}
                scoring = None
                if model_args["model_class_type"] == "Classifier":
                    #scoring = get_class_i_scorer(1, "f1")
                    scoring = "f1_macro"
                else:
                    scoring = "neg_root_mean_squared_error"
                grid_search = GridSearchCV(clf,
                                           search_params,
                                           cv=kf_indices,
                                           scoring=scoring,
                                           n_jobs=1 if model_args["model_class"] == "XGBoost" else -1,
                                           verbose=1)
                grid_search.fit(train_x, train_y.reshape(-1))
                print("Best Score: {}".format(grid_search.best_score_))
                # Update model with best hyperparameters
                new_params = {k.replace("model__", ""): v for k, v in grid_search.best_params_.items()}
                model.model.set_params(**new_params)

                # Add other fixed hyperparams and save
                if isinstance(model.model, TransformedTargetRegressor):
                    new_params = {k.replace("regressor__", ""): v for k, v in new_params.items()}
                new_params.update({k: v for k, v in model_args["model_kwargs"].items() if k not in new_params})
                with open(os.path.join(args["result_dir"], "best_hyperparams.yaml"), "w") as f:
                    yaml.dump(new_params, f, allow_unicode=True)
                print("Best hyperparameters saved to best_hyperparams.yaml")
            
            best_fold = None
            best_fold_normalizer = None
            best_fold_val_results = None
            best_score = -np.inf if model_args["model_class_type"] == "Classifier" else np.inf
            sele_score = "f1_score" if model_args["model_class_type"] == "Classifier" else "rmse"
            for fold, (train_index, val_index) in enumerate(kf_indices):
                print(f"=-=-=-Running K-fold {fold + 1}/{train_args['kfold']} -=-=-=")
                os.makedirs(os.path.join(args["result_dir"], f"fold_{fold + 1}"), exist_ok=True)

                normalizer = StandardScaler().fit(train_x[train_index])
                # save normalizer
                joblib.dump(normalizer, os.path.join(args["result_dir"], f"fold_{fold + 1}", "normalizer.pkl"))

                fit_x = normalizer.transform(train_x[train_index])
                fit_y = train_y[train_index]
                model.train(fit_x, 
                            fit_y, 
                            y_class_num=dataset.y_class_num,
                            save_model=True,
                            save_roc_curve=True if model_args["model_class_type"] == "Classifier" else False,
                            result_save_dir=os.path.join(args["result_dir"], f"fold_{fold + 1}"),
                            acid_desc_names=dataset.acid_desc_names,
                            ligand_desc_names=dataset.ligand_desc_names,
                            other_desc_names=dataset.other_desc_names,
                            importance=train_args["importance"] if train_args is not None and "importance" in train_args else None)
                
                val_x = normalizer.transform(train_x[val_index])
                val_y = train_y[val_index]
                val_results = model.test(val_x, 
                                        val_y, 
                                        y_class_num=dataset.y_class_num,
                                        result_save_dir=os.path.join(args["result_dir"], f"fold_{fold + 1}"),
                                        save_roc_curve=True if model_args["model_class_type"] == "Classifier" else False,
                                        acid_desc_names=dataset.acid_desc_names, 
                                        other_desc_names=dataset.other_desc_names,
                                        ligand_desc_names=dataset.ligand_desc_names, 
                                        train_x=fit_x,
                                        train_y=fit_y,
                                        importance=train_args["importance"] if train_args is not None and "importance" in train_args else None)
                
                # save fold train indices and validation indices
                np.savez(os.path.join(args["result_dir"], f"fold_{fold + 1}", "train_val_indices.npz"),
                         train_indices=train_index,
                         val_indices=val_index)
                
                score = val_results[sele_score]
                # average if score is a list
                if isinstance(score, list):
                    score = np.mean(score)
                # record best fold
                if (model_args["model_class_type"] == "Classifier" and score > best_score) or (model_args["model_class_type"] == "Regressor" and score < best_score):
                    best_score = score
                    best_fold_val_results = val_results
                    best_fold = fold + 1
                    best_fold_normalizer = normalizer
                
            # train on the whole dataset
            print("=-=-=-Running training on the whole dataset -=-=-=")
            train_x = dataset.data_x[dataset.train_dataset.indices]
            model.train(train_x,
                        train_y, 
                        y_class_num=dataset.y_class_num, 
                        result_save_dir=args["result_dir"],
                        acid_desc_names=dataset.acid_desc_names,
                        ligand_desc_names=dataset.ligand_desc_names,
                        other_desc_names=dataset.other_desc_names,
                        importance=train_args["importance"] if train_args is not None and "importance" in train_args else None)
            joblib.dump(dataset.normalizer, os.path.join(args["result_dir"], "normalizer.pkl"))
            
            if train_args["use_kfold_model"]:
                model = joblib.load(os.path.join(args["result_dir"], f"fold_{best_fold}", "model.pkl"))
                model.val_results = best_fold_val_results
                dataset.data_x = best_fold_normalizer.transform(dataset.normalizer.inverse_transform(dataset.data_x))
                dataset.normalizer = best_fold_normalizer
                print(f"Using model from fold {best_fold} with score {best_score:.4f}")
            
        else:
            fit_x = dataset.data_x[dataset.train_dataset.indices]
            fit_y = dataset.data_y[dataset.train_dataset.indices]
            model.train(fit_x, 
                        fit_y, 
                        y_class_num=dataset.y_class_num, 
                        result_save_dir=args["result_dir"],
                        acid_desc_names=dataset.acid_desc_names,
                        ligand_desc_names=dataset.ligand_desc_names,
                        other_desc_names=dataset.other_desc_names,
                        importance=train_args["importance"] if train_args is not None and "importance" in train_args else None)
            if dataset.valid_dataset:
                val_x = dataset.data_x[dataset.valid_dataset.indices]
                val_y = dataset.data_y[dataset.valid_dataset.indices]
                model.test(val_x, 
                           val_y, 
                           y_class_num=dataset.y_class_num,
                           result_save_dir=os.path.join(args["result_dir"]),
                           acid_desc_names=dataset.acid_desc_names, 
                           other_desc_names=dataset.other_desc_names,
                           ligand_desc_names=dataset.ligand_desc_names, 
                           train_x=fit_x,
                           train_y=fit_y,
                           importance=None)
    
    return model

def evaluate_model(model, dataset: SynMOF_Dataset, result_save_dir: str = None, train_args: dict = None):
    if isinstance(dataset, SynMOF_DescriptorDataset):
        test_x = dataset.data_x[dataset.test_dataset.indices]
        test_y = dataset.data_y[dataset.test_dataset.indices]
        results = model.test(test_x, 
                             test_y, 
                             y_class_num=dataset.y_class_num,
                             result_save_dir=result_save_dir,
                             acid_desc_names=dataset.acid_desc_names, 
                             other_desc_names=dataset.other_desc_names,
                             ligand_desc_names=dataset.ligand_desc_names, 
                             train_x=dataset.data_x[dataset.train_dataset.indices],
                             train_y=dataset.data_y[dataset.train_dataset.indices],
                             importance=train_args["importance"] if train_args is not None and "importance" in train_args else None)
        return results
    else:
        raise ValueError("Fail: Dataset class {} not suitable for evaluating model.".format(dataset.__class__.__name__))

def predict_model(model, dataset: SynMOF_Dataset, train_args, result_save_dir: str = None, type_idx2name: dict = {0: "NA", 1: "monoMOF", 2: "3DMOF"}):
    results_dir = os.path.join(result_save_dir, "predictions")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    pred_data_x, ligands_list, acids_list, acid_q_list = dataset.get_predict_data(**train_args["predict_kwargs"])
    pred_data_y = model.predict(pred_data_x)
    pred_data_y_prob = model.predict_proba(pred_data_x) if hasattr(model, "predict_proba") else None

    y_mat = pred_data_y.reshape(len(ligands_list), len(acids_list))
    df_y = pd.DataFrame(y_mat, index=ligands_list, columns=acids_list)
    df_y.to_csv(os.path.join(results_dir, "predictions.csv"), index=True, header=True)
    if pred_data_y_prob is not None:
        if pred_data_y_prob.shape[-1] <= 2:
            if pred_data_y_prob.shape[-1] == 2:
                pred_data_y_prob = pred_data_y_prob[:, 1]
            y_prob_mat = pred_data_y_prob.reshape(len(ligands_list), len(acids_list))
            df_y_prob = pd.DataFrame(y_prob_mat, index=ligands_list, columns=acids_list)
            df_y_prob.to_csv(os.path.join(results_dir, "predictions_proba.csv"), index=True, header=True)
        elif pred_data_y_prob.shape[-1] > 2:
            for i in range(pred_data_y_prob.shape[-1]):
                y_prob_mat = pred_data_y_prob[:, i].reshape(len(ligands_list), len(acids_list))
                df_y_prob = pd.DataFrame(y_prob_mat, index=ligands_list, columns=acids_list)
                df_y_prob.to_csv(os.path.join(results_dir, "predictions_proba_{}.csv".format(type_idx2name.get(i, str(i)))), index=True, header=True)

def main():
    args, dataset_args, model_args, train_args = get_args()

    global_seed(args["seed"])

    if args["result_dirname_with_time"]:
        args["result_dir"] = args["result_dir"] + time.strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(args["result_dir"]):
        os.makedirs(args["result_dir"])
    else:
        print("Warning: Directory {} already exists and will be overwritten.".format(args["result_dir"]))
    
    # Load dataset
    dataset = load_dataset(dataset_args)

    # Load model
    model = prepare_model(args, model_args, train_args, dataset)

    # Evaluate model
    if "evaluate" in args and args["evaluate"]:
        results = evaluate_model(model, dataset, result_save_dir=args["result_dir"], train_args=train_args)
    
    # Predict
    if "predict" in args and args["predict"]:
        predict_model(model, dataset, train_args, result_save_dir=args["result_dir"])

if __name__ == "__main__":
    main()