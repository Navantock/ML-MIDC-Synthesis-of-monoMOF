from .utils import get_all_classification_metrics, calc_precision, calc_recall, calc_f1_score_by_precision_recall, calc_accuracy, get_ROC_points, draw_ROC_curve

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import joblib
import shap

import os
import yaml
import abc
from typing import Optional


PREDEFINED_DESC_NAMES_BEESWARM = ["Acid pKa(DMF)", 
                                  "Acid SolFE dmf 0 eV",
                                  "Acid MolLogP",
                                  "Ligand pKa1(DMF)", 
                                  "Ligand SolFE dmf 0 eV",
                                  "Ligand MolLogP",
                                  "Ligand Distance COOH C-C", 
                                  "Ligand Distance C-C to Center", 
                                  "Ligand Distance Center to C-C Center Plane",
                                  "Ligand COOH C-C Gyration Radius",
                                  "Ligand COOH-COOH Dihedral Cosine"]


class Base_Classifier:
    def __init__(self):
        self.model = None
        self.train_results = {}
        self.val_results = {}
        self.explainer = None
    
    def train(self, x, y, y_class_num: int = 2, save_model: bool = True, save_roc_curve: bool = False, **kwargs):
        self.model.fit(x, y.reshape(-1))
        y_pred = torch.tensor(self.model.predict(x)).reshape(-1, 1)
        self.train_results["acc"] = calc_accuracy(y_pred, y)
        self.train_results["precision"] = calc_precision(y_pred, y, y_class_num)
        self.train_results["recall"] = calc_recall(y_pred, y, y_class_num)
        self.train_results["f1_score"] = calc_f1_score_by_precision_recall(self.train_results["precision"], self.train_results["recall"])

        if save_roc_curve:
            positive_prob = torch.tensor(self.predict_proba(x))[:, 1].reshape(-1, 1)
            fprs, tprs, thresholds, f1_scores = get_ROC_points(positive_prob, y, interval=0.001, get_f1_scores=True)
            self.best_threshold = float(thresholds[torch.argmax(torch.tensor(f1_scores))])
            self.train_results["best_threshold"] = self.best_threshold
            if "result_save_dir" in kwargs:
                draw_ROC_curve(fprs, tprs, os.path.join(kwargs["result_save_dir"], "train_roc_curve.svg"))
            else:
                print("Warning: result_save_dir not provided, cannot save ROC curve")

        print("Train Acc: ", self.train_results["acc"])
        print("Train Precision: ", self.train_results["precision"])
        print("Train Recall: ", self.train_results["recall"])
        print("Train F1 Score: ", self.train_results["f1_score"])

        if "acid_desc_names" in kwargs and "ligand_desc_names" in kwargs:
            if kwargs["importance"]:
                if kwargs["importance"] == "self":
                    self.train_results["feature_importances"] = {"Acid": {name: float(importance) for name, importance in zip(kwargs["acid_desc_names"], self.model.feature_importances_[:len(kwargs["acid_desc_names"])])},
                                                    "Ligand": {name: float(importance) for name, importance in zip(kwargs["ligand_desc_names"], self.model.feature_importances_[len(kwargs["acid_desc_names"]):])}}
                    if "other_desc_names" in kwargs:
                        self.train_results["feature_importances"]["Other"] = {name: float(importance) for name, importance in zip(kwargs["other_desc_names"], self.model.feature_importances_[len(kwargs["acid_desc_names"]) + len(kwargs["ligand_desc_names"]):])}
                else:
                    kwargs["save_stage"] = "train"
                    importance_results = self._get_feature_importances(self.model, x, y, **kwargs)
                    self.train_results.update(importance_results)
        
        if save_model and "result_save_dir" in kwargs:
            joblib.dump(self, os.path.join(kwargs["result_save_dir"], "model.pkl"))

    @abc.abstractmethod
    def test(self, x, y, y_class_num: int = 2, result_save_dir: Optional[str] = None, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def predict_proba(self, X):
        pass

    def _get_feature_importances(self, model, x, y, result_save_dir: str, importance: str, save_stage: str = "test", **kwargs):
        results = {}
        if importance == "permutation":
            result = permutation_importance(self.model, x, y, n_repeats=100, random_state=0)
            results["feature_importances"] = {"Acid": {name: float(importance) for name, importance in zip(kwargs["acid_desc_names"], result.importances_mean[:len(kwargs["acid_desc_names"])])}, "Ligand": {name: float(importance) for name, importance in zip(kwargs["ligand_desc_names"], result.importances_mean[len(kwargs["acid_desc_names"]):])}}
            if "other_desc_names" in kwargs:
                results["feature_importances"]["Other"] = {name: float(importance) for name, importance in zip(kwargs["other_desc_names"], result.importances_mean[len(kwargs["acid_desc_names"]) + len(kwargs["ligand_desc_names"]):])}
        elif importance == "shap":
            feature_names = ["Acid " + name for name in kwargs["acid_desc_names"]] + ["Ligand " + name for name in kwargs["ligand_desc_names"]] + kwargs.get("other_desc_names", [])
            if self.explainer is None:
                try:
                    self.explainer = shap.Explainer(model, feature_names=feature_names)
                except:
                    # Kernel Explainer
                    self.explainer = shap.KernelExplainer(model.predict_proba, np.asarray(x), feature_names=feature_names)
                    #self.explainer = shap.Explainer(model.predict_proba, np.asarray(x), feature_names=feature_names)
            shap_values = self.explainer(np.asarray(x))
            try:
                shap_interaction_values = self.explainer.shap_interaction_values(np.asarray(x)).values
            except:
                shap_interaction_values = np.array([])
            # sum the last dimension for multi output model
            abs_shap_values = np.sum(np.abs(shap_values.values), axis=-1) if len(shap_values.values.shape) == 3 else np.abs(shap_values.values)
            # average shap value as feature importance
            results["feature_importances"] = {"Acid": {name: float(importance) for name, importance in zip(kwargs["acid_desc_names"], np.mean(abs_shap_values[:, :len(kwargs["acid_desc_names"])], axis=0))}, "Ligand": {name: float(importance) for name, importance in zip(kwargs["ligand_desc_names"], np.mean(abs_shap_values[:, len(kwargs["acid_desc_names"]):len(kwargs["acid_desc_names"]) + len(kwargs["ligand_desc_names"])], axis=0))}}
            if "other_desc_names" in kwargs:
                results["feature_importances"]["Other"] = {name: float(importance) for name, importance in zip(kwargs["other_desc_names"], np.mean(abs_shap_values[:, len(kwargs["acid_desc_names"]) + len(kwargs["ligand_desc_names"]):], axis=0))}
            # save shap values
            if result_save_dir is not None and os.path.exists(result_save_dir):
                # save shap value
                np.savez(os.path.join(result_save_dir, f"{save_stage}_shap_values.npz"), shap_values=shap_values.values, shap_interaction_values=shap_interaction_values)
        else:
            print("Warning: importance method is not supported")
        return results


class TreeBased_Classifier(Base_Classifier):
    def __init__(self):
        super(TreeBased_Classifier, self).__init__()

    def test(self, x, y, y_class_num: int = 2, result_save_dir: Optional[str] = None, save_roc_curve: bool = False, **kwargs):
        th = self.best_threshold if self.best_threshold is not None else 0.5
        results = get_all_classification_metrics(torch.tensor(self.model.predict_proba(x)), 
                                                 torch.tensor(self.predict(x)).reshape(-1, 1),
                                                 y, 
                                                 y_class_num, 
                                                 result_save_dir=result_save_dir if save_roc_curve else None, 
                                                 save_roc_name="test_roc_curve")
        if "acid_desc_names" in kwargs and "ligand_desc_names" in kwargs:
            if kwargs["importance"]:
                if kwargs["importance"] == "self":
                    results["feature_importances"] = {"Acid": {name: float(importance) for name, importance in zip(kwargs["acid_desc_names"], self.model.feature_importances_[:len(kwargs["acid_desc_names"])])},
                                                      "Ligand": {name: float(importance) for name, importance in zip(kwargs["ligand_desc_names"], self.model.feature_importances_[len(kwargs["acid_desc_names"]):])}}
                    if "other_desc_names" in kwargs:
                        results["feature_importances"]["Other"] = {name: float(importance) for name, importance in zip(kwargs["other_desc_names"], self.model.feature_importances_[len(kwargs["acid_desc_names"]) + len(kwargs["ligand_desc_names"]):])}
                else:
                    importance_results = self._get_feature_importances(self.model, x, y, result_save_dir, **kwargs)
                    results.update(importance_results)
                    """ if "train_x" in kwargs and "train_y" in kwargs:
                        all_x = np.concatenate((kwargs["train_x"], x), axis=0)
                        all_y = np.concatenate((kwargs["train_y"], y), axis=0)
                        self._get_feature_importances(self.model, all_x, all_y, result_save_dir, **kwargs) """
                
        if result_save_dir is not None:
            if self.train_results:
                with open(os.path.join(result_save_dir, "train_results.yaml"), "w") as f:
                    yaml.dump(self.train_results, f, allow_unicode=True)
            with open(os.path.join(result_save_dir, "test_results.yaml"), "w") as f:
                yaml.dump(results, f, allow_unicode=True)
        return results
