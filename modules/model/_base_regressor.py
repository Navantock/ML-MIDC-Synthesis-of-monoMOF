import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.compose import TransformedTargetRegressor
import joblib
import shap

import os
import yaml
import abc
from typing import Optional


class Base_Regressor:
    def __init__(self):
        self.model = None
        self.train_results = {}
        self.valid_results = {}
        self.explainer = None

    def _attach_function_transformer(self):
        self.model = TransformedTargetRegressor(
            regressor=self.model,
            func=np.log,
            inverse_func=np.exp,
        )
    
    def train(self, x, y, save_model: bool = True, **kwargs):
        self.model.fit(x, y.reshape(-1))
        y_pred = torch.tensor(self.model.predict(x)).reshape(-1, 1)
        self.train_results["mae"] = float(torch.mean(torch.abs(y_pred - y)))
        self.train_results["rmse"] = float(torch.sqrt(torch.mean((y_pred - y) ** 2)))
        self.train_results["r2"] = float(self.model.score(x, y))

        print("Train MAE: {:.4f}".format(self.train_results["mae"]))
        print("Train RMSE: {:.4f}".format(self.train_results["rmse"]))
        print("Train R2: {:.4f}".format(self.train_results["r2"]))

        if "result_save_dir" in kwargs:
            plt.figure()
            y = y.numpy().reshape(-1)
            y_pred = y_pred.numpy().reshape(-1)
            max_y = max(max(y), max(y_pred))
            min_y = min(min(y), min(y_pred))
            y_range = max_y - min_y
            plt.scatter(y, y_pred, s=32)
            k, b = np.polyfit(y, y_pred, 1)
            plt.plot([min_y, max_y], [min_y, max_y], color="darkorange")
            plt.grid(alpha=0.5)
            plt.xlabel("True Acid Quantity (mmol)")
            plt.ylabel("Predicted Acid Quantity (mmol)")
            plt.xlim(0.0, max_y + 0.1 * y_range)
            plt.ylim(0.0, max_y + 0.1 * y_range)
            #plt.xlim(0.0, 0.5)
            #plt.ylim(0.0, 0.5)
            plt.savefig(os.path.join(kwargs["result_save_dir"], "train_true_pred.png"))
        
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

    def test(self, x, y, y_class_num: int = 2, result_save_dir: Optional[str] = None, **kwargs):
        y_pred = torch.tensor(self.model.predict(x)).reshape(-1, 1)

        results = {}
        results["mae"] = float(torch.mean(torch.abs(y_pred - y)))
        results["rmse"] = float(torch.sqrt(torch.mean((y_pred - y) ** 2)))
        results["r2"] = float(self.model.score(x, y))

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
                
        if result_save_dir is not None:
            if self.train_results:
                with open(os.path.join(result_save_dir, "train_results.yaml"), "w") as f:
                    yaml.dump(self.train_results, f, allow_unicode=True)
            with open(os.path.join(result_save_dir, "test_results.yaml"), "w") as f:
                yaml.dump(results, f, allow_unicode=True)
            
            plt.figure(figsize=(8, 8))
            y = y.numpy().reshape(-1)
            y_pred = y_pred.numpy().reshape(-1)
            max_y = max(max(y), max(y_pred))
            min_y = min(min(y), min(y_pred))
            if "train_x" in kwargs and "train_y" in kwargs:
                train_x = kwargs["train_x"]
                train_y = kwargs["train_y"].numpy().reshape(-1)
                train_y_pred = self.model.predict(train_x).reshape(-1)
                max_y = max(max_y, max(train_y), max(train_y_pred))
                min_y = min(min_y, min(train_y), min(train_y_pred))
                y_range = max_y - min_y
                plt.scatter(train_y, train_y_pred, s=32, label="Train", color="#548DCA")
            else:
                y_range = max_y - min_y
            plt.scatter(y, y_pred, s=32, label="Test", color="#65A30D")
            plt.plot([min_y, max_y], [min_y, max_y], color="#EE822F")
            plt.xlabel("True Acid Quantity (mmol)")
            plt.ylabel("Predicted Acid Quantity (mmol)")
            plt.xlim(0.0, max_y + 0.1 * y_range)
            plt.ylim(0.0, max_y + 0.1 * y_range)
            #plt.xlim(0.0, 0.5)
            #plt.ylim(0.0, 0.5)
            plt.grid(alpha=0.5)
            plt.annotate(f"Test $r^2 = {results['r2']:.3f}$", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
            if "train_x" in kwargs and "train_y" in kwargs:
                plt.annotate(f"Train $r^2 = {self.train_results['r2']:.3f}$", xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, ha='left', va='top')
                plt.legend()
            plt.savefig(os.path.join(result_save_dir, "test_true_pred.svg"))
            plt.close()

        return results

    @abc.abstractmethod
    def predict(self, X):
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
                    self.explainer = shap.KernelExplainer(model.predict, np.asarray(x), feature_names=feature_names)
                    #self.explainer = shap.Explainer(model.predict, np.asarray(x), feature_names=feature_names)
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
                np.savez(os.path.join(result_save_dir, f"{save_stage}_shap_values.npz"), shap_values=shap_values.values, shap_interaction_values=shap_interaction_values)
        else:
            print("Warning: importance method is not supported")
        return results