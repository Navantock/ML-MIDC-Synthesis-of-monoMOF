import torch
from sklearn import svm
import os
import yaml

from ._base_classifier import Base_Classifier
from ._base_regressor import Base_Regressor
from .utils import get_all_classification_metrics


class SynMOF_SVM_Classifier(Base_Classifier):
    def __init__(self, **kwargs):
        super(SynMOF_SVM_Classifier, self).__init__()
        self.model = svm.SVC(**kwargs)
        self.best_threshold: float = None

    def test(self, x, y, y_class_num: int = 2, result_save_dir: str = None, **kwargs):
        th = self.best_threshold if self.best_threshold is not None else 0.5
        results = get_all_classification_metrics(torch.tensor(self.predict_proba(x)), 
                                                 torch.tensor(self.predict(x)).reshape(-1, 1),
                                                 y, 
                                                 y_class_num, 
                                                 result_save_dir, 
                                                 save_roc_name="test_roc_curve")
        if "acid_desc_names" in kwargs and "ligand_desc_names" in kwargs:
            if self.model.get_params()["kernel"] == "linear":
                results["feature_coefs"] = {"Acid": {name: float(importance) for name, importance in zip(kwargs["acid_desc_names"], self.model.coef_[0][:len(kwargs["acid_desc_names"])])},
                                            "Ligand": {name: float(importance) for name, importance in zip(kwargs["ligand_desc_names"], self.model.coef_[0][len(kwargs["acid_desc_names"]):])}}
            if kwargs["importance"]:
                importance_results = self._get_feature_importances(self.model, x, y, result_save_dir, **kwargs)
                results.update(importance_results)
        if result_save_dir is not None:
            if self.train_results:
                with open(os.path.join(result_save_dir, "train_results.yaml"), "w") as f:
                    yaml.dump(self.train_results, f, allow_unicode=True)
            with open(os.path.join(result_save_dir, "test_results.yaml"), "w") as f:
                yaml.dump(results, f, allow_unicode=True)
        return results

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class SynMOF_SVM_Regressor(Base_Regressor):
    def __init__(self, **kwargs):
        super(SynMOF_SVM_Regressor, self).__init__()
        self.model = svm.SVR(**kwargs)
        self._attach_function_transformer()

    def predict(self, X):
        return self.model.predict(X)