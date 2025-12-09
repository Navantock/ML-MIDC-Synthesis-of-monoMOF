from ._base_classifier import Base_Classifier
from .decision_tree import SynMOF_DT_Classifier
from .svm import SynMOF_SVM_Classifier, SynMOF_SVM_Regressor
from .mlp import SynMOF_MLP_Classifier, SynMOF_MLP_Regressor
from .random_forest import SynMOF_RF_Classifier, SynMOF_RF_Regressor
from .xgboost import SynMOF_XGBoost_Classifier, SynMOF_XGBoost_Regressor
from .linear import SynMOF_Linear_Regressor, SynMOF_LASSO_Regressor
from typing import Dict


Classifier_Dict : Dict[str, Base_Classifier] = {
    "DecisionTree": SynMOF_DT_Classifier,
    "SVM": SynMOF_SVM_Classifier,
    "MLP": SynMOF_MLP_Classifier,
    "RandomForest": SynMOF_RF_Classifier,
    "XGBoost": SynMOF_XGBoost_Classifier
}

Regressor_Dict : Dict[str, Base_Classifier] = {
    "Linear": SynMOF_Linear_Regressor,
    "LASSO": SynMOF_LASSO_Regressor,
    "SVM": SynMOF_SVM_Regressor,
    "MLP": SynMOF_MLP_Regressor,
    "RandomForest": SynMOF_RF_Regressor,
    "XGBoost": SynMOF_XGBoost_Regressor
}