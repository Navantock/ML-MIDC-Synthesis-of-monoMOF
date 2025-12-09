import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ._base_classifier import TreeBased_Classifier
from ._base_regressor import Base_Regressor


class SynMOF_RF_Classifier(TreeBased_Classifier):
    def __init__(self, **kwargs):
        super(SynMOF_RF_Classifier, self).__init__()
        self.model = RandomForestClassifier(**kwargs)
        self.best_threshold: float = None

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class SynMOF_RF_Regressor(Base_Regressor):
    def __init__(self, **kwargs):
        super(SynMOF_RF_Regressor, self).__init__()
        self.model = RandomForestRegressor(**kwargs)
        self._attach_function_transformer()

    def predict(self, X):
        return self.model.predict(X)