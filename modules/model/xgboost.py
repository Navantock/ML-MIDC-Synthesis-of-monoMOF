import xgboost as xgb

from ._base_classifier import TreeBased_Classifier
from ._base_regressor import Base_Regressor


class SynMOF_XGBoost_Classifier(TreeBased_Classifier):
    def __init__(self, **kwargs):
        super(SynMOF_XGBoost_Classifier, self).__init__()
        self.model = xgb.XGBClassifier(**kwargs)
        self.best_threshold: float = None

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    

class SynMOF_XGBoost_Regressor(Base_Regressor):
    def __init__(self, **kwargs):
        super(SynMOF_XGBoost_Regressor, self).__init__()
        self.model = xgb.XGBRegressor(**kwargs)
        self._attach_function_transformer()

    def predict(self, X):
        return self.model.predict(X)
