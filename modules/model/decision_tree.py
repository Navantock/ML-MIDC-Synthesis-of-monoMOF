from sklearn.tree import DecisionTreeClassifier

from ._base_classifier import TreeBased_Classifier


class SynMOF_DT_Classifier(TreeBased_Classifier):
    def __init__(self, **kwargs):
        super(SynMOF_DT_Classifier, self).__init__()
        self.model = DecisionTreeClassifier(**kwargs)
        self.best_threshold: float = None

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

