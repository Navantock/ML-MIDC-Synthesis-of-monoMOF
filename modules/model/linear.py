from sklearn.linear_model import LinearRegression, Lasso, PoissonRegressor
from ._base_regressor import Base_Regressor


class SynMOF_Linear_Regressor(Base_Regressor):
    def __init__(self, **kwargs):
        super(SynMOF_Linear_Regressor, self).__init__()
        self.model = LinearRegression(**kwargs)
        self._attach_function_transformer()

    def predict(self, X):
        return self.model.predict(X)

class SynMOF_LASSO_Regressor(Base_Regressor):
    def __init__(self, **kwargs):
        super(SynMOF_LASSO_Regressor, self).__init__()
        self.model = Lasso(**kwargs)
        self._attach_function_transformer()

    def predict(self, X):
        return self.model.predict(X)

class SynMOF_Poisson_Regressor(Base_Regressor):
    def __init__(self, **kwargs):
        super(SynMOF_Poisson_Regressor, self).__init__()
        self.model = PoissonRegressor(**kwargs)

    def predict(self, X):
        return self.model.predict(X)