from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from eczema_model.get_data import read_data
import pandas as pd
from eczema_model.config import config


class SplitTrainTest(BaseEstimator, TransformerMixin):
    """Split data into train and test"""

    def __init__(self, variables=pd.DataFrame) -> None:
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit template."""

        return self

    def transform(self, X: pd.DataFrame):
        """Apply the transforms to the dataframe."""

        X = self.variables.copy()
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

        return X


class MissingDataHandler(BaseEstimator, TransformerMixin):
    """Fill in missing value."""

    def __init__(self, variables=None):

        self.variables = variables

    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        X = X.copy()
        X.fillna(method="ffill", inplace=True)
        X.fillna(method="bfill", inplace=True)
        return X


class BestFeaturesSelected(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X


# from sklearn.base import BaseEstimator
# class CustomPredictor(BaseEstimator):

# def __init__(self, estimator = SGDClassifier()):
#     """
#     A Custom BaseEstimator to run multiple models
#     """ 

#     self.estimator = estimator


# def fit(self, X, y=None, **kwargs):
#     self.estimator.fit(X, y)
#     return self


# def predict(self, X, y=None):
#     return self.estimator.predict(X)


# def predict_proba(self, X):
#     return self.estimator.predict_proba(X)


# def score(self, X, y):
    # return self.estimator.score(X, y)