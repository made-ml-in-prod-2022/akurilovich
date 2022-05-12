# -*- coding: utf-8 -*-

from typing import List, NoReturn

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class DatasetTransformer(BaseEstimator, TransformerMixin):
    """TBD"""

    def __init__(self,
                 categorical_features: List[str],
                 numerical_features: List[str]) -> NoReturn:
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        self.numerical_transformer = StandardScaler()
        self.categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')

    def fit(self, X: pd.DataFrame, y=None) -> 'DatasetTransformer':
        """TBD"""

        X_ = X.copy()
        X_cat = X_.loc[:, self.categorical_features].values
        X_num = X_.loc[:, self.numerical_features].values

        if len(self.categorical_features) == 1:
            X_cat = X_cat.reshape(-1,1)
        if len(self.numerical_features) == 1:
            X_num = X_num.reshape(-1,1)

        self.numerical_transformer.fit(X_num)
        self.categorical_transformer.fit(X_cat)

        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.array:
        """TBD"""

        X_ = X.copy()
        X_cat = X_.loc[:, self.categorical_features].values
        X_num = X_.loc[:, self.numerical_features].values

        if len(self.categorical_features) == 1:
            X_cat = X_cat.reshape(-1,1)
        if len(self.numerical_features) == 1:
            X_num = X_num.reshape(-1,1)

        X_ = np.concatenate((self.categorical_transformer.transform(X_cat),
                             self.numerical_transformer.transform(X_num)), axis=1)

        return X_


def extract_target(data: pd.DataFrame, column_name: str) -> np.ndarray:
    """TBD"""

    return data.loc[:, column_name].copy().values
