import pandas as pd
import numpy as np
from ml_project.features import (
    extract_target,
    DatasetTransformer,
)


def test_extract_feature():

    data = pd.DataFrame(data=np.array([[1, 3.2, 0], [0, -10, 1], [1, 12, 1]]), columns=["C", "N", "T"])

    assert np.allclose(extract_target(data, "T"), np.array([0, 1, 1]))


def test_data_transformer():
    cat_features = "C"
    num_features = "N"
    data = pd.DataFrame(data=np.array([[1, 3.2, 0], [0, -10, 1], [1, 12, 1]]), columns=["C", "N", "T"])

    transformer = DatasetTransformer(cat_features, num_features)

    assert transformer.categorical_features == cat_features
    assert transformer.numerical_features == num_features

    assert isinstance(transformer.fit(data), DatasetTransformer)
