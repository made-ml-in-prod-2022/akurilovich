import pytest
import pandas as pd
import numpy as np
import os

DATASET_COLUMNS = ['age', 'sex', 'cp',
                   'trestbps', 'chol', 'fbs',
                   'restecg', 'thalach', 'exang',
                   'oldpeak', 'slope', 'ca', 'thal', 'condition']

DATASET_SIZE = 100

@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    data = pd.DataFrame(columns=DATASET_COLUMNS)
    data.age = np.random.randint(0, 100, size=DATASET_SIZE)
    data.sex = np.random.choice([0, 1], size=DATASET_SIZE)
    data.cp = np.random.choice([0, 1, 2, 3], size=DATASET_SIZE)
    data.trestbps = np.random.randint(94, 200, size=DATASET_SIZE)
    data.chol = np.random.randint(126, 564, size=DATASET_SIZE)
    data.fbs = np.random.choice([0, 1], size=DATASET_SIZE)
    data.restecg = np.random.choice([0, 1, 2], size=DATASET_SIZE)
    data.thalach = np.random.randint(71, 202, size=DATASET_SIZE)
    data.exang = np.random.choice([0, 1], size=DATASET_SIZE)
    data.oldpeak = np.random.uniform(0, 6.2, size=DATASET_SIZE).round(1)
    data.slope = np.random.choice([0, 1, 2], size=DATASET_SIZE)
    data.ca = np.random.choice([0, 1, 2, 2], size=DATASET_SIZE)
    data.thal = np.random.choice([0, 1, 2], size=DATASET_SIZE)
    data.condition = np.random.choice([0, 1], size=DATASET_SIZE)

    return data

@pytest.fixture
def config(tmpdir) -> str:

    config = f'''input_data_path: {tmpdir.join("data.csv")}
output_model_path: {tmpdir.join("model.pkl")}
metric_path: {tmpdir.join("metrics.json")}
splitting_params:
  val_size: 0.3
  random_state: 300
train_params:
  model_type: "LogisticRegression"
  random_state: 333
feature_params:
  categorical_features:
    - "sex"
    - "cp"
    - "fbs"
    - "restecg"
    - "exang"
    - "slope"
    - "ca"
    - "thal"
  numerical_features:
    - "age"
    - "trestbps"
    - "chol"
    - "thalach"
    - "oldpeak"
  target_col: "condition"
    '''

    return config
