import pytest
from ml_project.data.make_dataset import (
load_dataset,
split_data_train_val
)
from ml_project.enities import SplittingParams

DATA_PATH = "data.csv"


def test_load_dataset(tmpdir, synthetic_dataset):

    data_size = synthetic_dataset.shape[0]

    synthetic_dataset.to_csv(tmpdir.join(DATA_PATH), index=False)

    loaded_data = load_dataset(tmpdir.join(DATA_PATH))

    assert loaded_data.shape == synthetic_dataset.shape


def test_split_dataset(synthetic_dataset):

    data_size = synthetic_dataset.shape[0]
    config = SplittingParams(val_size=0.2, random_state=300)

    data_train, data_val = split_data_train_val(synthetic_dataset,config)

    assert data_val.shape[0] == int(0.2 * data_size)
