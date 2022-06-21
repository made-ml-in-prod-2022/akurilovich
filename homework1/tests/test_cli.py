import pytest
import os
import sys
import numpy as np

from unittest.mock import patch
from ml_project.main import main

CONFIG_PATH = "train_config_lr.yaml"

DATA_PATH = "data.csv"
METRICS_PATH = "metrics.json"
MODEL_PATH = "model.pkl"
PREDICT_PATH = "result.txt"


def test_train_pipeline(tmpdir, synthetic_dataset, config):

    synthetic_dataset.to_csv(tmpdir.join(DATA_PATH), index=False)
    with open(tmpdir.join(CONFIG_PATH), "w") as output_stream:
        output_stream.write(config)

    command = ["main.py", "train_pipeline", tmpdir.join(CONFIG_PATH)]

    with patch.object(sys, "argv", command):
        with pytest.raises(SystemExit) as err:
            main()
    assert err.type == SystemExit
    assert os.path.isfile(tmpdir.join(DATA_PATH)), "Data file is not created"
    assert os.path.isfile(tmpdir.join(MODEL_PATH)), "Model is not saved"
    assert os.path.isfile(tmpdir.join(METRICS_PATH)), "Metrics is not saved"


def test_predict(tmpdir, synthetic_dataset, config):

    synthetic_dataset.to_csv(tmpdir.join(DATA_PATH), index=False)
    with open(tmpdir.join(CONFIG_PATH), "w") as output_stream:
        output_stream.write(config)

    command = ["main.py", "train_pipeline", tmpdir.join(CONFIG_PATH)]

    with patch.object(sys, "argv", command):
        with pytest.raises(SystemExit) as err:
            main()

    command = ["main.py", "predict", tmpdir.join(MODEL_PATH), tmpdir.join(DATA_PATH), tmpdir.join(PREDICT_PATH)]

    with patch.object(sys, "argv", command):
        with pytest.raises(SystemExit) as err:
            main()

    assert os.path.isfile(tmpdir.join(PREDICT_PATH)), "Predictions are not saved"
    assert np.loadtxt(tmpdir.join(PREDICT_PATH)).size == synthetic_dataset.shape[0], "Wrong length of vector with predictions"
