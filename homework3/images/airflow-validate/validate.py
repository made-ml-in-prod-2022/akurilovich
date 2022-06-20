import os
import pickle
import json

import click
from sklearn.metrics import classification_report
import pandas as pd

RANDOM_STATE = 300

DATA_FILENAME = "data_val.csv"
TARGET_FILENAME = "target_val.csv"

MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "metrics.json"

@click.command("validate")
@click.option("--dir-in-data", default=r"../data/processed")
@click.option("--dir-in-model", default=r"../data/models")
@click.option("--dir-out-metrics", default=r"../data/metrics")
def validate(
        dir_in_data: str,
        dir_in_model: str,
        dir_out_metrics: str,
):

    os.makedirs(dir_out_metrics, exist_ok=True)
    X = pd.read_csv(os.path.join(dir_in_data, DATA_FILENAME)).values
    y = pd.read_csv(os.path.join(dir_in_data, TARGET_FILENAME)).values.ravel()

    with open(os.path.join(dir_in_model, MODEL_FILENAME), 'rb') as input_stream:
        model = pickle.load(input_stream)

    y_pred= model.predict(X)

    metrics = classification_report(y, y_pred)
    with open(os.path.join(dir_out_metrics, METRICS_FILENAME), 'w') as output_stream:
        json.dump(metrics, output_stream)


if __name__ == "__main__":
    validate()
