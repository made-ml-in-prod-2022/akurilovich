import os
import pickle

import click
import pandas as pd

RANDOM_STATE = 300

DATA_FILENAME = "data.csv"

MODEL_FILENAME = "model.pkl"
PREDICTION_FILENAME = "prediction.csv"


@click.command("inference")
@click.option("--dir-in-data", default=r"../data/processed")
@click.option("--dir-in-model", default=r"../data/models")
@click.option("--dir-out-predictions", default=r"../data/predictions")
def inference(
        dir_in_data: str,
        dir_in_model: str,
        dir_out_predictions: str,
):

    os.makedirs(dir_out_predictions, exist_ok=True)
    X = pd.read_csv(os.path.join(dir_in_data, DATA_FILENAME)).values

    with open(os.path.join(dir_in_model, MODEL_FILENAME), 'rb') as input_stream:
        model = pickle.load(input_stream)

    y_pred = model.predict(X)
    pd.DataFrame(data=y_pred).to_csv(os.path.join(dir_out_predictions, PREDICTION_FILENAME), index=False)


if __name__ == "__main__":
    inference()
