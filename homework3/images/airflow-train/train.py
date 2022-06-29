import os
import pickle

import click
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

RANDOM_STATE = 300

DATA_FILENAME = "data_train.csv"
TARGET_FILENAME = "target_train.csv"

MODEL_FILENAME = "model.pkl"


@click.command("train")
@click.option("--dir-in-data", default=r"../data/processed")
@click.option("--dir-out-model", default=r"../data/models")
def train(
        dir_in_data: str,
        dir_out_model: str,
):

    os.makedirs(dir_out_model, exist_ok=True)
    X = pd.read_csv(os.path.join(dir_in_data, DATA_FILENAME)).values
    y = pd.read_csv(os.path.join(dir_in_data, TARGET_FILENAME)).values.ravel()

    model = Pipeline(
        [("imputer", SimpleImputer()),
         ("scaler", StandardScaler()),
         ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]
    )

    model.fit(X, y)
    with open(os.path.join(dir_out_model, MODEL_FILENAME), 'wb') as output_stream:
        pickle.dump(model, output_stream)


if __name__ == "__main__":
    train()
