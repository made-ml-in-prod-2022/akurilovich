import os

import click
from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_STATE = 300

DATA_FILENAME = "data.csv"
TARGET_FILENAME = "target.csv"

DATA_TRAIN_FILENAME = "data_train.csv"
DATA_VAL_FILENAME = "data_val.csv"

TARGET_TRAIN_FILENAME = "target_train.csv"
TARGET_VAL_FILENAME = "target_val.csv"


@click.command("split")
@click.option("--dir-in", default=r"../data/raw")
@click.option("--dir-out", default=r"../data/raw")
@click.option("--val-size", default=0.2)
def split(
        dir_in: str,
        dir_out: str,
        val_size: float,
):

    os.makedirs(dir_out, exist_ok=True)
    X = pd.read_csv(os.path.join(dir_in, DATA_FILENAME))
    y = pd.read_csv(os.path.join(dir_in, TARGET_FILENAME))
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=RANDOM_STATE,
    )

    pd.DataFrame(data=X_train).to_csv(os.path.join(dir_out, DATA_TRAIN_FILENAME), index=False)
    pd.DataFrame(data=X_val).to_csv(os.path.join(dir_out, DATA_VAL_FILENAME), index=False)

    pd.DataFrame(data=y_train).to_csv(os.path.join(dir_out, TARGET_TRAIN_FILENAME), index=False)
    pd.DataFrame(data=y_val).to_csv(os.path.join(dir_out, TARGET_VAL_FILENAME), index=False)


if __name__ == "__main__":
    split()
