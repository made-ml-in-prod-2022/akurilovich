import os

import click
from sklearn.datasets import make_classification
import pandas as pd

DATA_FILENAME = "data.csv"
TARGET_FILENAME = "target.csv"


@click.command("generate")
@click.option("--dir-out", default=r"../data/raw")
@click.option("--data-size", default=1000)
@click.option("--feature-num-tot", default=10)
@click.option("--feature-num-meaningful", default=3)
def generate(
        dir_out: str,
        data_size: int,
        feature_num_tot: int,
        feature_num_meaningful: int,
):

    os.makedirs(dir_out, exist_ok=True)
    X, y = make_classification(
        n_samples=data_size,
        n_features=feature_num_tot,
        n_informative=feature_num_meaningful,
        n_classes=2,
    )

    pd.DataFrame(data=X).to_csv(os.path.join(dir_out, DATA_FILENAME), index=False)
    pd.DataFrame(data=y).to_csv(os.path.join(dir_out, TARGET_FILENAME), index=False)


if __name__ == "__main__":
    generate()