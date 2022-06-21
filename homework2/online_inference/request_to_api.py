from typing import NoReturn, List, Dict, Union
import json
import logging

import requests
import pandas as pd
import numpy as np
import click

logger = logging.getLogger(__name__)


def load_data(path: str) -> Dict[str, Union[List[str], List[List[Union[int, float]]]]]:

    data_in = pd.read_csv(path)
    features = list(data_in.columns)
    data_raw = []
    for ii in range(data_in.shape[0]):
        data_raw.append(list(data_in.iloc[ii, :].values.ravel()))

    return {"features": features, "data": data_raw}


def save_data(path: str, data: dict) -> NoReturn:

    np.savetxt(path, np.array(data["prediction"]), fmt="%d")


def do_request(url:str, data: dict) -> dict:

    result = requests.get(url, json=data)

    return json.loads(result.text)


@click.command()
@click.argument("server-url")
@click.argument("path-to-data")
@click.argument("prediction-save-path")
def main(server_url, path_to_data, prediction_save_path):

    data_in = load_data(path_to_data)
    result = do_request(server_url, data_in)
    save_data(prediction_save_path, result)


if __name__ == "__main__":
    main()