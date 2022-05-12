# -*- coding: utf-8 -*-

import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from ml_project.enities import SplittingParams

logger = logging.getLogger(__name__)


def load_dataset(datapath: str) -> pd.DataFrame:
    """TBD"""

    try:
        data = pd.read_csv(datapath)
    except IOError:
        logger.error(f"The file cannot be loaded for provided path {datapath}")
        raise IOError

    return data


def split_data_train_val(data, config: SplittingParams):
    """TBD"""

    data_train, data_test = train_test_split(data,
                                             test_size=config.val_size,
                                             random_state=config.random_state)
    return data_train, data_test
