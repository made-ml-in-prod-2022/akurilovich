from typing import NoReturn, Union
import pickle
import json
import logging

import pandas as pd
import numpy as np

from ml_project.features import DatasetTransformer
from ml_project.enities import TrainingPipelineParams
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _select_model(model_type:str, random_state: int) -> Union[RandomForestClassifier, LogisticRegression]:
    """TBD"""

    if model_type == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_type == "LogisticRegression":
        model = LogisticRegression(random_state=random_state)
    else:
        logger.error(msg="Wrong model_type is specified")
        raise ValueError("model_type should be either 'RandomForestClassifier' or 'LogisticRegression'")

    logger.info(msg=f"Selected model_type: {model_type}, random_state: {random_state}")

    return model


def build_model_pipeline(config: TrainingPipelineParams) -> Pipeline:
    """TBD"""

    model = _select_model(config.train_params.model_type, config.train_params.random_state)

    pipeline = Pipeline([("transformer", DatasetTransformer(config.feature_params.categorical_features,
                                                           config.feature_params.numerical_features)),
                         ("model", model)])

    logger.info(msg="Model with feature transformer is created")

    return pipeline


def save_model(model: Pipeline, save_path: str) -> NoReturn:
    """TBD"""

    with open(save_path, "wb") as output_stream:
        pickle.dump(model, output_stream)


def load_model(load_path: str):
    """TBD"""

    with open(load_path, "rb") as input_stream:
        return pickle.load(input_stream)


def make_prediction(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """TBD"""

    return model.predict(X)


def save_prediction(y_pred: np.ndarray, save_path: str) -> NoReturn:
    """TBD"""

    np.savetxt(save_path, y_pred)


def calculate_metrics(y_pred, y_true: Union[pd.Series, np.ndarray]) -> dict:
    """TBD"""

    return classification_report(y_true, y_pred)


def save_metrics(metrics: dict, save_path: str):
    """TBD"""

    with open(save_path, "w") as output_stream:
        json.dump(metrics, output_stream)
