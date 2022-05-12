import click
import logging
import sys
from typing import NoReturn

from ml_project.enities import TrainingPipelineParams, read_training_pipeline_params
from ml_project.data import split_data_train_val, load_dataset
from ml_project.features import extract_target
from ml_project.models import (
    build_model_pipeline,
    calculate_metrics,
    save_metrics,
    make_prediction,
    save_prediction,
    save_model,
    load_model,
)

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(message)s")

handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.group()
def main():
    pass


@main.command(name="train_pipeline", help="Train the pipeline according to specified config")
@click.argument("config_path")
def train(config_path: TrainingPipelineParams) -> NoReturn:

    logging.info(msg="Starting the train_pipline procedure")

    config = read_training_pipeline_params(config_path)

    logging.info(msg="Config is extracted")

    data = load_dataset(config.input_data_path)

    logging.debug(msg="Data is loaded")

    data_train, data_val = split_data_train_val(data, config.splitting_params)

    logging.debug(msg="Data is splitted for train and validation")

    y_train = extract_target(data_train, config.feature_params.target_col)
    X_train = data_train.drop(columns=config.feature_params.target_col)

    logging.debug(msg="Train data is preprocessed")

    y_val = extract_target(data_val, config.feature_params.target_col)
    X_val = data_val.drop(columns=config.feature_params.target_col)

    logging.debug(msg="Validation data is preprocessed")

    model = build_model_pipeline(config)

    logging.debug(msg="Model pipeline has been built")

    model.fit(X_train, y_train)

    logging.debug(msg="Model is fitted")

    y_pred = make_prediction(model, X_val)
    metrics = calculate_metrics(y_val, y_pred)

    save_metrics(metrics, config.metric_path)

    logging.info(msg="Classification report is saved for validation dataset")

    save_model(model, config.output_model_path)

    logging.info(msg="Model is saved")


@main.command(name="predict")
@click.argument("model-path")
@click.argument("data-path")
@click.argument("prediction-save-path")
def predict(model_path, data_path, prediction_save_path) -> NoReturn:


    logging.info(msg="Starting the predict procedure")


    data = load_dataset(data_path)

    logging.info(msg="Data is loaded")

    model = load_model(model_path)

    logging.info(msg="Model is loaded")

    y_pred = model.predict(data)

    logging.debug(msg="Predictions are evaluated")

    save_prediction(y_pred, prediction_save_path)

    logging.info(msg="Predictions are saved")


if __name__ == "__main__":
    main()