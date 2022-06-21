import logging
import pickle
import os
import sys
from typing import List, Union, Optional

import pandas as pd

import gdown

from sklearn.pipeline import Pipeline

from fastapi import FastAPI, Response, status
from pydantic import BaseModel, conlist
import uvicorn

PATH_TO_DEFAULT_MODEL = r"./models/model_lr.pkl"
DEFAULT_APP_IP = "127.0.0.1"


class HeartDataSubmission(BaseModel):
    features: List[str]
    data: List[conlist(Union[int, float], min_items=13, max_items=13)]


class HeartDataDecision(BaseModel):
    prediction: List[int]


def load_model(load_path: str) -> Pipeline:
    """TBD"""

    with open(load_path, "rb") as input_stream:
        return pickle.load(input_stream)


def load_model_gdrive(load_path: str) -> Pipeline:

    model_path = gdown.download(load_path, "models//model.pkl")
    return load_model(model_path)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(message)s")

handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

app = FastAPI()
model: Optional[Pipeline] = None


@app.get("/")
async def root():
    return {"message": "This is the API for HeartCleavelend inference"}


@app.get("/health")
async def health(response: Response):
    if model is None:
        response.status_code = status.HTTP_424_FAILED_DEPENDENCY
        msg = "Status 424. The model is missing"
    elif isinstance(model, Pipeline):
        response.status_code = status.HTTP_200_OK
        msg = "Status 200. The model is ready"
    else:
        response.status_code = status.HTTP_409_CONFLICT
        msg = "Status 409. Wrong model is specified. It should be an instance of sklearn.pipelines.Pipeline class"

    return {"health_status": msg}


@app.get("/predict", response_model=HeartDataDecision)
def make_prediction(submission: HeartDataSubmission):

    logger.info("Starting the prediction on inference")
    data = pd.DataFrame(data=submission.data, columns=submission.features)
    logger.info("Inference data is loaded")
    result = list(model.predict(data))
    logger.info("Result is obtained")
    prediction = {"prediction": result}
    logger.info(f"Prediction is made: {len(prediction['prediction'])} entities")
    return prediction


@app.on_event("startup")
def start_app():

    global model

    try:
        logger.info(f"Loading model from Google Drive: {os.getenv('MODEL_URL', '')}")
        model = load_model_gdrive(os.getenv("MODEL_URL", ""))
        logger.info(f"Model from Google Drive is loaded")
    except Exception as err:
        logger.info(f"Model cannot be loaded from Google Drive: {err}. Loading the default model.")
        model = load_model(PATH_TO_DEFAULT_MODEL)
        logger.info(f"Default model is loaded")


if __name__ == "__main__":

    if isinstance(os.getenv("APP_IP", None), str):
        app_ip = os.getenv("APP_IP", None)
    else:
        app_ip = DEFAULT_APP_IP

    uvicorn.run(
        "online_inference_api:app",
        host=app_ip,
        port=os.getenv("PORT", default=8000),
    )