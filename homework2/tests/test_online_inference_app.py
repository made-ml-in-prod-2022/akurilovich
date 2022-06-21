import json

from fastapi.testclient import TestClient

from ..online_inference.online_inference_api import app, start_app

PATH_TO_DEFAULT_MODEL = r"tests/model_lr.pkl"

tested_app = TestClient(app)
start_app()


def test_root():
    response = tested_app.get("/")
    assert response.status_code == 200


def test_health():
    response = tested_app.get("/health")
    assert response.status_code == 200


def test_predict():

    data = {
        "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
                     "ca", "thal"],
        "data": [[69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0],
                 [69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0],
                 [69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0],
                 [69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0],
                 [69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0],
                 ]}

    response = tested_app.get("predict/", json=data)
    assert response.status_code == 200
    assert len(json.loads(response.text)["prediction"]) == 5


def test_predict_fails_on_wrong_data():

    data = {
        "features": ["age", "sex", "cp", "trestbps",],
        "data": [[69, 1, 0, 160, 234,],
                 [69, 1, 0, 160, 234,],
                 [69, 1, 0, 160, 234,],
                 [69, 1, 0, 160, 234,],
                 [69, 1, 0, 160, 234,],
                 ]}

    response = tested_app.get("predict/", json=data)
    assert response.status_code == 422
