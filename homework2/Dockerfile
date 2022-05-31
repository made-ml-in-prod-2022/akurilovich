# syntax=docker/dockerfile:1

FROM python:3.10-slim

WORKDIR .

COPY ./homework2 .
COPY ./../homework1 ./homework1/

RUN pip install --no-cache -r homework1/requirements.txt -r requirements.txt -e homework1/

ENV MODEL_URL=https://drive.google.com/uc?id=1W2wSmLjxvHJVpN2bTSDPQFm5tzFgnPeJ
ENV APP_IP=0.0.0.0

CMD ["python", "online_inference/online_inference_api.py"]
