import os

import airflow.utils.dates
from airflow import DAG

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from docker.types import Mount

from airflow.models import Variable

DIR_DATA = "/data/raw/{{ ds }}"
DIR_MODEL = Variable.get("DIR_MODEL")
DIR_METRICS = "/data/metrics/{{ ds }}"
DIR_DATA_HOST_MACHINE = "/home/lemm1ng/Programing/Made_VK/ML_in_Prod/homework3/data/"
MOUNT = Mount(
            source=DIR_DATA_HOST_MACHINE,
            target="/data",
            type="bind",)

DATA_FILENAME = "data.csv"
MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "metrics.json"
DIR_PREDICTIONS = "/data/predictions/{{ ds }}"

with DAG(
    dag_id="inference_pipeline",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval="@daily",
) as dag:

    data_sensor = FileSensor(
        task_id="data_sensor",
        filepath=os.path.join("/opt/airflow/", DIR_DATA, DATA_FILENAME),
        poke_interval=5,
        retries=200,
    )

    model_sensor = FileSensor(
        task_id="model_sensor",
        filepath=os.path.join("/opt/airflow/", DIR_MODEL, MODEL_FILENAME),
        poke_interval=5,
        retries=200,
    )

    metrics_sensor = FileSensor(
        task_id="metrics_sensor",
        filepath=os.path.join("/opt/airflow/", DIR_METRICS, METRICS_FILENAME),
        poke_interval=5,
        retries=200,
    )


    inference = DockerOperator(
        task_id="inference",
        image="airflow-inference",
        command=f"--dir-in-data {DIR_DATA} --dir-in-model {DIR_MODEL} --dir-out-predictions {DIR_PREDICTIONS}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT],
    )

    #[data_sensor, metrics_sensor, model_sensor] >> inference
