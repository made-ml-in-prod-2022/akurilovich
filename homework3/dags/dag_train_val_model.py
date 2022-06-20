import os

import airflow.utils.dates
from airflow import DAG

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.dummy import DummyOperator
from docker.types import Mount

DIR_GENERATED_DATA = "/data/raw/{{ ds }}"
DIR_PROCESSED_DATA = "/data/processed/{{ ds }}"
DIR_MODEL = "/data/models/{{ ds }}"
DIR_METRICS = "/data/metrics/{{ ds }}"
DIR_DATA_HOST_MACHINE = "/home/lemm1ng/Programing/Made_VK/ML_in_Prod/homework3/data/"
MOUNT = Mount(
            source=DIR_DATA_HOST_MACHINE,
            target="/data",
            type="bind",)

DATA_FILENAME = "data.csv"
TARGET_FILENAME = "target.csv"

VAL_SIZE = 0.2

with DAG(
    dag_id="model_train_val_pipeline",
    start_date=airflow.utils.dates.days_ago(10),
    schedule_interval="@weekly",
) as dag:

    data_sensor = FileSensor(
        task_id="data_sensor",
        filepath=os.path.join(DIR_GENERATED_DATA, DATA_FILENAME),
        poke_interval=5,
        retries=200,
    )

    target_sensor = FileSensor(
        task_id="target_sensor",
        filepath=os.path.join(DIR_GENERATED_DATA, TARGET_FILENAME),
        poke_interval=5,
        retries=200,
    )

    split_data = DockerOperator(
        task_id="split_data",
        image="airflow-split-data",
        command=f"--dir-in {DIR_GENERATED_DATA} --dir-out {DIR_PROCESSED_DATA} --val-size {VAL_SIZE}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT],
    )

    train_model = DockerOperator(
        task_id="train_model",
        image="airflow-train",
        command=f"--dir-in-data {DIR_PROCESSED_DATA} --dir-out-model {DIR_MODEL}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT],
    )

    validate_model = DockerOperator(
        task_id="validate_model",
        image="airflow-validate",
        command=f"--dir-in-data {DIR_PROCESSED_DATA} --dir-in-model {DIR_MODEL} --dir-out-metrics {DIR_METRICS}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT],
    )


    [data_sensor, target_sensor] >> split_data

    split_data >> [train_model, validate_model]
    train_model >> validate_model

