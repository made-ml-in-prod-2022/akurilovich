import airflow.utils.dates
from airflow import DAG

from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

DIR_GENERATED_DATA = "/data/raw/{{ ds }}"
DIR_DATA_HOST_MACHINE = "/home/lemm1ng/Programing/Made_VK/ML_in_Prod/homework3/data/"

# Host folder replace before the startup
MOUNT = Mount(
            source=DIR_DATA_HOST_MACHINE,
            target="/data",
            type="bind",)

with DAG(
    dag_id="generate_data_pipeline",
    start_date=airflow.utils.dates.days_ago(10),
    schedule_interval="@daily",
) as dag:

    generate_data = DockerOperator(
        task_id="generate_data",
        image="airflow-generate-data",
        command=f"--dir-out {DIR_GENERATED_DATA}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT],
    )

    generate_data

