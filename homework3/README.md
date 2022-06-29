# Kurilovich AA

Решение HW03 по курсу "ML в продакшене".

Airflow: Data Pipelines


Развернуть airflow, предварительно собрав контейнеры

~~~
sudo docker compose up --build
~~~
Использование:
1. Запустить DAG generate_data_pipeline
2. Запустить DAG model_train_val_pipeline
3. В Admin -> Variables указать директорию с актуальной моделью в Variable DIR_MODEL
4. Запустить DAG inference_pipeline

Монтируется следующая дирректория в docker контейнеры. Поменять в dags/... перед разворачиванием airflow
~~~
DIR_DATA_HOST_MACHINE = "/home/lemm1ng/Programing/Made_VK/ML_in_Prod/homework3/data/"
~~~