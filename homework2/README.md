# AA Kurilovich

Решение HW02 по курсу "ML в продакшене".

Сервис для online reference на FastApi Так же есть скрипты для запросов к сервису, работающие через CLI.

# Использование

### Установка

На текущий момент для десериализации модели требуется пакет ml_project из homework1. Сначала нужно установить его.

Относительно root-а проекта

~~~
conda create -n <envname> pip=21.2.4
source activate -n <envname>
pip install -r ./homework1/requirements.txt
pip install -e homework1/
pip install -r ./homework2/requirements.txt
~~~

###  Запуск API для online inference

Запуск из ./homework2/

~~~
python online_inference/online_inference_api.py
~~~

### Предсказание

~~~
python online_inference/request_to_api.py <server_url> <dath_to_data> <path_to_save_predictions>
~~~

### Тестирование

~~~
pytest tests/
~~~

### Сборка Docker образа

из homework2/ запустить

~~~
docker build -t hw02_online_inference:v1 -f Dockerfile ../
~~~

### Запуск на основе Online inference API на основе Docker Container

Локальная сборка образа:

~~~
docker run -p 8000:8000 -e MODEL_URL=<path to gdrive> <image_name>
~~~

Загрузка с Docker Hub

~~~
docker pull lemm1ng/hw02_online_inference:v1
~~~
