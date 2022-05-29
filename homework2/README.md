# AA Kurilovich

Решение HW02 по курсу "ML в продакшене".

Сервис для online reference на FastApi Так же есть скрипты для запросов к сервису, работающие через CLI.
# Использование

### Установка

На текущий момент для десериализации моделт требуется пакет ml_project из homework1. Сначала нужно установить его.

Относительно root-а проекта
~~~
conda create --name <envname> --file requirements.txt
conda activate <envname>
cd homework1
python setup.py install
cd ../homework2
pip install -r requirements.txt
~~~
### Запуск API для online inference
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