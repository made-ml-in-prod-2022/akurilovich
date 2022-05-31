# AA Kurilovich

Решение HW01 по курсу "ML в продакшене".

Работает через CLI. Параметры для обучения принимаются в виде конфига в .yaml

# Использование

### Установка
~~~
conda create --name <envname> --file requirements.txt
conda activate <envname>
setup.py install
~~~
### Обучение
~~~
python ml_project/main.py train_pipeline <path_to_config>
~~~

Примеры конфигов в .yaml формате находятся в папке configs
### Предсказание
~~~
python ml_project/main.py predict <path_to_saved_model> <path_to_data> <path_to_save_predictions>
~~~
### Тестирование
~~~
pytest tests/
~~~
