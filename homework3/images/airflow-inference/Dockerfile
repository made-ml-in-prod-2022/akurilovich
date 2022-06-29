FROM airflow-ml-base:latest

COPY requirements.txt /requirements.txt
COPY inference.py /inference.py
RUN pip install --no-cache -r requirements.txt

ENTRYPOINT ["python", "inference.py"]