FROM rocm/tensorflow:latest

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /opt/project
COPY requirements.txt /opt/project
COPY ./src /opt/project


RUN apt install graphviz
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r /opt/project/requirements.txt

CMD ["python3", "runner.py"]
