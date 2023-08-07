FROM python:3.10

COPY src src

COPY requirements.txt .
COPY pyproject.toml .

RUN pip install .
