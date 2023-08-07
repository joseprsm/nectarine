FROM python:3.10

COPY nectarine nectarine
COPY requirements.txt .
COPY pyproject.toml .

RUN pip install .
