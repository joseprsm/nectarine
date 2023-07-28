FROM python:3.10

COPY nectarine nectarine
COPY requirements requirements
COPY pyproject.toml .
RUN pip install .
