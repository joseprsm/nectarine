FROM python:3.10 AS base

COPY nectarine nectarine
COPY requirements.txt .
COPY pyproject.toml .

RUN python -m build


FROM tensorflow/tensorflow:2.14.0 AS train

COPY --from=base dist/nectarine-0.0.0-py3-none-any.whl nectarine.whl

RUN pip install nectarine.whl
