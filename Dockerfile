FROM python:3.10 AS base

COPY nectarine nectarine
COPY requirements.txt .
COPY pyproject.toml .

RUN pip install .

FROM base AS transform
ENTRYPOINT [ "python -m nectarine.transform" ]

FROM base AS train
ENTRYPOINT [ "python -m nectarine.train" ]
