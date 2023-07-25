FROM python:3.10 AS base

COPY nectarine nectarine
COPY requirements requirements
COPY pyproject.toml .

FROM base AS transform

RUN pip install -r requirements/transform.txt
RUN pip install .

ENTRYPOINT [ "python -m nectarine.transform" ]

FROM base AS train

RUN pip install -r requirements/train.txt
RUN pip install .

ENTRYPOINT [ "python -m nectarine.train" ]
