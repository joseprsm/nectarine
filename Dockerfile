ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}

WORKDIR /xtrax

COPY xtrax xtrax
COPY pyproject.toml .
COPY requirements.txt .

RUN pip install .
