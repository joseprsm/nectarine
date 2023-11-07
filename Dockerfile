FROM python:3.10 AS base

COPY nectarine nectarine
COPY requirements.txt .
COPY pyproject.toml .

RUN python -m build
