.PHONY: help install test lint format


install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run flake8 chatlocal
	poetry run mypy chatlocal

format:
	poetry run isort -v chatlocal
	poetry run black chatlocal