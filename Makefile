.PHONY: help install test lint format

deps-mac:
	brew install poppler swig

deps-linux:
	apt install build-essential libpoppler-cpp-dev pkg-config

install:
	pdm install

test:
	pdm run pytest

lint:
	pdm run ruff chatlocal --fix
	pdm run mypy chatlocal

format:
	pdm run black chatlocal