.PHONY: help install run test lint format docs docs-test

help:
	@python -c "print('Targets: install, run, test, lint, format, docs, docs-test')"

install:
	pip install -r requirements.txt

run:
	streamlit run streamlit_app.py

test:
	python -m pytest

lint:
	ruff check . --config .code_quality/ruff.toml

format:
	ruff format . --config .code_quality/ruff.toml

docs:
	mkdocs serve

docs-test:
	mkdocs build --strict

