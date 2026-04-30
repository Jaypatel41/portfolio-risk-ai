.PHONY: install test lint clean run-risk run-market run-explain run-stress run-serve

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"

test:
	. .venv/bin/activate && pytest

lint:
	. .venv/bin/activate && ruff check src tests

run-risk:
	. .venv/bin/activate && timecell risk examples/balanced.json

run-market:
	. .venv/bin/activate && timecell market

run-explain:
	. .venv/bin/activate && timecell explain examples/balanced.json --critic

run-stress:
	. .venv/bin/activate && timecell stress examples/balanced.json "what if BTC crashes 70% and gold rallies 20%?"

run-serve:
	. .venv/bin/activate && streamlit run app.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
