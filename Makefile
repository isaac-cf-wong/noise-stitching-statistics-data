.PHONY: all data tables figures test clean

all: tables figures

data/results data/psd:
	uv run python scripts/fetch_data.py

data: data/results

tables: data
	uv run python scripts/make_tables.py

figures: data
	uv run python scripts/make_fig_response_folding.py
	uv run python scripts/make_fig_parallel_responses.py
	uv run python scripts/make_fig_windows.py
	uv run python scripts/make_fig_tradeoff.py
	uv run python scripts/make_fig_c1_residuals.py

test:
	uv run pytest -q

clean:
	rm -rf output
