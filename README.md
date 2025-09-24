# Quantitative Trading Research

This repository collects quantitative trading and stock selection strategies built around the Chinese A-share market. Each strategy lives in its own folder under `strategies/` so ideas stay isolated and reproducible.

## Repository layout

- `strategies/` individual strategy workspaces (data, notebooks, code, docs)
- `scripts/` reusable command line utilities (data sync, batch jobs, etc.)
- `docs/` shared documentation and research notes
- `requirements.txt` base Python dependencies for all strategies

## Getting started

1. Create a virtual environment and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Explore strategy folders for detailed instructions.
3. Add new strategies by copying the template layout inside `strategies/`.

## Strategy index

- [`strategies/shilei1/`](strategies/shilei1/): implements the "first wave >100% then deep pullback" screening idea.

## Adding a new strategy

1. Make a new folder under `strategies/` (snake-case name).
2. Copy the template structure (`data/`, `notebooks/`, `src/`, `README.md`).
3. Document the selection logic in the strategy `README.md`.
4. Keep shared helpers in `scripts/` or a future `quant` Python package to avoid cross-strategy coupling.

Happy researching!
