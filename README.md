# US Housing Kaggle Advanced Streamlit Project

Advanced Streamlit dashboard for analyzing US housing and macroeconomic indicators from the Kaggle dataset. The app includes EDA, data-quality checks, administration comparisons, supervised machine-learning models, unsupervised learning, baseline forecasting, code generation, exports, and OLAP-style pivot tables.

## Dashboard Features

- Overview charts and target correlation summaries
- Data-quality checks and numeric profiling
- Administration comparison tables and charts
- Supervised ML lab with regression/classification model comparison
- Guided ML conclusions with actual-vs-predicted interpretation and theory checks
- Unsupervised lab with KMeans, DBSCAN, PCA, and IsolationForest
- Forecasting with multiple regression models
- Code Lab prompt for generating Streamlit snippets
- OLAP-style pivot tables and CSV exports

## Project Structure

```text
.
|-- conf/                  # Project configuration
|-- data/
|   |-- 01_raw/            # Original source data
|   |-- 02_intermediate/   # Typed or lightly cleaned data
|   |-- 03_primary/        # Domain-ready data
|   |-- 04_feature/        # Feature tables
|   |-- 05_model_input/    # Final model inputs
|   |-- 06_models/         # Model artifacts
|   |-- 07_model_output/   # Model outputs
|   `-- 08_reporting/      # Reporting exports
|-- docs/                  # MkDocs documentation
|-- models/                # Final saved models
|-- notebooks/             # Analysis notebooks by workflow stage
|-- src/us_housing/        # Reusable project code
|-- tests/                 # Pytest suite
`-- streamlit_app.py       # Streamlit dashboard entrypoint
```

## Dataset

The default CSV is expected at:

```text
data/01_raw/us_home_price_analysis_2004_2024.csv
```

You can also upload a CSV from the sidebar when the app is running.

## Install and Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or, if `make` is available:

```bash
make install
make run
```

## Development

```bash
pip install pytest ruff
python -m pytest
ruff check . --config .code_quality/ruff.toml
```

Documentation can be served with:

```bash
pip install mkdocs mkdocs-material pymdown-extensions
mkdocs serve
```
