# US Housing Kaggle Advanced Streamlit Project

This project contains an advanced Streamlit dashboard for exploring US housing and macroeconomic indicators. It includes exploratory analysis, administration comparisons, supervised model comparison, unsupervised learning, baseline forecasting, code generation, exports, and OLAP-style pivot views.

## App sections

- Overview and exploratory charts
- Data quality checks
- Administration comparisons
- Supervised ML Lab for regression/classification models
- Guided ML conclusion examples with theory checks
- Unsupervised Lab for KMeans, DBSCAN, PCA, and IsolationForest
- Forecasting
- Code Lab for Streamlit snippet generation
- OLAP and CSV export tools

## Run the app

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Development checks

```bash
pip install pytest ruff
python -m pytest
ruff check . --config .code_quality/ruff.toml
```

## Documentation

```bash
pip install mkdocs mkdocs-material pymdown-extensions
mkdocs serve
```

## Dataset

Place the Kaggle CSV in the raw data layer:

```text
data/01_raw/us_home_price_analysis_2004_2024.csv
```
