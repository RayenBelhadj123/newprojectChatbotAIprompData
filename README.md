# US Housing Intelligence Dashboard

An advanced Streamlit project for analyzing U.S. housing and macroeconomic data from a Kaggle-style dataset. The dashboard combines exploratory analysis, model evaluation, OLAP segmentation, forecasting, theory checks, paper comparison, and project-ready conclusions.

## Project Goal

The goal is to explain U.S. housing-price behavior using both data science and real-estate theory. The app does not only show charts; it helps answer practical project questions:

- Which variables move with home prices?
- Which machine-learning model performs best?
- Why do some models score better than others?
- Which OLAP segment is most important?
- Do the results agree with real housing-market theory?
- How do the results compare with current real-life housing conditions?

## Main Features

- **Executive Summary**: one-page project story with downloadable Markdown report.
- **Overview**: target trend, meaningful correlation matrix, and guided interpretation.
- **Explore**: line, scatter, histogram, box plot, and violin plot views.
- **Data Quality**: data-cleaning report, missing values, duplicate rows, and numeric profiling.
- **Administration Comparison**: period comparison tables, charts, and radar view.
- **ML Lab**: supervised regression/classification with multiple models.
- **Evaluation**: full model comparison table with `R2`, `MAE`, `RMSE`, `Accuracy`, `Precision`, `Recall`, `F1`, and `ROC AUC` where available.
- **Fit Diagnostics**: overfitting/underfitting detection with train-test gaps, graphs, recommendations, and direct hyperparameter tuning.
- **Model Explainability**: feature importance, coefficients, or permutation-style importance.
- **Unsupervised Lab**: KMeans, DBSCAN, PCA, and IsolationForest examples.
- **Reinforcement Lab**: educational Q-learning style market-decision example.
- **Forecast**: future target forecast using regression models and lag features.
- **Conclusion**: theory-based final conclusion and historical reality check.
- **Paper Review**: outside research and current 2026 market comparison.
- **Code Lab**: prompt box for generating Streamlit code snippets.
- **OLAP & Export**: pivot tables, interpretation, 3D OLAP cube, heatmap, and CSV downloads.
- **Data Dictionary**: column roles, types, missing values, unique counts, and examples.
- **Scenario Simulator**: multi-feature what-if model simulation using the most recent data values as defaults.
- **Production Readiness**: validation checks, drift monitoring, model card, and governance checklist.
- **Enterprise-style UI**: workflow strip, polished theme, clean metric cards, and professional tab styling.
- **Branding and advanced search**: HousingIQ logo lockup, ranked category search, dataset-column search, and beginner guide cards for ML sections.

## Screenshots To Capture

When preparing a final submission, capture these pages from the running app:

- `Overview`: target chart and important correlation matrix.
- `Evaluation`: model comparison table and feature importance.
- `OLAP & Export`: 3D OLAP cube and readable cube-face heatmap.
- `Conclusion`: final theory comparison and real-life period chart.
- `Paper Review`: outside paper/current-market comparison.
- `Executive Summary`: final report preview.

Save screenshots in `docs/assets/screenshots/` if you want to include them in documentation.

You can also open the standalone Paper Review poster:

```text
docs/paper_review_poster.html
```

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

The default dataset is expected at:

```text
data/01_raw/us_home_price_analysis_2004_2024.csv
```

You can also upload another CSV from the app sidebar.

## Install And Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or, if `make` is available:

```bash
make install
make run
```

Open the app at:

```text
http://localhost:8501
```

## Recommended Demo Flow

1. Open **Executive Summary** to understand the project story.
2. Use **Overview** to explain the target trend and meaningful correlations.
3. Use **Evaluation** to compare models and discuss why the best model wins.
4. Use **OLAP & Export** to explain the strongest segment and 3D cube.
5. Use **Paper Review** to compare the results with outside research and current market facts.
6. Use **Production Readiness** to show big-company validation, drift, and model-card practices.
7. Finish with **Conclusion** for the final project statement.

## UI Organization

The dashboard is organized like a modern analytics product:

1. Foundation: data quality and dictionary
2. Discovery: overview and exploration
3. Modeling: ML lab and evaluation
4. Segments: OLAP and unsupervised learning
5. Decisions: forecast and scenario simulator
6. Governance: report export and production readiness.

## Development Checks

```bash
python -m py_compile streamlit_app.py
pytest
```

Optional linting and documentation:

```bash
pip install ruff mkdocs mkdocs-material pymdown-extensions
ruff check . --config .code_quality/ruff.toml
mkdocs serve
```

## Code Documentation

The main app includes:

- A module-level description of the dashboard purpose.
- Docstrings for helper functions that explain inputs, outputs, and business purpose.
- Section comments that separate data preparation, UI helpers, model evaluation, OLAP, AI assistant logic, and app layout.

Use comments to explain why a block exists, not every line of what Python is already doing. This keeps the project readable for evaluators and future contributors.

## Final Project Conclusion

The project shows that U.S. housing prices are best explained with multiple evidence sources: housing theory, correlation analysis, model evaluation, OLAP segmentation, and current real-life market context. This is stronger than relying on one chart or one model because real housing markets are shaped by supply, demand, interest rates, affordability, time period, and market frictions together.
