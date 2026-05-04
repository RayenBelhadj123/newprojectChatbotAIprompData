# DataIQ Analytics Platform

An advanced Streamlit data platform for analyzing any CSV dataset. The bundled U.S. housing and macroeconomic file remains as a demo, but the app now supports uploaded datasets, selectable targets, exploratory analysis, model evaluation, OLAP segmentation, forecasting, scenario simulation, governance checks, and downloadable reports.

## Project Goal

The goal is to turn a raw CSV into a complete analytics workspace, not a housing-only dashboard. The app helps answer practical project questions:

- Which variables move with the selected target?
- Which machine-learning model performs best?
- Why do some models score better than others?
- Which OLAP segment is most important?
- How does the model react to what-if feature changes?
- Is the dataset ready enough for reporting or production-style use?

## Main Features

- **Dataset Project Manager**: name the project, confirm source/date/target setup, inspect dataset size and health, and download a project profile.
- **Dataset Templates**: organized by category: General, Business, Education, Regulated, and Demo.
- **App Modes**: switch between Beginner, Data Scientist, Business Manager, and Presentation modes for different workflows and guidance.
- **Smart Auto-Setup**: automatically recommends the date column, target column, task type, numeric/categorical roles, and default ML feature set.
- **Data Cleaning Studio**: remove duplicates, drop empty/selected columns, repair missing numeric values with median/mean/correlation estimates, repair outliers with IQR caps, mean/median, correlation estimates, or row removal, then download the cleaned dataset.
- **Executive Summary**: one-page project story with downloadable Markdown report.
- **Report Generator**: full project report with project profile, smart setup, cleaning audit, validation, model evidence, numeric summary, data dictionary, and Markdown/HTML downloads.
- **Overview**: target trend, meaningful correlation matrix, and guided interpretation.
- **Explore**: line, scatter, histogram, box plot, and violin plot views.
- **Data Quality**: data-cleaning report, missing values, duplicate rows, and numeric profiling.
- **ML Lab**: supervised regression/classification with multiple models.
- **Evaluation**: full model comparison table with `R2`, `MAE`, `RMSE`, `Accuracy`, `Precision`, `Recall`, `F1`, and `ROC AUC` where available.
- **Prediction Page**: train a selected model, enter feature values for one prediction, batch-score the cleaned dataset, and download predictions.
- **Model Save / Load**: download trained model bundles as `.pkl`, upload trusted bundles later, and score single or batch predictions with loaded models.
- **Model Explainability**: feature importance, coefficients, or permutation-style importance.
- **Forecast**: future target forecast using regression models and lag features.
- **OLAP & Export**: pivot tables, interpretation, 3D OLAP cube, heatmap, and CSV downloads.
- **Data Dictionary**: column roles, types, missing values, unique counts, and examples.
- **Scenario Simulator**: multi-feature what-if model simulation using the most recent data values as defaults.
- **Production Readiness**: validation checks, drift monitoring, model card, and governance checklist.
- **Focused navigation**: advanced/academic pages are hidden by default and can be restored with the sidebar toggle.
- **Enterprise-style UI**: workflow strip, polished theme, clean metric cards, and professional tab styling.
- **Branding and advanced search**: DataIQ logo lockup, ranked category search, dataset-column search, and beginner guide cards for ML sections.

## Screenshots To Capture

When preparing a final submission, capture these pages from the running app:

- `Overview`: target chart and important correlation matrix.
- `Evaluation`: model comparison table and feature importance.
- `OLAP & Export`: 3D OLAP cube and readable cube-face heatmap.
- `Prediction Page`: single and batch prediction workflow.
- `Report Generator`: final Markdown/HTML report.
- `Executive Summary`: final report preview.

Save screenshots in `docs/assets/screenshots/` if you want to include them in documentation.

For the bundled housing demo, you can also open the standalone research poster:

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

Upload any CSV from the app sidebar and choose the primary target column in **Analysis Setup**. The default demo dataset is expected at:

```text
data/01_raw/us_home_price_analysis_2004_2024.csv
```

The default file is only a demo dataset for reproducible housing examples.

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

1. Upload a CSV or keep the bundled housing demo.
2. Select a **Dataset Template** and **App Mode** for the workflow you want.
3. Use **Dataset Project Manager**, **Smart Auto-Setup**, and **Data Cleaning Studio** to name the project and confirm date/target/features/cleaning.
4. Open **Executive Summary** to understand the project story.
5. Use **Overview** to explain the target trend and meaningful correlations.
6. Use **Evaluation** to compare models and discuss why the best model wins.
7. Use **Prediction Page** to score a manual example or download batch predictions.
8. Use **Model Save / Load** to download the trained model bundle.
9. Use **OLAP & Export** to explain the strongest segment and 3D cube.
10. Use **Scenario Simulator** to test what-if changes.
11. Use **Report Generator** to export the final Markdown or HTML report.
12. Use **Production Readiness** to show validation, drift, and model-card practices.

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
