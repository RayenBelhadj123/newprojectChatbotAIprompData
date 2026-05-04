# DataIQ Analytics Platform

A focused Streamlit app for one practical workflow: **upload a CSV, clean it, train a model, make predictions, and export a report**. The bundled U.S. housing file remains as a demo, but the app is designed to work with any CSV that has numeric columns.

## Project Goal

The goal is to turn a raw CSV into a usable prediction workflow:

- What file did I upload?
- Is the data clean enough?
- What target should I predict?
- What prediction does the model make?
- Can I export the result as a report?

## Main Features

- **Dataset Project Manager**: name the project, confirm source/date/target setup, inspect dataset size and health, and download a project profile.
- **Dataset Templates**: organized by category: General, Business, Education, Regulated, and Demo.
- **App Modes**: switch between Beginner, Data Scientist, Business Manager, and Presentation modes for different workflows and guidance.
- **Smart Auto-Setup**: automatically recommends the date column, target column, task type, numeric/categorical roles, and default ML feature set.
- **Data Cleaning Studio**: remove duplicates, drop empty/selected columns, repair missing numeric values with median/mean/correlation estimates, repair outliers with IQR caps, mean/median, correlation estimates, or row removal, then download the cleaned dataset.
- **Report Generator**: full project report with project profile, smart setup, cleaning audit, validation, model evidence, numeric summary, data dictionary, and Markdown/HTML downloads.
- **Overview**: target trend, meaningful correlation matrix, and guided interpretation.
- **Explore**: line, scatter, histogram, box plot, and violin plot views.
- **Data Quality**: data-cleaning report, missing values, duplicate rows, and numeric profiling.
- **Prediction Page**: train a selected model, enter feature values for one prediction, batch-score the cleaned dataset, and download predictions.
- **Focused navigation**: extra charting, academic, governance, and demo pages are hidden by default and can be restored with the sidebar toggle.
- **Enterprise-style UI**: workflow strip, polished theme, clean metric cards, and professional tab styling.
- **Branding and advanced search**: DataIQ logo lockup, ranked category search, dataset-column search, and beginner guide cards for ML sections.

## Screenshots To Capture

When preparing a final submission, capture these pages from the running app:

- `Data Quality`: cleaning audit and before/after data health.
- `Prediction Page`: single and batch prediction workflow.
- `Report Generator`: final Markdown/HTML report.

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
4. Use **Data Quality** to confirm the cleaned result.
5. Use **Prediction Page** to train a model and make predictions.
6. Use **Report Generator** to export the final Markdown or HTML report.

## UI Organization

The default dashboard is organized around one focused product flow:

1. Upload: choose a CSV and target.
2. Clean: repair missing values and outliers.
3. Train: fit a prediction model.
4. Predict: score one row or the whole dataset.
5. Export: download the final report.

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
- Section comments that separate data preparation, cleaning, prediction, reporting, and app layout.

Use comments to explain why a block exists, not every line of what Python is already doing. This keeps the project readable for evaluators and future contributors.

## Final Project Conclusion

The project is now a focused data product: upload a CSV, clean it, train a prediction model, score data, and export a report. Jupyter is still useful for research, but this app is the shareable product version that someone else can use without editing notebook code.
