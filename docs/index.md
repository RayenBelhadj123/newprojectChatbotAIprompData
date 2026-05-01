# US Housing Intelligence Dashboard

This documentation describes the final Streamlit project, its methodology, and how to present the results.

## 1. Project Overview

This project analyzes U.S. housing and macroeconomic indicators from 2004 to 2024 using an advanced Streamlit dashboard. The goal is to explain housing-price movement with a combination of exploratory analysis, machine learning, OLAP segmentation, forecasting, outside research, and theory-based conclusions.

The dashboard is designed as a complete data-science project rather than a simple visualization app. It includes model comparison, interpretation, real-life comparison, and downloadable outputs.

## 2. Research Questions

- How did the selected housing-price target change over time?
- Which meaningful variables have the strongest relationship with the target?
- Which supervised model performs best for prediction or classification?
- Why do some models score higher than others?
- What market segments appear strongest in OLAP analysis?
- Do the dataset results support classic real-estate theory?
- How do the project results compare with current real-life U.S. housing conditions?

## 3. Dataset

The default file is:

```text
data/01_raw/us_home_price_analysis_2004_2024.csv
```

The app also supports uploading a CSV from the sidebar. During loading, the app attempts to detect a date column, clean date values, and identify numeric variables for analysis and modeling.

## 4. Dashboard Pages

The interface is organized like an enterprise analytics product. A HousingIQ logo lockup, advanced sidebar search, and workflow strip guide the user from foundation checks to discovery, modeling, segmentation, decisions, and governance. The search panel ranks results, filters by category, supports common misspellings, and can search dataset column names. Beginner guide cards under the ML pages explain what each section does, how to use it, and what to report.

### Executive Summary

Provides a project-level summary with:

- dataset size
- target variable
- strongest meaningful signal
- best evaluated model
- downloadable Markdown report

### Overview

Shows the target trend over time and a focused correlation matrix. The correlation logic avoids spam variables such as smoothed, lagged, and target-derived columns unless the user chooses to include them.

### Explore

Includes line, scatter, histogram, box plot, and violin plot views. These charts help explain distribution, outliers, and period differences.

### Data Quality

Reports the data-cleaning process, duplicates, missing values, numeric profiles, and missing-value percentages. The cleaning report is shown even when the dataset is already clean so the project clearly documents that data preparation was checked.

### Compare

Compares selected metrics across administrations and provides visual differences between major time periods.

### ML Lab

Runs supervised models for regression and classification. Models include linear models, tree models, support vector regression, KNN, gradient boosting, random forest, and a neural network.

### Evaluation

Compares all supervised models in a clean metrics table:

- Regression: `MAE`, `RMSE`, `R2`
- Classification: `Accuracy`, `Precision`, `Recall`, `F1`, `ROC AUC` when available

The page also explains why the best model scores higher and why weaker models may score lower.

### Fit Diagnostics

Checks whether supervised models are overfitting, underfitting, or generalizing reasonably. The page compares train score, test score, train-test gap, cross-validation score, and regression errors where available. It also recommends hyperparameter fixes such as changing `max_depth`, `min_samples_leaf`, `alpha`, `C`, `n_neighbors`, `learning_rate`, or neural-network regularization.

### Model Explainability

The Evaluation page includes feature importance. Depending on the selected best model, the app uses:

- built-in tree importance
- coefficient size
- permutation-style importance

This helps answer why the model makes better predictions.

### Unsupervised Lab

Includes:

- KMeans clustering
- DBSCAN clustering
- PCA dimensionality reduction
- IsolationForest anomaly detection

The page explains what each unsupervised method teaches about the housing data.

### Reinforcement Lab

Provides an educational reinforcement-learning example. It creates market states, actions, and rewards to explain how a Q-learning style policy can learn decisions such as buy, hold, or wait.

This is not financial advice. It is an educational simulation for explaining reinforcement learning.

### Forecast

Creates baseline forecasts using lagged features and selected regression models.

### Conclusion

Builds a final project conclusion using:

- housing theory
- dataset evidence
- correlation
- real-life period comparison
- final project statement

### Paper Review

Compares the project results with outside research and current market context. The page includes:

- DiPasquale-Wheaton four-quadrant housing theory
- mortgage-rate impact research
- NAR March 2026 existing-home sales context
- Case-Shiller / FRED index context

### Code Lab

Includes an "Enter a code prompt" box to generate Streamlit code snippets. It can use an OpenAI API key or local templates.

### OLAP & Export

Includes:

- OLAP pivot table
- guided OLAP interpretation
- top segment table
- 3D OLAP cube
- readable cube-face heatmap
- CSV downloads

### Data Dictionary

Lists every column with role, type, missing values, unique values, and example or range.

### Scenario Simulator

Lets the user change multiple features and estimate the predicted impact on the selected target with a trained regression model. Each scenario input starts from the most recent available value in the filtered dataset.

### Production Readiness

Adds practices commonly used in larger companies:

- data validation checks
- duplicate and missing-data checks
- infinite-value detection
- baseline-to-current drift monitoring
- model card export
- governance checklist

This page makes the project stronger because it shows how the model would be reviewed before being trusted in a real workflow.

## 5. Methodology

The project follows this workflow:

1. Load and clean the dataset.
2. Detect date and numeric columns.
3. Explore target trends and distributions.
4. Remove misleading engineered variables from the default correlation view.
5. Compare models using train/test evaluation.
6. Explain model performance using feature importance.
7. Segment the data with OLAP and unsupervised learning.
8. Compare results with theory and real-life housing-market evidence.
9. Generate project-ready conclusions and report exports.
10. Review validation, drift, model-card, and governance checks.

## 6. Model Evaluation

The Evaluation page is the main model-comparison area. It helps explain why some models perform better:

- Random forests and gradient boosting often perform well because housing data is non-linear.
- Linear models are easier to interpret but may miss shocks or interaction effects.
- Neural networks can learn flexible patterns but may need more data.
- KNN can struggle when the recent market differs from older examples.
- Decision trees are readable but can overfit.

The best model should be selected based on test metrics, not only training performance.

## 7. OLAP Interpretation

The OLAP page helps answer segmented business questions:

- Which administration or period has the highest average target value?
- Which segment has the strongest concentration?
- Which 3D cube cell is most important?
- Does the segment result agree with ML and trend analysis?

The 3D OLAP cube is supported by a heatmap because 3D plots can be harder to read. The heatmap gives a clearer "cube face" view.

## 8. Theory And Real-Life Comparison

The project compares results with established housing-market ideas:

- affordability theory
- supply-demand theory
- income-demand theory
- labor-market theory
- mortgage-rate sensitivity
- real-estate asset and space-market theory

It also compares with current U.S. housing context, where sales can be weak while prices remain elevated because inventory is still limited.

## 9. Limitations

- Correlation does not prove causation.
- The dataset may not include every important housing variable.
- ML results depend on selected features and train/test split.
- Forecasts are educational and should not be treated as financial advice.
- The reinforcement-learning page is a simulation, not a production RL environment.
- National housing patterns can hide regional differences.

## 10. How To Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run checks:

```bash
python -m py_compile streamlit_app.py
pytest
```

## 11. Code Documentation

The application code is documented in English so the project can be reviewed, maintained, and presented clearly.

- The top of `streamlit_app.py` explains the purpose of the dashboard.
- Helper functions include concise docstrings.
- Section comments group the code into data preparation, UI helpers, model evaluation, reporting, OLAP, reinforcement-learning demo logic, and Streamlit layout.
- The comments focus on intent and project logic rather than repeating obvious Python syntax.

## 12. Final Conclusion

The dashboard supports a realistic housing-market conclusion: U.S. housing prices are shaped by multiple forces at once. Classic theory is useful, but the best explanation combines theory, model evaluation, segmentation, historical timing, and current market context.

The strongest project message is that housing prices can remain resilient even when affordability and sales are pressured, especially when supply is limited and market conditions are mixed.
