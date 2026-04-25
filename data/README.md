# Data Layers

This project follows a layered data-science folder convention.

| Folder | Purpose |
| --- | --- |
| `01_raw` | Original immutable source files. |
| `02_intermediate` | Typed or lightly cleaned source data. |
| `03_primary` | Domain-ready tables for analysis. |
| `04_feature` | Feature tables for modeling. |
| `05_model_input` | Final model input datasets. |
| `06_models` | Serialized model artifacts. |
| `07_model_output` | Prediction and evaluation outputs. |
| `08_reporting` | Exports used by reports, dashboards, and presentations. |

The default Streamlit dataset lives at:

```text
data/01_raw/us_home_price_analysis_2004_2024.csv
```

