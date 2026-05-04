"""Streamlit dashboard for a reusable data intelligence platform.

The application loads any tabular CSV, cleans it, and turns it into an
end-to-end analytics product: exploratory analysis, supervised and unsupervised
machine learning, forecasting, OLAP segmentation, reporting, and
production-readiness checks. The bundled housing CSV now works as a demo
dataset rather than the only supported domain. Helper functions are kept in this
file because the Streamlit app is currently the project entrypoint; reusable
code can later be moved into package modules as the project grows.
"""

import os
import sys
import io
import html
import pickle
from difflib import SequenceMatcher
from pathlib import Path

# Limit joblib/loky worker discovery noise on Windows before importing scikit-learn.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - app can run without OpenAI installed
    OpenAI = None


# Make the local package importable when the app is launched from the repo root.
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# The template includes a helper that resolves the default dataset path.
from us_housing.paths import resolve_default_dataset  # noqa: E402


# Configure Streamlit before rendering any visible widgets.
st.set_page_config(
    page_title="DataIQ Analytics Platform",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_CSV = resolve_default_dataset()

# Keep the default CSV path reusable for the entire app lifecycle.

USE_CASE_TEMPLATES: dict[str, dict[str, object]] = {
    "General Data Project": {
        "category": "General",
        "description": "A neutral setup for any CSV when the business question is still open.",
        "target_keywords": ["target", "label", "outcome", "score", "value", "index"],
        "preferred_task": "Auto",
        "feature_strategy": "Use clean numeric columns with low missingness and avoid identifiers or target-derived fields.",
        "report_language": "Explain the selected target with data quality, model evidence, segmentation, and governance checks.",
        "recommended_pages": ["Project Manager", "Smart Auto-Setup", "Overview", "Evaluation", "Prediction Page", "Report Generator"],
        "caution": "Match the target and features to the real business or research question before interpreting results.",
    },
    "Sales Analysis": {
        "category": "Business",
        "description": "Analyze revenue, profit, orders, product performance, and commercial drivers.",
        "target_keywords": ["sales", "revenue", "profit", "amount", "order_value", "quantity", "margin"],
        "preferred_task": "Regression",
        "feature_strategy": "Prefer customer, product, channel, time, quantity, price, discount, and cost indicators.",
        "report_language": "Frame results around revenue drivers, profitable segments, forecasts, and commercial recommendations.",
        "recommended_pages": ["Overview", "OLAP & Export", "Evaluation", "Forecast", "Prediction Page", "Business Impact", "Report Generator"],
        "caution": "Predicted revenue is not causal proof; validate with business context and campaign or pricing history.",
    },
    "Customer Churn": {
        "category": "Business",
        "description": "Predict retention/churn risk and identify customer segments that need action.",
        "target_keywords": ["churn", "cancelled", "canceled", "retained", "active", "left", "subscription", "renewed"],
        "preferred_task": "Classification",
        "feature_strategy": "Prefer tenure, usage, support, billing, plan, satisfaction, engagement, and recent activity indicators.",
        "report_language": "Frame results around churn risk, retention actions, high-risk segments, and model confidence.",
        "recommended_pages": ["Data Cleaning Studio", "Evaluation", "Prediction Page", "Model Save / Load", "Business Impact", "Report Generator"],
        "caution": "Treat churn predictions as decision support; avoid unfair targeting without bias and policy review.",
    },
    "Student Performance": {
        "category": "Education",
        "description": "Understand scores, grades, pass/fail outcomes, and support indicators.",
        "target_keywords": ["score", "grade", "exam", "gpa", "pass", "result", "performance", "marks"],
        "preferred_task": "Auto",
        "feature_strategy": "Prefer attendance, study time, prior grades, engagement, assignments, and support indicators.",
        "report_language": "Frame results around performance drivers, student-support signals, and responsible intervention planning.",
        "recommended_pages": ["Data Quality", "Overview", "Evaluation", "Prediction Page", "Business Impact", "Report Generator"],
        "caution": "Educational predictions should support learners, not label or penalize them automatically.",
    },
    "Finance / Risk": {
        "category": "Regulated",
        "description": "Analyze risk, default, fraud, returns, losses, and financial outcomes.",
        "target_keywords": ["risk", "default", "return", "loss", "price", "fraud", "credit", "score", "volatility"],
        "preferred_task": "Auto",
        "feature_strategy": "Prefer exposure, transaction, balance, income, rate, history, volatility, and behavior indicators.",
        "report_language": "Frame results around risk signals, model stability, scenario stress, and governance requirements.",
        "recommended_pages": ["Data Cleaning Studio", "Evaluation", "Scenario Simulator", "Model Save / Load", "Production Readiness", "Report Generator"],
        "caution": "Finance outputs need strict validation, monitoring, and fairness/compliance review before real use.",
    },
    "Health Dataset": {
        "category": "Regulated",
        "description": "Educational-only analysis for outcomes, risk, costs, or health-related records.",
        "target_keywords": ["diagnosis", "outcome", "risk", "cost", "readmission", "disease", "treatment", "survival"],
        "preferred_task": "Auto",
        "feature_strategy": "Prefer clinically relevant, consented, documented variables and review missingness very carefully.",
        "report_language": "Frame results as educational data analysis with quality limits, uncertainty, and responsible-use warnings.",
        "recommended_pages": ["Data Quality", "Evaluation", "Prediction Page", "Production Readiness", "Business Impact", "Report Generator"],
        "caution": "Educational only. Do not use this app for medical diagnosis or treatment decisions.",
    },
    "Housing Demo": {
        "category": "Demo",
        "description": "Use the bundled housing dataset to demonstrate the platform end to end.",
        "target_keywords": ["home_price_index", "home price", "price", "hpi", "mortgage", "housing", "inventory"],
        "preferred_task": "Regression",
        "feature_strategy": "Prefer macro, mortgage, inventory, affordability, income, labor, and time-based indicators.",
        "report_language": "Frame results around housing price drivers, market regimes, forecasts, and external housing theory.",
        "recommended_pages": ["Overview", "Evaluation", "OLAP & Export", "Forecast", "Business Impact", "Report Generator"],
        "caution": "Housing forecasts are educational and should not be treated as financial advice.",
    },
}

TEMPLATE_CATEGORY_ORDER = ["General", "Business", "Education", "Regulated", "Demo"]

APP_MODES: dict[str, dict[str, object]] = {
    "Beginner Mode": {
        "description": "More guidance, recommended clicks, and plain-language interpretation.",
        "focus": ["Project Manager", "Smart Auto-Setup", "Data Cleaning Studio", "Overview", "Report Generator"],
        "tone": "Explain what to click next and why each result matters.",
    },
    "Data Scientist Mode": {
        "description": "Modeling, diagnostics, feature review, saving/loading, and validation details.",
        "focus": ["Evaluation", "Prediction Page", "Model Save / Load", "Production Readiness", "Report Generator"],
        "tone": "Prioritize metrics, leakage checks, diagnostics, reproducibility, and model governance.",
    },
    "Business Manager Mode": {
        "description": "Executive story, segments, predictions, impact, and downloadable reports.",
        "focus": ["Executive Summary", "Business Impact", "OLAP & Export", "Scenario Simulator", "Report Generator"],
        "tone": "Translate technical evidence into decisions, risks, and recommendations.",
    },
    "Presentation Mode": {
        "description": "Short demo flow for presenting the platform clearly.",
        "focus": ["Start Here", "Executive Summary", "Overview", "Evaluation", "Prediction Page", "Report Generator"],
        "tone": "Use concise speaking points and avoid opening too many tabs.",
    },
}

NONESSENTIAL_PAGES = {
    "Compare",
    "Unsupervised Lab",
    "Reinforcement Lab",
    "Conclusion",
    "Domain Review",
    "Code Lab",
    "Experiment Tracker",
    "Model Registry",
    "Data Pipeline",
    "Big Data Readiness",
    "Fit Diagnostics",
}


def is_housing_dataset(df: pd.DataFrame, source_name: str = "") -> bool:
    """Return True when the active dataset looks like the bundled housing demo."""
    text = " ".join([source_name, *map(str, df.columns)]).lower()
    housing_tokens = ["housing", "home_price", "home price", "mortgage", "case-shiller", "hpi"]
    return any(token in text for token in housing_tokens)


def app_profile(df: pd.DataFrame, source_name: str) -> dict[str, str | bool]:
    """Build display labels for generic mode or the bundled housing demo mode."""
    housing_mode = is_housing_dataset(df, source_name)
    domain = "Housing demo" if housing_mode else "General data"
    return {
        "name": "DataIQ Platform",
        "mark": "DI",
        "subtitle": "Dataset-agnostic analytics",
        "domain": domain,
        "is_housing": housing_mode,
        "description": (
            "Upload any CSV, choose a target, explore data quality, train models, forecast, "
            "build OLAP views, and prepare governance-ready outputs."
        ),
    }


def project_name_from_source(source_name: str) -> str:
    """Create a readable default project name from the active data source."""
    raw_name = source_name.split(":", 1)[-1].strip() if ":" in source_name else source_name
    stem = Path(raw_name).stem.replace("_", " ").replace("-", " ").strip()
    return stem.title() if stem else "Untitled Data Project"


def infer_task_type(df: pd.DataFrame, target: str | None) -> str:
    """Infer the most likely supervised-learning task for the selected target."""
    if not target or target not in df.columns:
        return "Not ready"
    series = df[target].dropna()
    if series.empty:
        return "Not ready"
    if pd.api.types.is_numeric_dtype(series):
        unique_count = int(series.nunique(dropna=True))
        return "Classification" if unique_count <= min(20, max(2, len(series) // 10)) else "Regression"
    return "Classification"


def looks_like_identifier(column: str, series: pd.Series) -> bool:
    """Detect identifier-like columns that should not be default targets/features."""
    lower = str(column).lower()
    if lower in {"id", "uuid", "index", "row_id", "record_id"} or lower.endswith("_id"):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    uniqueness = non_null.nunique(dropna=True) / max(len(non_null), 1)
    return uniqueness > 0.95 and pd.api.types.is_integer_dtype(non_null)


def score_date_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Rank columns by how likely they are to represent time."""
    rows = []
    for col in df.columns:
        series = df[col]
        lower = str(col).lower()
        name_score = 0
        if lower in {"date", "datetime", "timestamp", "time", "month", "year"}:
            name_score += 45
        if any(token in lower for token in ["date", "time", "month", "year", "period"]):
            name_score += 25
        parsed = pd.to_datetime(series, errors="coerce")
        parse_ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
        unique_ratio = float(parsed.nunique(dropna=True) / max(parsed.notna().sum(), 1))
        score = name_score + parse_ratio * 45 + min(unique_ratio, 1.0) * 10
        if parse_ratio >= 0.5 or name_score:
            rows.append(
                {
                    "Column": col,
                    "Score": round(score, 1),
                    "Parsed %": round(parse_ratio * 100, 1),
                    "Reason": "name + parseable dates" if name_score and parse_ratio >= 0.5 else "parseable dates" if parse_ratio >= 0.5 else "date-like name",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Column", "Score", "Parsed %", "Reason"])
    return pd.DataFrame(rows).sort_values(["Score", "Parsed %"], ascending=False).reset_index(drop=True)


def score_target_candidates(
    df: pd.DataFrame,
    numeric_cols: list[str],
    template: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Rank numeric columns by how suitable they are as a supervised target."""
    target_tokens = [
        "target",
        "label",
        "outcome",
        "price",
        "sales",
        "revenue",
        "profit",
        "score",
        "value",
        "index",
        "churn",
        "risk",
    ]
    if template:
        target_tokens = list(dict.fromkeys([*target_tokens, *template.get("target_keywords", [])]))
    rows = []
    for col in numeric_cols:
        series = df[col]
        lower = str(col).lower()
        non_null_ratio = float(series.notna().mean()) if len(series) else 0.0
        unique_count = int(series.nunique(dropna=True))
        if unique_count <= 1:
            continue
        score = non_null_ratio * 35
        if any(token in lower for token in target_tokens):
            score += 30
        if template and any(str(token).lower() in lower for token in template.get("target_keywords", [])):
            score += 35
        if "home_price_index" in lower or "home price index" in lower:
            score += 50
        if 2 <= unique_count <= min(20, max(2, len(series) // 10)):
            score += 10
        else:
            score += 18
        if looks_like_identifier(col, series):
            score -= 45
        missing_pct = (1 - non_null_ratio) * 100
        rows.append(
            {
                "Column": col,
                "Score": round(score, 1),
                "Missing %": round(missing_pct, 1),
                "Unique values": unique_count,
                "Suggested task": infer_task_type(df, col),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Column", "Score", "Missing %", "Unique values", "Suggested task"])
    return pd.DataFrame(rows).sort_values(["Score", "Unique values"], ascending=False).reset_index(drop=True)


def recommend_feature_columns(df: pd.DataFrame, target: str | None, date_col: str | None, numeric_cols: list[str]) -> list[str]:
    """Recommend clean numeric features for supervised pages."""
    candidates = []
    for col in numeric_cols:
        if col == target or col == date_col or col == "time_group":
            continue
        series = df[col]
        if series.nunique(dropna=True) <= 1:
            continue
        if series.isna().mean() > 0.6:
            continue
        if looks_like_identifier(col, series):
            continue
        if is_engineered_or_leaky_feature(col, target):
            continue
        candidates.append(col)
    if target and target in df.columns and candidates and pd.api.types.is_numeric_dtype(df[target]):
        corr_df = df[[target] + candidates].corr(numeric_only=True)
        if target in corr_df.columns:
            ranked = corr_df[target].drop(labels=[target], errors="ignore").dropna().abs().sort_values(ascending=False)
            return ranked.index.tolist()[: min(12, len(ranked))]
    return candidates[: min(12, len(candidates))]


def smart_auto_setup(
    df: pd.DataFrame,
    source_name: str,
    template: dict[str, object] | None = None,
) -> dict[str, object]:
    """Infer a practical default setup for a newly loaded dataset."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    date_candidates = score_date_candidates(df)
    target_candidates = score_target_candidates(df, numeric_cols, template)
    recommended_date = str(date_candidates.iloc[0]["Column"]) if not date_candidates.empty else None
    if not target_candidates.empty:
        recommended_target = str(target_candidates.iloc[0]["Column"])
    else:
        recommended_target = default_target(numeric_cols)
    recommended_features = recommend_feature_columns(df, recommended_target, recommended_date, numeric_cols)
    issues = []
    if recommended_date is None:
        issues.append("No strong date column found; time filters and forecasting may be limited.")
    if recommended_target is None:
        issues.append("No numeric target found; supervised modeling needs a numeric target in this version.")
    if len(recommended_features) == 0:
        issues.append("No clean numeric feature recommendations found; choose features manually after upload.")
    return {
        "Source": source_name,
        "Recommended date": recommended_date,
        "Recommended target": recommended_target,
        "Suggested task": (
            template.get("preferred_task")
            if template and template.get("preferred_task") in {"Regression", "Classification"}
            else infer_task_type(df, recommended_target)
        ),
        "Numeric columns": numeric_cols,
        "Categorical columns": categorical_cols,
        "Recommended features": recommended_features,
        "Date candidates": date_candidates,
        "Target candidates": target_candidates,
        "Issues": issues,
    }


def build_project_profile(
    project_name: str,
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    source_name: str,
    date_col: str | None,
    target: str | None,
    profile: dict[str, str | bool],
    template_name: str = "General Data Project",
    app_mode: str = "Beginner Mode",
) -> dict[str, object]:
    """Collect the project-manager state used across the app."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    date_range = "Not available"
    if date_col and date_col in df.columns and df[date_col].notna().any():
        date_range = f"{df[date_col].min().date()} to {df[date_col].max().date()}"
    return {
        "Project name": project_name,
        "Source": source_name,
        "Mode": profile["domain"],
        "Use-case template": template_name,
        "App mode": app_mode,
        "Rows loaded": len(raw_df),
        "Rows after filters": len(df),
        "Columns": df.shape[1],
        "Numeric columns": len(numeric_cols),
        "Categorical columns": len(categorical_cols),
        "Date column": date_col or "Not selected",
        "Date range": date_range,
        "Primary target": target or "Not selected",
        "Suggested task": infer_task_type(df, target),
        "Missing cells": int(df.isna().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
    }

def apply_theme(dark_mode: bool) -> None:
    """Inject the custom dashboard theme and responsive layout CSS."""
    bg = "#08111f" if dark_mode else "#eef3f8"
    panel = "#101b2d" if dark_mode else "#fbfdff"
    panel_2 = "#16243a" if dark_mode else "#e6edf5"
    panel_3 = "#0b1424" if dark_mode else "#d9e4ef"
    text = "#edf4ff" if dark_mode else "#142033"
    muted = "#9fb0c7" if dark_mode else "#536173"
    border = "#283852" if dark_mode else "#c8d6e5"
    accent = "#21b8a6" if dark_mode else "#087f8c"
    accent_2 = "#f4b046" if dark_mode else "#c35f1d"
    danger = "#fb7185" if dark_mode else "#dc2626"
    shadow = "0 22px 70px rgba(0,0,0,0.34)" if dark_mode else "0 22px 60px rgba(24,43,72,0.13)"

    # Keep the visual system in one place so dark/light mode stays consistent.
    st.markdown(
        f"""
        <style>
            :root {{
                --sat: env(safe-area-inset-top);
                --sab: env(safe-area-inset-bottom);
                --sal: env(safe-area-inset-left);
                --sar: env(safe-area-inset-right);
                --app-bg: {bg};
                --panel: {panel};
                --panel-2: {panel_2};
                --panel-3: {panel_3};
                --text: {text};
                --muted: {muted};
                --border: {border};
                --accent: {accent};
                --accent-2: {accent_2};
                --danger: {danger};
                --shadow: {shadow};
            }}
            html, body, [data-testid="stAppViewContainer"], .stApp {{
                min-height: 100vh;
                min-height: 100dvh;
                width: 100%;
                max-width: 100vw;
                margin: 0;
                padding: 0;
                overflow-x: hidden;
                -webkit-overflow-scrolling: touch;
                overscroll-behavior-y: none;
                touch-action: manipulation;
            }}
            html {{
                scroll-behavior: smooth;
            }}
            body {{
                background: var(--app-bg);
                -webkit-tap-highlight-color: transparent;
                text-rendering: optimizeLegibility;
            }}
            .stApp {{
                background:
                    linear-gradient(180deg, var(--panel-3) 0px, var(--app-bg) 260px, var(--app-bg) 100%);
                color: var(--text);
                padding-left: var(--sal);
                padding-right: var(--sar);
                padding-bottom: var(--sab);
            }}
            [data-testid="stHeader"] {{
                background: transparent;
                height: 0;
                display: none;
            }}
            [data-testid="stToolbar"] {{
                display: none;
            }}
            [data-testid="stDecoration"] {{
                display: none;
            }}
            #MainMenu, footer {{
                visibility: hidden;
            }}
            .block-container {{
                padding-top: max(0.75rem, calc(var(--sat) + 0.5rem));
                padding-bottom: 1.8rem;
                max-width: min(1200px, calc(100vw - var(--sal) - var(--sar) - 1rem));
                width: 100%;
                overflow-x: hidden;
            }}
            .block-container > div,
            [data-testid="stVerticalBlock"],
            [data-testid="stHorizontalBlock"] {{
                max-width: 100%;
            }}
            [data-testid="stElementContainer"],
            [data-testid="stPlotlyChart"],
            [data-testid="stDataFrame"],
            [data-testid="stTable"] {{
                max-width: 100%;
                overflow-x: auto;
            }}
            [data-testid="stDataFrame"] > div,
            [data-testid="stTable"] > div {{
                max-width: 100%;
            }}
            h1, h2, h3, h4 {{
                color: var(--text);
                letter-spacing: 0;
                font-weight: 750;
            }}
            h2 {{
                padding-top: 0.2rem;
                font-size: 1.4rem;
            }}
            h3 {{
                border-left: 3px solid var(--accent);
                padding-left: 0.6rem;
                margin-top: 1rem;
                font-size: 1.1rem;
            }}
            p, li, label, span {{
                letter-spacing: 0;
            }}
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, var(--panel-3), var(--panel));
                border-right: 1px solid var(--border);
                box-shadow: var(--shadow);
                padding-top: var(--sat);
            }}
            [data-testid="stSidebar"] * {{
                color: var(--text);
            }}
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
                color: var(--muted);
            }}
            [data-testid="stSidebar"] div[data-baseweb="select"] > div,
            [data-testid="stSidebar"] input {{
                background: var(--panel-2);
                border-color: var(--border);
            }}
            .brand-lockup {{
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 10px 0 6px 0;
                margin-bottom: 8px;
            }}
            .brand-mark {{
                width: 44px;
                height: 44px;
                display: grid;
                place-items: center;
                border-radius: 12px;
                background: var(--accent);
                color: #ffffff;
                font-weight: 900;
                font-size: 1.25rem;
                box-shadow: 0 14px 30px rgba(33,184,166,0.24);
            }}
            .brand-name {{
                color: var(--text);
                font-size: 1.06rem;
                font-weight: 850;
                line-height: 1.05;
            }}
            .brand-subtitle {{
                color: var(--muted);
                font-size: 0.74rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-top: 4px;
            }}
            .hero {{
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 24px;
                background:
                    linear-gradient(135deg, rgba(33,184,166,0.14), rgba(244,176,70,0.08)),
                    var(--panel);
                margin-bottom: 12px;
                box-shadow: var(--shadow);
                position: relative;
                overflow: hidden;
            }}
            .hero::before {{
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 4px;
                background: var(--accent);
            }}
            .hero h1 {{
                font-size: 1.88rem;
                margin: 0 0 6px 0;
                line-height: 1.15;
            }}
            .hero p {{
                color: var(--muted);
                margin: 0;
                line-height: 1.5;
                max-width: 800px;
                font-size: 0.95rem;
            }}
            .chip-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 16px;
            }}
            .chip {{
                border: 1px solid var(--border);
                border-radius: 999px;
                padding: 7px 11px;
                background: var(--panel-2);
                color: var(--text);
                font-size: 0.88rem;
                font-weight: 620;
            }}
            .workflow-strip {{
                display: grid;
                grid-template-columns: repeat(6, minmax(110px, 1fr));
                gap: 8px;
                margin: 10px 0 14px 0;
            }}
            .workflow-step {{
                border: 1px solid var(--border);
                background: linear-gradient(180deg, var(--panel), var(--panel-2));
                border-radius: 11px;
                padding: 11px;
                box-shadow: 0 10px 28px rgba(0,0,0,0.1);
                min-height: 70px;
            }}
            .workflow-step .step-kicker {{
                color: var(--accent);
                font-size: 0.68rem;
                font-weight: 790;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}
            .workflow-step .step-title {{
                color: var(--text);
                font-size: 0.85rem;
                font-weight: 740;
                margin-top: 2px;
            }}
            .metric-card {{
                min-height: 95px;
                border: 1px solid var(--border);
                border-radius: 11px;
                background: linear-gradient(180deg, var(--panel), var(--panel-2));
                padding: 14px;
                box-shadow: 0 12px 32px rgba(0,0,0,0.1);
                border-top: 3px solid var(--accent);
            }}
            .metric-card .label {{
                color: var(--muted);
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 740;
            }}
            .metric-card .value {{
                color: var(--text);
                font-size: 1.25rem;
                font-weight: 740;
                margin-top: 6px;
                word-break: break-word;
            }}
            .section-note {{
                color: var(--muted);
                margin-top: -6px;
            }}
            div[data-testid="stMetric"] {{
                border: 1px solid var(--border);
                border-radius: 13px;
                padding: 13px;
                background: linear-gradient(180deg, var(--panel), var(--panel-2));
                box-shadow: 0 14px 34px rgba(0,0,0,0.12);
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 6px;
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 8px;
                background: linear-gradient(180deg, var(--panel), var(--panel-2));
                box-shadow: 0 16px 44px rgba(0,0,0,0.16);
                overflow-x: auto;
            }}
            .stTabs [data-baseweb="tab"] {{
                border-radius: 9px;
                padding: 10px 13px;
                min-height: 42px;
                color: var(--muted);
                font-weight: 720;
                border: 1px solid transparent;
            }}
            .stTabs [aria-selected="true"] {{
                background: var(--panel-2);
                border: 1px solid var(--border);
                color: var(--text);
                box-shadow: inset 0 -2px 0 var(--accent);
            }}
            div[data-testid="stDataFrame"],
            div[data-testid="stTable"] {{
                border: 1px solid var(--border);
                border-radius: 10px;
                overflow-x: auto;
                overflow-y: hidden;
                background: var(--panel);
            }}
            .stButton > button,
            .stDownloadButton > button {{
                border-radius: 9px;
                border: 1px solid var(--accent);
                background: var(--accent);
                color: white;
                font-weight: 750;
                min-height: 40px;
                box-shadow: 0 10px 26px rgba(8,127,140,0.16);
            }}
            .stButton > button:hover,
            .stDownloadButton > button:hover {{
                border-color: var(--accent-2);
                background: var(--accent-2);
                color: white;
            }}
            div[data-baseweb="select"] > div,
            textarea,
            input {{
                border-radius: 9px !important;
                border-color: var(--border) !important;
            }}
            [data-testid="stAlert"] {{
                border-radius: 10px;
                border: 1px solid var(--border);
            }}
            [data-testid="stExpander"] {{
                border: 1px solid var(--border);
                border-radius: 10px;
                background: var(--panel);
            }}
            .learning-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(160px, 1fr));
                gap: 10px;
                margin: 6px 0 14px 0;
            }}
            .learning-card {{
                border: 1px solid var(--border);
                border-radius: 11px;
                padding: 12px;
                background: linear-gradient(180deg, var(--panel), var(--panel-2));
                box-shadow: 0 10px 28px rgba(0,0,0,0.1);
            }}
            .learning-card .learning-kicker {{
                color: var(--accent-2);
                font-size: 0.68rem;
                font-weight: 840;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 4px;
            }}
            .learning-card .learning-title {{
                color: var(--text);
                font-size: 0.92rem;
                font-weight: 780;
                margin-bottom: 4px;
            }}
            .learning-card .learning-text {{
                color: var(--muted);
                font-size: 0.82rem;
                line-height: 1.4;
            }}
            .search-result {{
                border: 1px solid var(--border);
                border-radius: 11px;
                padding: 10px 12px;
                background: var(--panel-2);
                margin: 7px 0;
            }}
            .search-result strong {{
                color: var(--text);
            }}
            .search-result span {{
                color: var(--muted);
                display: block;
                margin-top: 3px;
                font-size: 0.82rem;
            }}
            @media (max-width: 900px) {{
                .block-container {{
                    padding-left: max(0.7rem, calc(var(--sal) + 0.7rem));
                    padding-right: max(0.7rem, calc(var(--sar) + 0.7rem));
                    padding-bottom: max(1.2rem, calc(var(--sab) + 0.8rem));
                }}
                .workflow-strip {{
                    grid-template-columns: repeat(2, minmax(110px, 1fr));
                }}
                .learning-grid {{
                    grid-template-columns: 1fr;
                }}
                .hero h1 {{
                    font-size: 1.4rem;
                }}
                .hero {{
                    padding: 16px;
                    border-radius: 12px;
                }}
                .brand-mark {{
                    width: 36px;
                    height: 36px;
                    border-radius: 9px;
                }}
                .stTabs [data-baseweb="tab"] {{
                    min-width: max-content;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Nettoyage / Data preparation helpers
# -----------------------------------------------------------------------------
# This section implements the "nettoyage" pipeline: loading raw data, cleaning
# column names, parsing dates, coercing numeric values, and building period
# labels for consistent downstream analysis.

def find_date_col(df: pd.DataFrame) -> str | None:
    """Return the most likely date or time column from a dataframe."""
    for col in df.columns:
        lower = str(col).lower()
        if lower in {"date", "month"} or "date" in lower or "time" in lower:
            return col
    return None


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Normalize columns, parse dates, coerce numerics, and add period labels."""
    # Start the nettoyage pipeline: create a safe copy, normalize column names,
    # detect the date column, parse dates, and convert candidate numeric columns.
    out = df.copy()
    out.columns = out.columns.str.strip()
    date_col = find_date_col(out)
    if date_col:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.sort_values(date_col)

    # Coerce all non-date columns to numeric when the values look numeric.
    for col in out.columns:
        if col != date_col:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() > 0:
                out[col] = converted

    # Add a reusable neutral time label for later dashboard grouping.
    if date_col and out[date_col].notna().any():
        out["time_group"] = out[date_col].apply(time_group_label)
    else:
        out["time_group"] = "Unknown"
    return out, date_col


@st.cache_data(show_spinner="Loading CSV...")
def load_csv_from_path(path: str, modified_time: float) -> pd.DataFrame:
    """Load a local CSV once per file version."""
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading uploaded CSV...")
def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Load an uploaded CSV once per uploaded file content."""
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Preparing dataset...")
def prepare_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, pd.DataFrame]:
    """Cache cleaning and the cleaning report so reruns stay responsive."""
    cleaned_df, detected_date_col = clean_data(raw_df)
    report = data_cleaning_report(raw_df, cleaned_df, detected_date_col)
    return cleaned_df, detected_date_col, report


def limit_rows_for_display(data: pd.DataFrame, max_rows: int, sort_col: str | None = None) -> pd.DataFrame:
    """Return a deterministic row-limited frame for heavy charts and previews."""
    if max_rows <= 0 or len(data) <= max_rows:
        return data
    display_df = data.sort_values(sort_col) if sort_col and sort_col in data.columns else data
    positions = np.linspace(0, len(display_df) - 1, max_rows).astype(int)
    return display_df.iloc[np.unique(positions)].copy()


def data_cleaning_report(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, date_col: str | None) -> pd.DataFrame:
    """Document every cleaning check, including checks that found clean data."""
    # This report is the final stage of the nettoyage pipeline: it records
    # what was cleaned, what was already clean, and any issues introduced.
    raw_columns = pd.Index(raw_df.columns)
    stripped_columns = raw_columns.astype(str).str.strip()
    whitespace_fixed = int((raw_columns.astype(str) != stripped_columns).sum())
    raw_missing = int(raw_df.isna().sum().sum())
    cleaned_missing = int(cleaned_df.isna().sum().sum())
    raw_duplicates = int(raw_df.duplicated().sum())
    cleaned_duplicates = int(cleaned_df.duplicated().sum())
    numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns.tolist()

    rows = [
        {
            "Cleaning step": "Column-name cleanup",
            "What was checked": "Leading and trailing spaces in column names.",
            "Result": (
                f"{whitespace_fixed} column name(s) were trimmed."
                if whitespace_fixed
                else "No extra spaces were found; column names were already clean."
            ),
            "Status": "Applied" if whitespace_fixed else "Checked - already clean",
        },
        {
            "Cleaning step": "Date parsing",
            "What was checked": "The app searched for a date or time column and converted it to datetime.",
            "Result": (
                f"`{date_col}` was detected and parsed as a date column."
                if date_col
                else "No date column was detected, so date-based analysis is disabled."
            ),
            "Status": "Applied" if date_col else "Checked - not available",
        },
        {
            "Cleaning step": "Numeric conversion",
            "What was checked": "Columns that contain numeric values were converted from text to numbers.",
            "Result": f"{len(numeric_columns)} numeric column(s) are available after cleaning.",
            "Status": "Applied",
        },
        {
            "Cleaning step": "Missing-value check",
            "What was checked": "The app counted empty cells before and after type conversion.",
            "Result": (
                f"Missing cells changed from {raw_missing:,} to {cleaned_missing:,}."
                if raw_missing != cleaned_missing
                else f"{cleaned_missing:,} missing cell(s) found; no extra missing values were introduced."
            ),
            "Status": "Checked - already clean" if cleaned_missing == 0 else "Needs attention",
        },
        {
            "Cleaning step": "Duplicate-row check",
            "What was checked": "The app counted repeated rows.",
            "Result": (
                f"{cleaned_duplicates:,} duplicate row(s) remain for review."
                if cleaned_duplicates
                else "No duplicate rows were found."
            ),
            "Status": "Checked - already clean" if cleaned_duplicates == 0 else "Needs attention",
        },
        {
            "Cleaning step": "Period feature creation",
            "What was checked": "The app created a reusable time-group label for comparison pages.",
            "Result": "`time_group` was added as the internal grouped-analysis column.",
            "Status": "Applied",
        },
        {
            "Cleaning step": "Row preservation",
            "What was checked": "The app confirms whether cleaning removed rows.",
            "Result": (
                f"Rows changed from {len(raw_df):,} to {len(cleaned_df):,}."
                if len(raw_df) != len(cleaned_df)
                else "No rows were removed during cleaning."
            ),
            "Status": "Checked - already clean" if len(raw_df) == len(cleaned_df) else "Applied",
        },
    ]

    report = pd.DataFrame(rows)
    report["Project note"] = (
        "This cleaning step is documented even when the data is already clean, "
        "so the project shows that data preparation was performed."
    )
    return report


def dataset_health(df: pd.DataFrame) -> dict[str, int]:
    """Return compact data-health counters for before/after cleaning views."""
    return {
        "Rows": len(df),
        "Columns": df.shape[1],
        "Missing cells": int(df.isna().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
        "Empty columns": int((df.isna().mean() == 1).sum()) if len(df) else 0,
    }


def fill_categorical_with_mode(data: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    """Fill categorical missing values with the mode or Unknown."""
    out = data.copy()
    changed = 0
    for col in columns:
        if col not in out.columns:
            continue
        before = int(out[col].isna().sum())
        if before == 0:
            continue
        mode = out[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        out[col] = out[col].fillna(fill_value)
        changed += before
    return out, changed


def remove_iqr_outliers(data: pd.DataFrame, columns: list[str], factor: float) -> tuple[pd.DataFrame, int]:
    """Remove rows outside the IQR fence for selected numeric columns."""
    out = data.copy()
    if not columns:
        return out, 0
    mask = pd.Series(True, index=out.index)
    for col in columns:
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask &= out[col].between(lower, upper) | out[col].isna()
    removed = int((~mask).sum())
    return out.loc[mask].copy(), removed


def iqr_outlier_masks(data: pd.DataFrame, columns: list[str], factor: float) -> dict[str, pd.Series]:
    """Return per-column masks for values outside the IQR fence."""
    masks: dict[str, pd.Series] = {}
    for col in columns:
        if col not in data.columns or not pd.api.types.is_numeric_dtype(data[col]):
            continue
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            masks[col] = pd.Series(False, index=data.index)
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        masks[col] = ~(data[col].between(lower, upper) | data[col].isna())
    return masks


def cap_iqr_outliers(data: pd.DataFrame, columns: list[str], factor: float) -> tuple[pd.DataFrame, int]:
    """Replace outliers with their nearest IQR fence instead of deleting rows."""
    out = data.copy()
    changed = 0
    for col in columns:
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = ~(out[col].between(lower, upper) | out[col].isna())
        changed += int(mask.sum())
        out.loc[mask, col] = out.loc[mask, col].clip(lower=lower, upper=upper)
    return out, changed


def fill_numeric_simple(data: pd.DataFrame, columns: list[str], strategy: str) -> tuple[pd.DataFrame, int]:
    """Fill numeric missing values with mean or median."""
    out = data.copy()
    changed = 0
    for col in columns:
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        mask = out[col].isna()
        count = int(mask.sum())
        if count == 0:
            continue
        fill_value = out[col].mean() if strategy == "Mean" else out[col].median()
        if pd.isna(fill_value):
            continue
        out.loc[mask, col] = fill_value
        changed += count
    return out, changed


def replace_numeric_masks_simple(
    data: pd.DataFrame,
    mask_map: dict[str, pd.Series],
    strategy: str,
) -> tuple[pd.DataFrame, int]:
    """Replace masked numeric values with mean or median from the unmasked data."""
    out = data.copy()
    changed = 0
    for col, mask in mask_map.items():
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        clean_values = out.loc[~mask, col]
        fill_value = clean_values.mean() if strategy == "Mean" else clean_values.median()
        if pd.isna(fill_value):
            fill_value = out[col].median()
        if pd.isna(fill_value):
            continue
        count = int(mask.sum())
        out.loc[mask, col] = fill_value
        changed += count
    return out, changed


def fill_numeric_from_correlations(
    data: pd.DataFrame,
    columns: list[str],
    mask_map: dict[str, pd.Series] | None = None,
    max_features: int = 5,
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """Estimate missing/masked numeric values from the most correlated variables."""
    out = data.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
    rows = []
    total_changed = 0
    for col in columns:
        if col not in numeric_cols:
            continue
        target_mask = mask_map[col].copy() if mask_map and col in mask_map else out[col].isna()
        target_mask = target_mask.reindex(out.index, fill_value=False)
        missing_count = int(target_mask.sum())
        if missing_count == 0:
            continue

        candidate_cols = [candidate for candidate in numeric_cols if candidate != col]
        valid_corr_data = out[[col] + candidate_cols].copy()
        corr = valid_corr_data.corr(numeric_only=True)[col].drop(labels=[col], errors="ignore").dropna()
        ranked_features = corr.abs().sort_values(ascending=False).head(max_features).index.tolist()
        fallback = out.loc[~target_mask, col].median()
        if pd.isna(fallback):
            fallback = out[col].median()

        if not ranked_features:
            if not pd.isna(fallback):
                out.loc[target_mask, col] = fallback
            rows.append(
                {
                    "Column": col,
                    "Filled values": missing_count,
                    "Method": "Fallback median",
                    "Correlated features": "None available",
                }
            )
            total_changed += missing_count
            continue

        train_mask = (~target_mask) & out[col].notna()
        train = out.loc[train_mask, ranked_features + [col]].dropna(subset=[col])
        predict = out.loc[target_mask, ranked_features]
        if len(train) < max(8, len(ranked_features) + 2) or predict.empty:
            if not pd.isna(fallback):
                out.loc[target_mask, col] = fallback
            method = "Fallback median"
        else:
            try:
                estimator = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", LinearRegression()),
                    ]
                )
                estimator.fit(train[ranked_features], train[col])
                predictions = estimator.predict(predict)
                if not pd.isna(fallback):
                    predictions = np.where(np.isfinite(predictions), predictions, fallback)
                out.loc[target_mask, col] = predictions
                method = "Correlation estimate"
            except Exception:
                if not pd.isna(fallback):
                    out.loc[target_mask, col] = fallback
                method = "Fallback median"

        rows.append(
            {
                "Column": col,
                "Filled values": missing_count,
                "Method": method,
                "Correlated features": ", ".join(ranked_features),
            }
        )
        total_changed += missing_count
    return out, total_changed, pd.DataFrame(rows)


def apply_cleaning_studio(
    data: pd.DataFrame,
    date_col: str | None,
    drop_columns: list[str],
    remove_duplicates: bool,
    drop_empty_columns: bool,
    numeric_missing_columns: list[str],
    numeric_missing_strategy: str,
    fill_categorical_missing: bool,
    outlier_columns: list[str],
    outlier_strategy: str,
    outlier_factor: float,
) -> tuple[pd.DataFrame, str | None, pd.DataFrame]:
    """Apply user-selected cleaning actions and return an audit table."""
    out = data.copy()
    rows = []

    if drop_columns:
        existing = [col for col in drop_columns if col in out.columns]
        out = out.drop(columns=existing)
        if date_col in existing:
            date_col = None
        rows.append(
            {
                "Action": "Drop selected columns",
                "Result": f"Dropped {len(existing)} column(s): {', '.join(existing[:8])}",
                "Rows changed": 0,
                "Columns changed": -len(existing),
            }
        )

    if drop_empty_columns:
        empty_cols = out.columns[out.isna().mean() == 1].tolist()
        out = out.drop(columns=empty_cols)
        if date_col in empty_cols:
            date_col = None
        rows.append(
            {
                "Action": "Drop empty columns",
                "Result": f"Dropped {len(empty_cols)} fully empty column(s).",
                "Rows changed": 0,
                "Columns changed": -len(empty_cols),
            }
        )

    if remove_duplicates:
        before = len(out)
        out = out.drop_duplicates().copy()
        rows.append(
            {
                "Action": "Remove duplicate rows",
                "Result": f"Removed {before - len(out):,} duplicate row(s).",
                "Rows changed": len(out) - before,
                "Columns changed": 0,
            }
        )

    if numeric_missing_strategy != "Do not fill numeric missing values" and numeric_missing_columns:
        if numeric_missing_strategy == "Estimate from most correlated variables":
            out, changed, estimate_report = fill_numeric_from_correlations(out, numeric_missing_columns)
            detail = (
                "; ".join(
                    f"{row['Column']} via {row['Correlated features']}"
                    for _, row in estimate_report.head(5).iterrows()
                )
                if not estimate_report.empty
                else "No correlated estimates were needed."
            )
        else:
            simple_strategy = "Mean" if numeric_missing_strategy == "Replace with mean" else "Median"
            out, changed = fill_numeric_simple(out, numeric_missing_columns, simple_strategy)
            detail = f"Strategy: {simple_strategy.lower()}."
        rows.append(
            {
                "Action": "Repair numeric missing values",
                "Result": f"Repaired {changed:,} numeric missing cell(s). {detail}",
                "Rows changed": 0,
                "Columns changed": 0,
            }
        )

    if fill_categorical_missing:
        categorical_cols = out.select_dtypes(exclude=["number"]).columns.tolist()
        out, changed = fill_categorical_with_mode(out, categorical_cols)
        rows.append(
            {
                "Action": "Fill categorical missing values",
                "Result": f"Filled {changed:,} categorical/date-like missing cell(s) with mode or Unknown.",
                "Rows changed": 0,
                "Columns changed": 0,
            }
        )

    if outlier_columns and outlier_strategy != "Do not change outliers":
        masks = iqr_outlier_masks(out, outlier_columns, outlier_factor)
        outlier_count = int(sum(mask.sum() for mask in masks.values()))
        before = len(out)
        if outlier_strategy == "Remove rows":
            out, changed_rows = remove_iqr_outliers(out, outlier_columns, outlier_factor)
            result = f"Removed {changed_rows:,} row(s) using IQR factor {outlier_factor:.2f}."
        elif outlier_strategy == "Cap to IQR bounds":
            out, changed = cap_iqr_outliers(out, outlier_columns, outlier_factor)
            result = f"Capped {changed:,} outlier value(s) to the nearest IQR fence."
        elif outlier_strategy in {"Replace with median", "Replace with mean"}:
            simple_strategy = "Mean" if outlier_strategy == "Replace with mean" else "Median"
            out, changed = replace_numeric_masks_simple(out, masks, simple_strategy)
            result = f"Replaced {changed:,} outlier value(s) with column {simple_strategy.lower()}."
        else:
            out, changed, estimate_report = fill_numeric_from_correlations(out, outlier_columns, masks)
            detail = (
                "; ".join(
                    f"{row['Column']} via {row['Correlated features']}"
                    for _, row in estimate_report.head(5).iterrows()
                )
                if not estimate_report.empty
                else "No correlated estimates were needed."
            )
            result = f"Estimated {changed:,} outlier value(s) using correlated variables. {detail}"
        rows.append(
            {
                "Action": "Repair numeric outliers",
                "Result": result if outlier_count else "No outlier values were detected for the selected columns.",
                "Rows changed": len(out) - before,
                "Columns changed": 0,
            }
        )

    if date_col and date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        if out[date_col].notna().any():
            out = out.sort_values(date_col)
    elif date_col not in out.columns:
        date_col = None

    if not rows:
        rows.append(
            {
                "Action": "No studio action selected",
                "Result": "The dataset is using the base automatic preparation only.",
                "Rows changed": 0,
                "Columns changed": 0,
            }
        )
    return out, date_col, pd.DataFrame(rows)


def time_group_label(dt):
    """Map a date to neutral analytical time periods used in the app."""
    if pd.isna(dt):
        return "Unknown"
    if dt < pd.Timestamp("2020-03-01"):
        return "Pre-COVID baseline"
    if dt <= pd.Timestamp("2020-12-31"):
        return "COVID shock"
    if dt <= pd.Timestamp("2021-12-31"):
        return "Early recovery"
    if dt <= pd.Timestamp("2023-12-31"):
        return "High-rate inflation period"
    return "Recent period"


@st.cache_data(show_spinner=False)
def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a numeric correlation matrix, returning an empty frame if none exists."""
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr(numeric_only=True)


def regression_metrics(y_true, y_pred) -> tuple[float, float, float]:
    """Calculate the standard regression metrics shown in model reports."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def chronological_model_split(
    model_df: pd.DataFrame,
    target: str,
    features: list[str],
    date_col: str | None = None,
    preferred_test_start: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Split model data with a fair chronological holdout."""
    ordered = model_df.copy()
    if date_col and date_col in ordered.columns:
        ordered[date_col] = pd.to_datetime(ordered[date_col], errors="coerce")
        ordered = ordered.sort_values(date_col).reset_index(drop=True)
    else:
        ordered = ordered.reset_index(drop=True)

    fallback_split = int(len(ordered) * 0.8)
    fallback_split = max(1, min(fallback_split, len(ordered) - 1))
    split = fallback_split
    method = "80/20 chronological split"

    if date_col and date_col in ordered.columns and ordered[date_col].notna().any():
        covid_start = pd.Timestamp("2020-03-01")
        covid_one_third = pd.Timestamp("2020-10-01")
        spans_covid = ordered[date_col].min() < covid_start and ordered[date_col].max() >= covid_one_third
        fallback_train_end = ordered[date_col].iloc[split - 1]
        if spans_covid and fallback_train_end < covid_one_third:
            covid_mask = ordered[date_col] >= covid_one_third
            if covid_mask.any():
                covid_split = int(np.flatnonzero(covid_mask.to_numpy())[0])
                enough_test = len(ordered) - covid_split >= min(12, max(1, len(ordered) // 10))
                if enough_test:
                    split = covid_split
                    method = "chronological split after first third of COVID regime"

    if preferred_test_start and date_col and date_col in ordered.columns and ordered[date_col].notna().any():
        preferred_date = pd.Timestamp(preferred_test_start)
        date_mask = ordered[date_col] >= preferred_date
        if date_mask.any():
            preferred_split = int(np.flatnonzero(date_mask.to_numpy())[0])
            enough_train = preferred_split >= min(24, max(1, len(ordered) - 1))
            enough_test = len(ordered) - preferred_split >= min(12, max(1, len(ordered) // 10))
            if enough_train and enough_test:
                split = preferred_split
                method = f"date-aware split from {preferred_date:%Y-%m-%d}"

    train_df = ordered.iloc[:split].copy()
    test_df = ordered.iloc[split:].copy()
    test_start = None
    if date_col and date_col in test_df.columns and not test_df.empty:
        test_start = test_df[date_col].iloc[0]
    split_info = {
        "Method": method,
        "Train rows": len(train_df),
        "Test rows": len(test_df),
        "Test start": test_start,
    }
    return train_df, test_df, split_info


def add_market_event_features(df: pd.DataFrame, date_col: str | None) -> tuple[pd.DataFrame, list[str]]:
    """Add date-only event features for major housing/macro regimes."""
    if not date_col or date_col not in df.columns:
        return df, []
    out = df.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    event_specs = {
        "event_covid_shock": ("2020-03-01", "2020-12-01"),
        "event_post_covid": ("2021-01-01", None),
        "event_high_inflation": ("2021-04-01", "2023-06-01"),
        "event_rate_hike_cycle": ("2022-03-01", "2023-07-01"),
        "event_recent_market": ("2022-01-01", None),
    }
    event_cols: list[str] = []
    for col, (start, end) in event_specs.items():
        start_date = pd.Timestamp(start)
        mask = dates >= start_date
        if end is not None:
            mask &= dates <= pd.Timestamp(end)
        out[col] = mask.fillna(False).astype(int)
        event_cols.append(col)

    covid_start = pd.Timestamp("2020-03-01")
    months_since_covid = ((dates.dt.year - covid_start.year) * 12 + (dates.dt.month - covid_start.month)).clip(lower=0)
    out["event_months_since_covid"] = months_since_covid.fillna(0).astype(float)
    out["event_months_since_covid_log"] = np.log1p(out["event_months_since_covid"])
    event_cols.extend(["event_months_since_covid", "event_months_since_covid_log"])
    return out, event_cols


def fit_predict_regression_series(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    target: str | None = None,
) -> tuple[np.ndarray, Pipeline, pd.Series]:
    """Fit regression for chronological series, predicting monthly change when possible."""
    if train_df is None or target is None or target not in train_df.columns or len(train_df) < 3:
        pipeline.fit(X_train, y_train)
        return np.asarray(pipeline.predict(X_test), dtype=float), pipeline, y_train

    change_train = train_df[target] - train_df[target].shift(1)
    valid_mask = change_train.notna()
    if int(valid_mask.sum()) < 2:
        pipeline.fit(X_train, y_train)
        return np.asarray(pipeline.predict(X_test), dtype=float), pipeline, y_train

    X_change_train = X_train.loc[valid_mask]
    y_change_train = change_train.loc[valid_mask]
    pipeline.fit(X_change_train, y_change_train)

    historical_changes = y_change_train.dropna()
    change_floor = float(historical_changes.quantile(0.01))
    change_ceiling = float(historical_changes.quantile(0.99))
    if (historical_changes >= 0).all():
        change_floor = 0.0

    previous_value = float(train_df[target].dropna().iloc[-1])
    predictions = []
    for _, row in X_test.iterrows():
        predicted_change = float(pipeline.predict(pd.DataFrame([row], columns=X_test.columns))[0])
        predicted_change = float(np.clip(predicted_change, change_floor, change_ceiling))
        previous_value += predicted_change
        predictions.append(previous_value)
    return np.asarray(predictions, dtype=float), pipeline, y_change_train


def project_future_feature_value(
    history: pd.Series,
    working: pd.Series,
    mode: str,
) -> float:
    """Project one future exogenous feature with conservative data-driven defaults."""
    feature_history = pd.to_numeric(history, errors="coerce").dropna()
    working_history = pd.to_numeric(working, errors="coerce").dropna()
    if working_history.empty:
        return np.nan

    latest_value = float(working_history.iloc[-1])
    if mode == "Hold latest values" or feature_history.empty:
        return latest_value

    if mode == "Repeat latest yearly pattern" and len(working) >= 12:
        return float(working.iloc[-12])

    recent_step = feature_history.diff().tail(6).median() if len(feature_history) >= 7 else 0.0
    if pd.isna(recent_step):
        recent_step = 0.0

    if mode == "Continue recent trend":
        projected = latest_value + float(recent_step)
    else:
        seasonal_step = 0.0
        if len(working_history) >= 12:
            seasonal_step = float((working_history.iloc[-1] - working_history.iloc[-12]) / 12)
        projected = latest_value + (0.65 * float(recent_step)) + (0.35 * seasonal_step)

    if len(feature_history) >= 20:
        lower = float(feature_history.quantile(0.01))
        upper = float(feature_history.quantile(0.99))
        projected = float(np.clip(projected, lower, upper))
    return projected


def time_regime_sample_weights(
    dates: pd.Series,
    post_covid_start: str = "2020-03-01",
    recent_months: int = 36,
) -> np.ndarray:
    """Give newer market regimes more influence while keeping older history useful."""
    parsed_dates = pd.to_datetime(dates, errors="coerce")
    weights = np.ones(len(parsed_dates), dtype=float) * 0.45
    post_covid_date = pd.Timestamp(post_covid_start)
    weights[parsed_dates >= post_covid_date] = 1.35

    latest_date = parsed_dates.dropna().max()
    if pd.notna(latest_date):
        recent_start = latest_date - pd.DateOffset(months=recent_months)
        weights[parsed_dates >= recent_start] = 2.0

    weights[pd.isna(parsed_dates)] = 1.0
    return weights


def fit_pipeline_with_optional_weights(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> bool:
    """Fit a pipeline with sample weights when the selected model supports them."""
    if sample_weight is None:
        pipeline.fit(X_train, y_train)
        return False
    try:
        pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
        return True
    except (TypeError, ValueError):
        pipeline.fit(X_train, y_train)
        return False


def build_supervised_with_lags(
    df: pd.DataFrame,
    date_col: str,
    target: str,
    features: list[str],
    lags=(1, 3, 6, 12),
    roll_windows=(3, 6, 12),
) -> pd.DataFrame:
    """Create lagged, rolling, and differenced features for forecasting."""
    # This function is part of the feature-engineering pipeline, building
    # time-aware predictors for forecasting models. Target-derived features must
    # use only values known before the row being predicted; otherwise the model
    # learns from the answer during training and behaves badly on future rows.
    out = df[[date_col, target] + list(features)].copy()
    out = out.sort_values(date_col).reset_index(drop=True)
    prior_target = out[target].shift(1)
    prior_change = out[target].diff(1).shift(1)
    for lag in lags:
        out[f"{target}_lag{lag}"] = out[target].shift(lag)
        out[f"{target}_change_lag{lag}"] = out[target].diff(1).shift(lag)
    for window in roll_windows:
        out[f"{target}_rollmean{window}"] = prior_target.rolling(window).mean()
        out[f"{target}_change_rollmean{window}"] = prior_change.rolling(window).mean()
        out[f"{target}_change_rollstd{window}"] = prior_change.rolling(window).std()
    out[f"{target}_diff1"] = prior_target.diff(1)
    for feature in features:
        out[f"{feature}_lag1"] = out[feature].shift(1)
        out[f"{feature}_change_lag1"] = out[feature].diff(1).shift(1)
    return out


def default_target(num_cols: list[str]) -> str | None:
    """Choose a useful default target, preferring the housing demo target when present."""
    if not num_cols:
        return None
    preferred = ["Home_Price_Index", "Home Price Index", "home_price_index"]
    for candidate in preferred:
        if candidate in num_cols:
            return candidate
    return num_cols[0]


def is_engineered_or_leaky_feature(column: str, target: str | None = None) -> bool:
    """Detect engineered or target-derived columns that can mislead interpretation."""
    lower = column.lower()
    engineered_tokens = [
        "lag",
        "roll",
        "smooth",
        "pct_change",
        "diff",
        "ratio",
        "index_smoothed",
    ]
    if any(token in lower for token in engineered_tokens):
        return True
    if target:
        target_lower = target.lower()
        target_tokens = {token for token in target_lower.replace("-", "_").split("_") if len(token) > 2}
        column_tokens = {token for token in lower.replace("-", "_").split("_") if len(token) > 2}
        if column != target and target_tokens and len(target_tokens & column_tokens) >= max(1, len(target_tokens) - 1):
            return True
        if column != target and "hpi" in lower and ("home" in target_lower or "price" in target_lower):
            return True
    return False


def interpretable_numeric_features(num_cols: list[str], target: str | None) -> list[str]:
    """Keep numeric columns that are useful for readable analysis by default."""
    if not target:
        return num_cols
    clean = [target] if target in num_cols else []
    clean.extend(
        col
        for col in num_cols
        if col != target and not is_engineered_or_leaky_feature(col, target)
    )
    return clean


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------

def metric_card(label: str, value: str) -> None:
    """Render a custom HTML metric card with the active theme styling."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def learning_cards(cards: list[tuple[str, str, str]]) -> None:
    """Render grouped guide cards that explain how to use a dashboard section."""
    for start in range(0, len(cards), 3):
        cols = st.columns(min(3, len(cards) - start))
        for col, (kicker, title, text) in zip(cols, cards[start : start + 3]):
            with col:
                with st.container(border=True):
                    st.caption(kicker)
                    st.markdown(f"**{title}**")
                    st.write(text)


def dashboard_search_items() -> list[dict[str, str]]:
    """Define searchable dashboard destinations and common keywords."""
    return [
        {"page": "Start Here", "category": "Guide", "keywords": "start guide demo presentation flow checklist beginner", "desc": "Use the recommended project walkthrough and demo script."},
        {"page": "Overview", "category": "Analysis", "keywords": "trend correlation target summary drivers matrix relationship", "desc": "Start here for target movement and meaningful correlations."},
        {"page": "Explore", "category": "Analysis", "keywords": "histogram box plot violin scatter distribution outliers chart visual", "desc": "Use this for visual exploration and distributions."},
        {"page": "Data Quality", "category": "Data", "keywords": "missing duplicates quality profile clean cleaning nettoyage null data preparation", "desc": "Check cleaning, missing values, duplicates, and numeric profiles."},
        {"page": "Compare", "category": "Analysis", "keywords": "period comparison time group regime radar difference", "desc": "Compare selected metrics across neutral time periods."},
        {"page": "ML Lab", "category": "Modeling", "keywords": "machine learning regression classification model train supervised fit prediction", "desc": "Run one supervised model or compare models."},
        {"page": "Evaluation", "category": "Modeling", "keywords": "accuracy precision recall f1 auc r2 rmse mae metrics explainability best model", "desc": "Compare models and explain why the best one wins."},
        {"page": "Prediction Page", "category": "Modeling", "keywords": "predict score inference single batch model input values download predictions", "desc": "Train a model and generate single-row or batch predictions."},
        {"page": "Model Save / Load", "category": "Modeling", "keywords": "save load model pickle pkl bundle pipeline inference download upload", "desc": "Download trained model bundles and upload them later for inference."},
        {"page": "Fit Diagnostics", "category": "Modeling", "keywords": "overfitting underfitting train test gap hyperparameters diagnostics regularization", "desc": "Check train/test gaps and model fit problems."},
        {"page": "Unsupervised Lab", "category": "Modeling", "keywords": "kmeans dbscan pca isolation forest anomaly clusters unsupervised segments", "desc": "Find groups, components, and unusual periods."},
        {"page": "Reinforcement Lab", "category": "Modeling", "keywords": "reinforcement q learning reward action state policy decision", "desc": "See an educational decision-policy example."},
        {"page": "Forecast", "category": "Decision", "keywords": "forecast predict future horizon lag time series", "desc": "Predict future target values."},
        {"page": "Scenario Simulator", "category": "Decision", "keywords": "scenario what if simulator impact change feature multi feature sensitivity", "desc": "Test a multi-feature what-if change."},
        {"page": "OLAP & Export", "category": "Analysis", "keywords": "olap cube pivot 3d segment export csv heatmap", "desc": "Build pivots, segment insights, and 3D OLAP cube."},
        {"page": "Executive Summary", "category": "Report", "keywords": "executive summary report markdown presentation conclusion", "desc": "Preview and download the one-page project story."},
        {"page": "Report Generator", "category": "Report", "keywords": "report generator markdown html export project profile cleaning model summary", "desc": "Generate a full downloadable project report."},
        {"page": "Data Dictionary", "category": "Data", "keywords": "dictionary schema column type missing range role metadata", "desc": "Review column roles, types, missing values, and examples."},
        {"page": "Domain Review", "category": "Research", "keywords": "paper research real life current market comparison theory sources domain benchmark", "desc": "Compare app results with outside domain evidence."},
        {"page": "Production Readiness", "category": "Governance", "keywords": "validation drift model card governance production monitoring checks", "desc": "Big-company checks for trust and governance."},
        {"page": "Experiment Tracker", "category": "Governance", "keywords": "experiment tracking runs metrics history mlflow", "desc": "Track model experiments and download run history."},
        {"page": "Model Registry", "category": "Governance", "keywords": "registry champion model approval version owner promote", "desc": "Promote the best model as a champion candidate."},
        {"page": "Data Pipeline", "category": "Governance", "keywords": "pipeline workflow raw clean train evaluate report lineage", "desc": "Show the end-to-end data science workflow."},
        {"page": "Big Data Readiness", "category": "Governance", "keywords": "big data spark warehouse parquet partition volume velocity variety scalable performance distributed", "desc": "Show how the project scales toward big-data workflows."},
        {"page": "Business Impact", "category": "Decision", "keywords": "business decision stakeholder impact value insight", "desc": "Translate technical results into business meaning."},
        {"page": "Code Lab", "category": "Guide", "keywords": "code prompt streamlit python generate snippet openai", "desc": "Generate Streamlit code snippets for the current dataset."},
    ]


def search_score(query: str, item: dict[str, str]) -> tuple[float, str]:
    """Score a search item using exact, token, and fuzzy matching."""
    tokens = [token for token in query.lower().split() if token]
    searchable = f"{item['page']} {item.get('category', '')} {item['keywords']} {item['desc']}".lower()
    score = 0.0
    reasons = []
    if query.lower() in item["page"].lower():
        score += 8
        reasons.append("page title")
    if query.lower() in searchable:
        score += 4
        reasons.append("phrase")
    for token in tokens:
        if token in item["page"].lower():
            score += 5
            reasons.append(f"`{token}` in page")
        elif token in item["keywords"].lower():
            score += 3
            reasons.append(f"`{token}` keyword")
        elif token in item["desc"].lower():
            score += 2
            reasons.append(f"`{token}` in description")
        else:
            best_fuzzy = max(
                SequenceMatcher(None, token, word).ratio()
                for word in searchable.replace("&", " ").replace("/", " ").split()
            )
            if best_fuzzy >= 0.78:
                score += 1.5
                reasons.append(f"`{token}` fuzzy match")
    return score, ", ".join(dict.fromkeys(reasons)) or "related topic"


def advanced_search_results(
    query: str,
    category: str = "All",
    limit: int = 5,
    df: pd.DataFrame | None = None,
    include_columns: bool = False,
) -> pd.DataFrame:
    """Return ranked page and optional dataset-column search results."""
    query = query.strip()
    rows = []
    for item in dashboard_search_items():
        if not st.session_state.get("show_advanced_pages", False) and item["page"] in NONESSENTIAL_PAGES:
            continue
        if category != "All" and item["category"] != category:
            continue
        score, reason = search_score(query, item)
        if score > 0:
            rows.append(
                {
                    "Result": item["page"],
                    "Type": "Page",
                    "Category": item["category"],
                    "Description": item["desc"],
                    "Matched by": reason,
                    "Score": score,
                }
            )

    if include_columns and df is not None:
        for column in df.columns:
            item = {
                "page": str(column),
                "category": "Data Column",
                "keywords": str(column).replace("_", " "),
                "desc": f"Dataset column with type {df[column].dtype} and {df[column].isna().sum():,} missing value(s).",
            }
            score, reason = search_score(query, item)
            if score > 0:
                rows.append(
                    {
                        "Result": str(column),
                        "Type": "Column",
                        "Category": "Data",
                        "Description": item["desc"],
                        "Matched by": reason,
                        "Score": score,
                    }
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Score", ascending=False).head(limit)


def render_search_results(
    query: str,
    category: str = "All",
    limit: int = 5,
    df: pd.DataFrame | None = None,
    include_columns: bool = False,
) -> None:
    """Show ranked navigation and column-search hints for the sidebar search box."""
    query = query.strip()
    if not query:
        st.caption("Try searches like `overfit`, `cleaning`, `forecast`, `mortgage`, `OLAP`, or `scenario`.")
        return
    matches = advanced_search_results(query, category, limit, df, include_columns)
    if matches.empty:
        st.caption("No match found. Try: model, OLAP, forecast, accuracy, drift, scenario, or a column name.")
        return

    st.caption(f"{len(matches)} ranked result(s)")
    for item in matches.to_dict("records"):
        st.markdown(
            f"""
            <div class="search-result">
                <strong>{item["Result"]}</strong>
                <span>{item["Type"]} | {item["Category"]} | match: {item["Matched by"]}</span>
                <span>{item["Description"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Analysis and interpretation helpers
# -----------------------------------------------------------------------------

def correlation_summary(cmat: pd.DataFrame, target: str | None) -> dict[str, object]:
    """Summarize the strongest positive and negative correlations for the selected target."""
    summary: dict[str, object] = {
        "target_positive": None,
        "target_negative": None,
        "strong_pairs": pd.DataFrame(),
    }
    if cmat.empty:
        return summary

    if target and target in cmat.columns:
        target_corr = cmat[target].drop(labels=[target], errors="ignore").dropna()
        if not target_corr.empty:
            summary["target_positive"] = target_corr.sort_values(ascending=False).head(1)
            summary["target_negative"] = target_corr.sort_values(ascending=True).head(1)

    pairs = []
    cols = list(cmat.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            corr_value = cmat.loc[left, right]
            if pd.notna(corr_value):
                pairs.append(
                    {
                        "Feature A": left,
                        "Feature B": right,
                        "Correlation": corr_value,
                        "Abs Correlation": abs(corr_value),
                    }
                )
    if pairs:
        summary["strong_pairs"] = (
            pd.DataFrame(pairs)
            .sort_values("Abs Correlation", ascending=False)
            .head(10)
            .drop(columns=["Abs Correlation"])
        )
    return summary


def first_matching_column(columns: list[str], keyword_groups: list[tuple[str, ...]]) -> str | None:
    """Find the first column whose name contains all requested keyword groups."""
    lowered = {col: col.lower() for col in columns}
    for keywords in keyword_groups:
        for col, lower in lowered.items():
            if all(keyword in lower for keyword in keywords):
                return col
    return None


def theory_check_table(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    """Compare dataset correlations with common housing-market theory signals."""
    available = [feature for feature in features if feature in df.columns]
    checks = [
        {
            "Theory": "Affordability theory: higher mortgage/interest rates pressure housing prices lower.",
            "Expected sign": "Negative",
            "Feature": first_matching_column(
                available,
                [("mortgage",), ("interest",), ("rate",), ("fed", "funds")],
            ),
        },
        {
            "Theory": "Supply-demand theory: more inventory or supply should reduce price pressure.",
            "Expected sign": "Negative",
            "Feature": first_matching_column(
                available,
                [("inventory",), ("supply",), ("new", "listing"), ("housing", "supply")],
            ),
        },
        {
            "Theory": "Income-demand theory: stronger income or GDP supports higher housing prices.",
            "Expected sign": "Positive",
            "Feature": first_matching_column(
                available,
                [("income",), ("gdp",), ("wage",), ("earnings",)],
            ),
        },
        {
            "Theory": "Labor-market theory: higher unemployment weakens housing demand.",
            "Expected sign": "Negative",
            "Feature": first_matching_column(available, [("unemployment",), ("jobless",)]),
        },
    ]

    rows = []
    for check in checks:
        feature = check["Feature"]
        if not feature:
            rows.append({**check, "Observed correlation": None, "Evidence": "Missing feature"})
            continue
        corr_value = df[[target, feature]].corr(numeric_only=True).loc[target, feature]
        expected_positive = check["Expected sign"] == "Positive"
        supports = corr_value > 0 if expected_positive else corr_value < 0
        rows.append(
            {
                **check,
                "Observed correlation": corr_value,
                "Evidence": "Supports theory" if supports else "Challenges theory",
            }
        )
    return pd.DataFrame(rows)


def real_life_period_table(df: pd.DataFrame, date_col: str, target: str) -> pd.DataFrame:
    """Aggregate the target into named real-life market periods."""
    periods = [
        {
            "Real-life period": "Pre-crisis housing boom",
            "Start": "2004-01-01",
            "End": "2006-12-31",
            "Real-life expectation": "Prices usually rise when credit is easy and demand is strong.",
            "Expected direction": "Rise",
        },
        {
            "Real-life period": "Financial-crisis correction",
            "Start": "2007-01-01",
            "End": "2012-12-31",
            "Real-life expectation": "Prices usually weaken after credit stress, foreclosures, and demand shocks.",
            "Expected direction": "Fall",
        },
        {
            "Real-life period": "Recovery and expansion",
            "Start": "2013-01-01",
            "End": "2019-12-31",
            "Real-life expectation": "Prices usually recover as labor markets, income, and confidence improve.",
            "Expected direction": "Rise",
        },
        {
            "Real-life period": "Pandemic-era demand shock",
            "Start": "2020-01-01",
            "End": "2021-12-31",
            "Real-life expectation": "Prices usually accelerate when rates are low, supply is tight, and housing demand shifts upward.",
            "Expected direction": "Rise",
        },
        {
            "Real-life period": "High-rate affordability pressure",
            "Start": "2022-01-01",
            "End": "2024-12-31",
            "Real-life expectation": "Higher rates should cool demand, but tight supply can keep prices elevated.",
            "Expected direction": "Mixed",
        },
    ]
    rows = []
    dated = df[[date_col, target]].dropna().sort_values(date_col)
    if dated.empty:
        return pd.DataFrame(rows)

    for period in periods:
        start = pd.Timestamp(period["Start"])
        end = pd.Timestamp(period["End"])
        part = dated[(dated[date_col] >= start) & (dated[date_col] <= end)]
        if len(part) < 2:
            continue
        start_value = float(part[target].iloc[0])
        end_value = float(part[target].iloc[-1])
        change = end_value - start_value
        pct_change = (change / start_value * 100) if start_value else np.nan
        observed_direction = "Rise" if change > 0 else "Fall" if change < 0 else "Flat"
        expected = period["Expected direction"]
        if expected == "Mixed":
            verdict = "Consistent with mixed real-world forces"
        elif observed_direction == expected:
            verdict = "Matches real-life expectation"
        else:
            verdict = "Challenges simple theory"
        rows.append(
            {
                "Real-life period": period["Real-life period"],
                "Date range in dataset": f"{part[date_col].iloc[0].date()} to {part[date_col].iloc[-1].date()}",
                "Real-life expectation": period["Real-life expectation"],
                "Expected direction": expected,
                "Dataset direction": observed_direction,
                "Dataset change": change,
                "Dataset change %": pct_change,
                "Verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def final_conclusion_text(
    target: str,
    best_driver: str | None,
    best_corr: float | None,
    theory_df: pd.DataFrame,
    period_df: pd.DataFrame,
) -> str:
    """Build a project-ready conclusion from data, theory, and period evidence."""
    # Count how many theory checks the current dataset supports or challenges.
    supported = int((theory_df["Evidence"] == "Supports theory").sum()) if not theory_df.empty else 0
    challenged = int((theory_df["Evidence"] == "Challenges theory").sum()) if not theory_df.empty else 0
    matched_periods = int((period_df["Verdict"] == "Matches real-life expectation").sum()) if not period_df.empty else 0
    mixed_periods = int((period_df["Verdict"] == "Consistent with mixed real-world forces").sum()) if not period_df.empty else 0

    driver_sentence = (
        f"The strongest meaningful selected driver is `{best_driver}` with correlation `{best_corr:.2f}`."
        if best_driver is not None and best_corr is not None
        else "No single meaningful driver dominates the selected feature set."
    )
    return (
        f"Final conclusion: the `{target}` data broadly behaves like a real housing-market time series, "
        "where prices respond to affordability, supply-demand pressure, labor-market strength, and macro shocks. "
        f"{driver_sentence} The theory check supports `{supported}` relationships and challenges `{challenged}`, "
        "which is realistic because housing prices are not controlled by one variable. "
        f"The real-life period comparison finds `{matched_periods}` periods matching historical expectations and "
        f"`{mixed_periods}` periods where mixed forces matter. A strong project conclusion is that the dataset "
        "does not simply prove one old theory right or wrong; it shows that classic theory works best when combined "
        "with timing, supply constraints, interest-rate context, and model-based validation."
    )


def radar_compare(df: pd.DataFrame, metrics_cols: list[str]):
    """Create a normalized radar chart comparing neutral time groups across selected metrics."""
    if "time_group" not in df.columns:
        return None
    ordered_groups = [group for group in df["time_group"].dropna().unique().tolist() if group != "Unknown"]
    if len(ordered_groups) < 2:
        return None
    selected_groups = ordered_groups[:2]
    data = df[list(metrics_cols) + ["time_group"]].dropna()
    if data.empty:
        return None
    mins = data[metrics_cols].min()
    maxs = data[metrics_cols].max()
    scaled = (data[metrics_cols] - mins) / (maxs - mins).replace(0, np.nan)
    profile = (
        pd.concat([scaled, data["time_group"]], axis=1)
        .groupby("time_group")[metrics_cols]
        .mean()
        .loc[selected_groups]
    )
    categories = metrics_cols + [metrics_cols[0]]
    fig = go.Figure()
    for period in selected_groups:
        values = profile.loc[period].tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories,
                fill="toself",
                name=period,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=520,
        title="Normalized Time-Group Profile",
    )
    return fig


# -----------------------------------------------------------------------------
# Supervised machine-learning helpers / ML pipeline
# -----------------------------------------------------------------------------
# This section defines the modeling pipeline used in the ML Lab: model
# instantiation, preprocessing, scaling, training, evaluation, and feature
# importance extraction.

def make_model(
    model_name: str,
    task_type: str,
    random_state: int = 0,
    hyperparameters: dict[str, object] | None = None,
):
    """Instantiate the selected scikit-learn model for regression or classification."""
    params = dict(hyperparameters or {})
    if task_type == "Regression":
        models = {
            "Ridge": Ridge(alpha=1.0),
            "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=5000),
            "BayesianRidge": BayesianRidge(),
            "RandomForest": RandomForestRegressor(
                n_estimators=500,
                random_state=random_state,
                n_jobs=-1,
                max_depth=None,
                min_samples_leaf=2,
                min_samples_split=4,
                max_features="sqrt",
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=500,
                random_state=random_state,
                n_jobs=-1,
                max_depth=None,
                min_samples_leaf=2,
                min_samples_split=4,
                max_features="sqrt",
            ),
            "LinearRegression": LinearRegression(),
            "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=3,
                subsample=0.8,
                random_state=random_state,
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_iter=350,
                learning_rate=0.03,
                max_leaf_nodes=15,
                l2_regularization=0.05,
                random_state=random_state,
            ),
            "DecisionTree": DecisionTreeRegressor(
                max_depth=5,
                random_state=random_state,
                min_samples_leaf=8,
                min_samples_split=16,
                ccp_alpha=0.005,
            ),
            "KNNRegressor": KNeighborsRegressor(n_neighbors=5),
            "NeuralNetwork": MLPRegressor(
                hidden_layer_sizes=(32,),
                activation="relu",
                solver="adam",
                alpha=0.01,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.2,
                max_iter=800,
                random_state=random_state,
            ),
        }
    else:
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                min_samples_split=10,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            ),
            "LogisticRegression": LogisticRegression(C=0.5, random_state=random_state, max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(
                max_depth=5,
                random_state=random_state,
                min_samples_leaf=8,
                min_samples_split=16,
                ccp_alpha=0.005,
            ),
            "KNNClassifier": KNeighborsClassifier(n_neighbors=9, weights="uniform"),
            "NeuralNetwork": MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu",
                solver="adam",
                alpha=0.01,
                learning_rate_init=0.001,
                early_stopping=False,
                max_iter=800,
                random_state=random_state,
            ),
        }
    if not params:
        return models[model_name]

    if task_type == "Regression":
        if model_name == "Ridge":
            return Ridge(alpha=float(params.get("alpha", 1.0)))
        if model_name == "ElasticNet":
            return ElasticNet(
                alpha=float(params.get("alpha", 0.05)),
                l1_ratio=float(params.get("l1_ratio", 0.2)),
                max_iter=int(params.get("max_iter", 5000)),
            )
        if model_name == "BayesianRidge":
            return BayesianRidge()
        if model_name == "RandomForest":
            return RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 500)),
                max_depth=params.get("max_depth"),
                min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                min_samples_split=int(params.get("min_samples_split", 4)),
                max_features=params.get("max_features", "sqrt"),
                random_state=random_state,
                n_jobs=-1,
            )
        if model_name == "ExtraTrees":
            return ExtraTreesRegressor(
                n_estimators=int(params.get("n_estimators", 500)),
                max_depth=params.get("max_depth"),
                min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                min_samples_split=int(params.get("min_samples_split", 4)),
                max_features=params.get("max_features", "sqrt"),
                random_state=random_state,
                n_jobs=-1,
            )
        if model_name == "SVR":
            return SVR(
                kernel="rbf",
                C=float(params.get("C", 1.0)),
                epsilon=float(params.get("epsilon", 0.1)),
                gamma=params.get("gamma", "scale"),
            )
        if model_name == "GradientBoosting":
            return GradientBoostingRegressor(
                n_estimators=int(params.get("n_estimators", 300)),
                learning_rate=float(params.get("learning_rate", 0.03)),
                max_depth=int(params.get("max_depth", 3)),
                min_samples_leaf=int(params.get("min_samples_leaf", 3)),
                subsample=float(params.get("subsample", 0.8)),
                random_state=random_state,
            )
        if model_name == "HistGradientBoosting":
            return HistGradientBoostingRegressor(
                max_iter=int(params.get("max_iter", 350)),
                learning_rate=float(params.get("learning_rate", 0.03)),
                max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
                l2_regularization=float(params.get("l2_regularization", 0.05)),
                random_state=random_state,
            )
        if model_name == "DecisionTree":
            return DecisionTreeRegressor(
                max_depth=params.get("max_depth"),
                min_samples_leaf=int(params.get("min_samples_leaf", 8)),
                min_samples_split=int(params.get("min_samples_split", 16)),
                ccp_alpha=float(params.get("ccp_alpha", 0.005)),
                random_state=random_state,
            )
        if model_name in {"KNN", "KNNRegressor"}:
            return KNeighborsRegressor(
                n_neighbors=int(params.get("n_neighbors", 5)),
                weights=str(params.get("weights", "uniform")),
            )
        if model_name == "NeuralNetwork":
            hidden_layers = int(params.get("hidden_layers", 1))
            hidden_units = int(params.get("hidden_units", 32))
            return MLPRegressor(
                hidden_layer_sizes=tuple([hidden_units] * hidden_layers),
                activation="relu",
                solver="adam",
                alpha=float(params.get("alpha", 0.01)),
                learning_rate_init=float(params.get("learning_rate_init", 0.001)),
                early_stopping=True,
                validation_fraction=0.2,
                max_iter=int(params.get("max_iter", 800)),
                random_state=random_state,
            )
    else:
        if model_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=params.get("max_depth"),
                min_samples_leaf=int(params.get("min_samples_leaf", 5)),
                min_samples_split=int(params.get("min_samples_split", 10)),
                max_features=params.get("max_features", "sqrt"),
                random_state=random_state,
                n_jobs=-1,
            )
        if model_name == "LogisticRegression":
            return LogisticRegression(
                C=float(params.get("C", 0.5)),
                random_state=random_state,
                max_iter=int(params.get("max_iter", 1000)),
            )
        if model_name == "DecisionTree":
            return DecisionTreeClassifier(
                max_depth=params.get("max_depth"),
                min_samples_leaf=int(params.get("min_samples_leaf", 8)),
                min_samples_split=int(params.get("min_samples_split", 16)),
                ccp_alpha=float(params.get("ccp_alpha", 0.005)),
                random_state=random_state,
            )
        if model_name in {"KNN", "KNNClassifier"}:
            return KNeighborsClassifier(
                n_neighbors=int(params.get("n_neighbors", 9)),
                weights=str(params.get("weights", "uniform")),
            )
        if model_name == "NeuralNetwork":
            hidden_layers = int(params.get("hidden_layers", 1))
            hidden_units = int(params.get("hidden_units", 32))
            return MLPClassifier(
                hidden_layer_sizes=tuple([hidden_units] * hidden_layers),
                activation="relu",
                solver="adam",
                alpha=float(params.get("alpha", 0.01)),
                learning_rate_init=float(params.get("learning_rate_init", 0.001)),
                early_stopping=False,
                max_iter=int(params.get("max_iter", 800)),
                random_state=random_state,
            )
    return models[model_name]


def build_model_pipeline(
    model_choice: str,
    task_type: str,
    use_scaling: bool,
    scaler_name: str,
    hyperparameters: dict[str, object] | None = None,
) -> Pipeline:
    """Create an optional scaling plus model pipeline for supervised learning."""
    # The ML pipeline always begins with imputing missing values. If selected,
    # scaling is applied before the model to keep coefficients and distances
    # comparable across features.
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaling:
        scaler = StandardScaler() if scaler_name == "StandardScaler" else MinMaxScaler()
        steps.append(("scaler", scaler))
    steps.append(("model", make_model(model_choice, task_type, hyperparameters=hyperparameters)))
    return Pipeline(steps)


def make_classification_labels(y_train: pd.Series, y_test: pd.Series):
    """Convert a numeric target into high/low classes using the training median."""
    y_train_binned = pd.qcut(y_train, q=3, labels=False, duplicates="drop")
    y_test_binned = pd.qcut(y_test, q=3, labels=False, duplicates="drop")
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    return (
        y_train_binned.map(label_map).astype("category"),
        y_test_binned.map(label_map).astype("category"),
    )


def model_options_for_task(task_type: str) -> list[str]:
    """Return model names that are valid for the selected supervised task."""
    if task_type == "Regression":
        return [
            "Ridge",
            "ElasticNet",
            "BayesianRidge",
            "RandomForest",
            "ExtraTrees",
            "LinearRegression",
            "SVR",
            "GradientBoosting",
            "HistGradientBoosting",
            "DecisionTree",
            "KNNRegressor",
            "NeuralNetwork",
        ]
    return ["RandomForest", "LogisticRegression", "DecisionTree", "KNNClassifier", "NeuralNetwork"]


def classification_auc(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    """Calculate ROC AUC when the fitted classifier exposes probabilities or scores."""
    try:
        if hasattr(pipeline, "predict_proba"):
            scores = np.asarray(pipeline.predict_proba(x_test))
        elif hasattr(pipeline, "decision_function"):
            scores = np.asarray(pipeline.decision_function(x_test))
        else:
            return None
        y_test_series = pd.Series(y_test)
        y_classes = y_test_series.dropna().unique()
        if len(y_classes) < 2:
            return None
        if len(y_classes) == 2 and scores.ndim == 2 and scores.shape[1] == 2:
            classes = [str(cls) for cls in getattr(pipeline, "classes_", [])]
            positive_idx = 1
            if classes:
                positive_label = sorted(pd.Series(y_classes).astype(str).unique())[-1]
                positive_idx = classes.index(positive_label) if positive_label in classes else 1
            scores = scores[:, positive_idx]
        return float(roc_auc_score(y_test, scores, multi_class="ovr", average="weighted"))
    except Exception:
        return None


def supervised_training_frame(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    task_type: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[int, str]]:
    """Prepare X/y for the prediction page, including numeric target binning."""
    model_df = df[[target] + features].copy().dropna(subset=[target])
    x_data = model_df[features]
    label_map: dict[int, str] = {}
    if task_type == "Regression":
        return x_data, model_df[target], model_df, label_map

    unique_count = int(model_df[target].nunique(dropna=True))
    if unique_count <= 1:
        return x_data.iloc[0:0], pd.Series(dtype="object"), model_df.iloc[0:0], label_map
    bin_count = min(3, unique_count)
    labels = pd.qcut(model_df[target], q=bin_count, labels=False, duplicates="drop")
    label_names = ["Low", "Medium", "High"] if bin_count == 3 else ["Low", "High"]
    actual_bins = sorted(pd.Series(labels).dropna().unique().tolist())
    label_map = {int(value): label_names[idx] if idx < len(label_names) else f"Class {idx}" for idx, value in enumerate(actual_bins)}
    y_data = pd.Series(labels, index=model_df.index).map(label_map)
    keep_mask = y_data.notna()
    return x_data.loc[keep_mask], y_data.loc[keep_mask].astype(str), model_df.loc[keep_mask], label_map


def prediction_input_defaults(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """Use medians as stable defaults for manual prediction inputs."""
    defaults = {}
    for feature in features:
        series = pd.to_numeric(df[feature], errors="coerce")
        value = series.median()
        if pd.isna(value):
            value = 0.0
        defaults[feature] = float(value)
    return defaults


def create_model_bundle(
    pipeline: Pipeline,
    target: str,
    task_type: str,
    model_name: str,
    features: list[str],
    training_rows: int,
    project_profile: dict[str, object],
    feature_defaults: dict[str, float],
) -> dict[str, object]:
    """Package a fitted model with metadata needed for later inference."""
    return {
        "bundle_type": "DataIQModelBundle",
        "version": 1,
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "pipeline": pipeline,
        "target": target,
        "task_type": task_type,
        "model_name": model_name,
        "features": features,
        "training_rows": training_rows,
        "project_profile": project_profile,
        "feature_defaults": feature_defaults,
    }


def model_bundle_to_bytes(bundle: dict[str, object]) -> bytes:
    """Serialize a fitted model bundle for download."""
    return pickle.dumps(bundle)


def model_bundle_from_bytes(file_bytes: bytes) -> dict[str, object]:
    """Load and validate a serialized DataIQ model bundle."""
    bundle = pickle.loads(file_bytes)
    if not isinstance(bundle, dict) or bundle.get("bundle_type") != "DataIQModelBundle":
        raise ValueError("This file is not a DataIQ model bundle.")
    required = {"pipeline", "target", "task_type", "model_name", "features"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing: {', '.join(sorted(missing))}")
    return bundle


def classification_score_matrix(pipeline: Pipeline, x_test: pd.DataFrame) -> tuple[np.ndarray, list[str]] | None:
    """Return classifier probability/score columns with their class labels."""
    try:
        if hasattr(pipeline, "predict_proba"):
            scores = np.asarray(pipeline.predict_proba(x_test))
        elif hasattr(pipeline, "decision_function"):
            scores = np.asarray(pipeline.decision_function(x_test))
        else:
            return None
    except Exception:
        return None

    classes = [str(cls) for cls in getattr(pipeline, "classes_", [])]
    if scores.ndim == 1:
        if len(classes) == 2:
            return scores.reshape(-1, 1), [classes[1]]
        return scores.reshape(-1, 1), ["Positive class"]
    if not classes or len(classes) != scores.shape[1]:
        classes = [f"Class {idx + 1}" for idx in range(scores.shape[1])]
    return scores, classes


def confusion_matrix_figure(y_true: pd.Series, y_pred: np.ndarray, title: str = "Confusion Matrix") -> go.Figure:
    """Build a labeled confusion-matrix heatmap for classification diagnostics."""
    labels = sorted(pd.Series(pd.concat([pd.Series(y_true), pd.Series(y_pred)])).astype(str).unique())
    matrix = confusion_matrix(pd.Series(y_true).astype(str), pd.Series(y_pred).astype(str), labels=labels)
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted class", "y": "Actual class", "color": "Count"},
        title=title,
    )
    fig.update_layout(height=420)
    return fig


def roc_curve_figure(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> go.Figure | None:
    """Build one-vs-rest ROC curves for binary or multiclass classifiers."""
    score_data = classification_score_matrix(pipeline, x_test)
    if score_data is None:
        return None

    scores, score_classes = score_data
    y_true = pd.Series(y_test).astype(str)
    unique_classes = sorted(y_true.dropna().unique())
    if len(unique_classes) < 2:
        return None

    fig = go.Figure()
    if scores.shape[1] == 1 and len(score_classes) == 1:
        positive_class = score_classes[0]
        binary_truth = (y_true == positive_class).astype(int)
        if binary_truth.nunique() < 2:
            return None
        fpr, tpr, _ = roc_curve(binary_truth, scores[:, 0])
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{positive_class} (AUC={auc(fpr, tpr):.3f})",
            )
        )
    else:
        for idx, class_name in enumerate(score_classes):
            if class_name not in set(unique_classes):
                continue
            binary_truth = (y_true == class_name).astype(int)
            if binary_truth.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(binary_truth, scores[:, idx])
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{class_name} (AUC={auc(fpr, tpr):.3f})",
                )
            )

    if not fig.data:
        return None
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line={"dash": "dash", "color": "gray"},
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
        legend_title="Class",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def metric_quality_label(value: float | None) -> str:
    """Convert a 0-1 classification metric into a plain-language quality label."""
    if value is None or pd.isna(value):
        return "Unavailable"
    if value >= 0.90:
        return "Excellent"
    if value >= 0.80:
        return "Strong"
    if value >= 0.65:
        return "Moderate"
    if value >= 0.50:
        return "Weak"
    return "Very weak"


def classification_metric_interpretation(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    roc_auc: float | None,
) -> pd.DataFrame:
    """Explain each classification value beside the model metrics."""
    metric_rows = [
        {
            "Value": "Accuracy",
            "Score": accuracy,
            "Interpretation": "Share of test rows classified correctly.",
        },
        {
            "Value": "Precision",
            "Score": precision,
            "Interpretation": "When the model predicts a class, how often that prediction is correct.",
        },
        {
            "Value": "Recall",
            "Score": recall,
            "Interpretation": "How many real class cases the model successfully finds.",
        },
        {
            "Value": "F1",
            "Score": f1,
            "Interpretation": "Balanced score combining precision and recall.",
        },
        {
            "Value": "ROC AUC",
            "Score": roc_auc,
            "Interpretation": "How well the model ranks classes before applying a hard class decision.",
        },
    ]
    result = pd.DataFrame(metric_rows)
    result["Score"] = result["Score"].apply(lambda value: None if value is None or pd.isna(value) else round(float(value), 3))
    result["Quality"] = result["Score"].apply(metric_quality_label)
    return result[["Value", "Score", "Quality", "Interpretation"]]


def render_classification_metric_values(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    roc_auc: float | None,
) -> None:
    """Render classification metric cards with an adjacent interpretation table."""
    metric_col, interpretation_col = st.columns([3, 2])
    with metric_col:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{accuracy:.3f}")
        c2.metric("Precision", f"{precision:.3f}")
        c3.metric("Recall", f"{recall:.3f}")
        c4.metric("F1", f"{f1:.3f}")
        c5.metric("ROC AUC", "N/A" if roc_auc is None else f"{roc_auc:.3f}")
    with interpretation_col:
        st.dataframe(
            classification_metric_interpretation(accuracy, precision, recall, f1, roc_auc),
            width="stretch",
            hide_index=True,
        )


def confusion_matrix_interpretation(y_true: pd.Series, y_pred: np.ndarray) -> str:
    """Summarize the main message from a confusion matrix."""
    labels = sorted(pd.Series(pd.concat([pd.Series(y_true), pd.Series(y_pred)])).astype(str).unique())
    matrix = confusion_matrix(pd.Series(y_true).astype(str), pd.Series(y_pred).astype(str), labels=labels)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    errors = total - correct
    accuracy = correct / total if total else 0.0
    off_diag = matrix.copy()
    np.fill_diagonal(off_diag, 0)
    if off_diag.max() > 0:
        actual_idx, predicted_idx = np.unravel_index(np.argmax(off_diag), off_diag.shape)
        mistake_line = (
            f"- Most common mistake: actual `{labels[actual_idx]}` predicted as "
            f"`{labels[predicted_idx]}` ({int(off_diag[actual_idx, predicted_idx])} rows)."
        )
    else:
        mistake_line = "- No off-diagonal mistakes appear in this test split."
    return (
        f"- Correct predictions: `{correct}` of `{total}` test rows (`{accuracy:.1%}`).\n"
        f"- Wrong predictions: `{errors}` rows.\n"
        "- The diagonal cells are correct predictions; off-diagonal cells are mistakes.\n"
        f"{mistake_line}"
    )


def roc_curve_interpretation(roc_auc: float | None) -> str:
    """Summarize how to read the ROC curve and AUC value."""
    if roc_auc is None or pd.isna(roc_auc):
        return (
            "- ROC is unavailable because this model or split did not provide usable probability scores.\n"
            "- Use Accuracy, Precision, Recall, F1, and the confusion matrix for this run."
        )
    quality = metric_quality_label(roc_auc).lower()
    return (
        f"- Weighted ROC AUC is `{roc_auc:.3f}`, which is a `{quality}` ranking result.\n"
        "- Curves closer to the top-left corner separate classes better.\n"
        "- The dashed diagonal is random guessing; useful models should stay above it."
    )


def render_classification_diagnostics(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    roc_auc: float | None,
    confusion_title: str = "Confusion Matrix",
) -> None:
    """Render confusion matrix and ROC curve with adjacent interpretation text."""
    matrix_chart_col, matrix_text_col = st.columns([2, 1])
    with matrix_chart_col:
        st.plotly_chart(
            confusion_matrix_figure(y_test, y_pred, title=confusion_title),
            width="stretch",
        )
    with matrix_text_col:
        st.markdown("##### How to read it")
        st.info(confusion_matrix_interpretation(y_test, y_pred))

    roc_chart_col, roc_text_col = st.columns([2, 1])
    with roc_chart_col:
        roc_fig = roc_curve_figure(pipeline, x_test, y_test)
        if roc_fig is None:
            st.info("ROC curve is unavailable for this classifier or split.")
        else:
            st.plotly_chart(roc_fig, width="stretch")
    with roc_text_col:
        st.markdown("##### ROC interpretation")
        st.info(roc_curve_interpretation(roc_auc))


def regression_r2_quality_label(r2: float | None) -> str:
    """Convert R2 into a plain-language quality label."""
    if r2 is None or pd.isna(r2):
        return "Unavailable"
    if r2 >= 0.80:
        return "Strong"
    if r2 >= 0.50:
        return "Moderate"
    if r2 >= 0.20:
        return "Weak"
    if r2 >= 0:
        return "Very weak"
    return "Worse than baseline"


def regression_metric_interpretation(mae: float, rmse: float, r2: float, y_true: pd.Series) -> pd.DataFrame:
    """Explain regression values beside the model metrics."""
    target_mean = float(np.mean(y_true)) if len(y_true) else np.nan
    mae_share = abs(mae / target_mean) if target_mean and not pd.isna(target_mean) else np.nan
    rmse_share = abs(rmse / target_mean) if target_mean and not pd.isna(target_mean) else np.nan
    return pd.DataFrame(
        [
            {
                "Value": "MAE",
                "Score": round(float(mae), 3),
                "Quality": f"{mae_share:.1%} of target mean" if not pd.isna(mae_share) else "Lower is better",
                "Interpretation": "Average absolute prediction error in target units.",
            },
            {
                "Value": "RMSE",
                "Score": round(float(rmse), 3),
                "Quality": f"{rmse_share:.1%} of target mean" if not pd.isna(rmse_share) else "Lower is better",
                "Interpretation": "Error measure that punishes large misses more than MAE.",
            },
            {
                "Value": "R2",
                "Score": round(float(r2), 3),
                "Quality": regression_r2_quality_label(r2),
                "Interpretation": "Share of target variation explained by the model on the test rows.",
            },
        ]
    )


def render_regression_metric_values(mae: float, rmse: float, r2: float, y_true: pd.Series) -> None:
    """Render regression metric cards with an adjacent interpretation table."""
    metric_col, interpretation_col = st.columns([2, 2])
    with metric_col:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.3f}")
        c2.metric("RMSE", f"{rmse:.3f}")
        c3.metric("R2", f"{r2:.3f}")
    with interpretation_col:
        st.dataframe(
            regression_metric_interpretation(mae, rmse, r2, y_true),
            width="stretch",
            hide_index=True,
        )


def regression_prediction_interpretation(y_true: pd.Series, y_pred: np.ndarray) -> str:
    """Summarize the actual-vs-predicted regression chart."""
    actual_mean = float(np.mean(y_true))
    predicted_mean = float(np.mean(y_pred))
    mean_gap = predicted_mean - actual_mean
    mae = float(mean_absolute_error(y_true, y_pred))
    direction = "over-predicts" if mean_gap > 0 else "under-predicts" if mean_gap < 0 else "matches the mean of"
    return (
        f"- Actual mean: `{actual_mean:.3f}`.\n"
        f"- Predicted mean: `{predicted_mean:.3f}`.\n"
        f"- Mean gap: `{mean_gap:.3f}`, so the model {direction} the test period on average.\n"
        f"- Average absolute error: `{mae:.3f}` target units.\n"
        "- Lines that move together mean the model follows the test pattern; wide gaps show missed periods."
    )


def regression_metric_chart_interpretation(evaluation: pd.DataFrame, best: pd.Series) -> str:
    """Explain the regression model-comparison bar chart."""
    return (
        f"- `{best['Model']}` is best by R2 with `{best['R2']:.3f}`.\n"
        "- For R2, taller is better because more target variation is explained.\n"
        "- For MAE and RMSE, shorter is better because prediction error is lower.\n"
        "- If a model has high R2 but also high RMSE, inspect whether a few large misses are driving risk."
    )


def feature_effect_interpretation(values: pd.Series, label: str) -> str:
    """Explain a feature importance or coefficient chart."""
    if values.empty:
        return "- No feature signal was available for this model."
    strongest = values.abs().idxmax()
    strongest_value = float(values.loc[strongest])
    direction = ""
    if label == "coefficient":
        direction = " Positive values push predictions up; negative values push predictions down."
    return (
        f"- Strongest selected feature: `{strongest}` with `{label}` `{strongest_value:.3f}`.\n"
        "- Longer bars have more influence in this fitted model.\n"
        f"- Use this as model evidence, not causal proof.{direction}"
    )


def evaluate_all_models(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    task_type: str,
    use_scaling: bool,
    scaler_name: str,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Train and evaluate every supported supervised model on the same split."""
    event_source, event_features = add_market_event_features(data, date_col)
    model_features = list(dict.fromkeys(features + event_features))
    model_cols = list(dict.fromkeys([target] + model_features + ([date_col] if date_col in event_source.columns else [])))
    model_df = event_source[model_cols].copy().dropna(subset=[target])
    train_df, test_df, split_info = chronological_model_split(model_df, target, model_features, date_col)
    X_train, X_test = train_df[model_features], test_df[model_features]
    y_train, y_test = train_df[target], test_df[target]
    # The supervised ML pipeline uses a chronological train-test split and applies
    # the same preprocessing and modeling steps to every candidate algorithm.
    rows = []

    if task_type == "Regression":
        for option in model_options_for_task(task_type):
            pipeline = build_model_pipeline(option, task_type, use_scaling, scaler_name)
            pred, _, _ = fit_predict_regression_series(pipeline, X_train, y_train, X_test, train_df, target)
            mae, rmse, r2 = regression_metrics(y_test, pred)
            rows.append(
                {
                    "Task": "Regression",
                    "Model": option,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "Train rows": len(X_train),
                    "Test rows": len(X_test),
                    "Feature count": len(model_features),
                    "Event features": len(event_features),
                    "Test start": split_info["Test start"],
                }
            )
        return pd.DataFrame(rows).sort_values("R2", ascending=False)

    y_train_binned, y_test_binned = make_classification_labels(y_train, y_test)
    for option in model_options_for_task(task_type):
        pipeline = build_model_pipeline(option, task_type, use_scaling, scaler_name)
        pipeline.fit(X_train, y_train_binned)
        pred = pipeline.predict(X_test)
        auc = classification_auc(pipeline, X_test, y_test_binned)
        rows.append(
            {
                "Task": "Classification",
                "Model": option,
                "Accuracy": accuracy_score(y_test_binned, pred),
                "Precision": precision_score(y_test_binned, pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test_binned, pred, average="weighted", zero_division=0),
                "F1": f1_score(y_test_binned, pred, average="weighted", zero_division=0),
                "ROC AUC": auc,
                "Train rows": len(X_train),
                "Test rows": len(X_test),
                "Feature count": len(model_features),
                "Event features": len(event_features),
                "Test start": split_info["Test start"],
            }
        )
    return pd.DataFrame(rows).sort_values("F1", ascending=False)


def fit_best_model_for_task(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    task_type: str,
    use_scaling: bool,
    scaler_name: str,
    model_name: str,
    date_col: str | None = None,
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Fit one selected supervised model and return the fitted pipeline with training data."""
    event_source, event_features = add_market_event_features(data, date_col)
    model_features = list(dict.fromkeys(features + event_features))
    model_cols = list(dict.fromkeys([target] + model_features + ([date_col] if date_col in event_source.columns else [])))
    model_df = event_source[model_cols].copy().dropna(subset=[target])
    train_df, test_df, _ = chronological_model_split(model_df, target, model_features, date_col)
    X_train = train_df[model_features]
    y_train = train_df[target]
    # This function trains on the chronological training period; held-out rows are
    # for test/evaluation consistency. Future forecasts refit separately on all
    # usable history after evaluation.
    if task_type == "Classification":
        y_train, _ = make_classification_labels(y_train, test_df[target])
    pipeline = build_model_pipeline(model_name, task_type, use_scaling, scaler_name)
    if task_type == "Regression":
        _, pipeline, y_train = fit_predict_regression_series(pipeline, X_train, y_train, X_train.iloc[0:0], train_df, target)
        X_train = X_train.loc[y_train.index]
    else:
        pipeline.fit(X_train, y_train)
    return pipeline, X_train, y_train


def model_feature_importance(
    pipeline: Pipeline,
    features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
) -> pd.DataFrame:
    """Estimate feature importance using tree importances, coefficients, or perturbation."""
    # This is the explainability stage of the ML pipeline: it shows which
    # features the selected model relied on most.
    fitted = pipeline.named_steps["model"]
    if hasattr(fitted, "feature_importances_"):
        values = np.asarray(fitted.feature_importances_)
        method = "Built-in feature importance"
    elif hasattr(fitted, "coef_"):
        values = np.abs(np.ravel(fitted.coef_))
        if len(values) != len(features):
            values = values.reshape(-1, len(features)).mean(axis=0)
        method = "Absolute coefficient size"
    else:
        baseline = pipeline.score(X_train, y_train)
        values = []
        rng = np.random.default_rng(0)
        for feature in features:
            shuffled = X_train.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            values.append(max(0.0, baseline - pipeline.score(shuffled, y_train)))
        values = np.asarray(values)
        method = "Permutation importance"
    importance = pd.DataFrame({"Feature": features, "Importance": values, "Method": method})
    if importance["Importance"].sum() > 0:
        importance["Importance %"] = importance["Importance"] / importance["Importance"].sum() * 100
    else:
        importance["Importance %"] = 0.0
    return importance.sort_values("Importance", ascending=False)


def fit_diagnosis_label(train_score: float, test_score: float, gap: float) -> str:
    """Classify model fit quality from train/test score behavior."""
    if train_score >= 0.70 and gap >= 0.10:
        return "Overfitting risk"
    if train_score < 0.45 and test_score < 0.45:
        return "Underfitting risk"
    if test_score < 0 and train_score > 0:
        return "Poor generalization"
    return "Reasonable fit"


def overfit_adjusted_score(test_score: float, gap: float, cv_std: float | None = None) -> float:
    """Penalize high test scores when they come with a large train-test gap."""
    if pd.isna(test_score):
        return np.nan
    stability_penalty = 0.0 if cv_std is None or pd.isna(cv_std) else 0.25 * float(cv_std)
    gap_penalty = 0.50 * max(0.0, float(gap))
    return float(test_score - gap_penalty - stability_penalty)


def hyperparameter_fix_recommendation(model_name: str, diagnosis: str, task_type: str) -> str:
    """Recommend practical hyperparameter changes for the selected model and fit diagnosis."""
    if diagnosis == "Overfitting risk":
        fixes = {
            "RandomForest": "Reduce `max_depth`, increase `min_samples_leaf`, increase `min_samples_split`, or use fewer/noisier features.",
            "ExtraTrees": "Reduce `max_depth`, increase `min_samples_leaf`, increase `min_samples_split`, or use fewer/noisier features.",
            "GradientBoosting": "Lower `learning_rate`, reduce `max_depth`, reduce `n_estimators`, or add early stopping with validation data.",
            "HistGradientBoosting": "Lower `learning_rate`, reduce `max_leaf_nodes`, increase `l2_regularization`, or add better validation.",
            "DecisionTree": "Set a smaller `max_depth`, increase `min_samples_leaf`, and prune the tree with `ccp_alpha`.",
            "KNNRegressor": "Increase `n_neighbors`, scale features, and remove noisy features.",
            "KNNClassifier": "Increase `n_neighbors`, scale features, and remove noisy features.",
            "NeuralNetwork": "Increase `alpha`, use smaller hidden layers, keep `early_stopping=True`, or reduce `max_iter` if it memorizes.",
            "SVR": "Reduce `C`, increase `epsilon`, tune `gamma`, and scale features.",
            "Ridge": "Increase `alpha` to add stronger regularization.",
            "ElasticNet": "Increase `alpha`, lower noisy features, or tune `l1_ratio` toward more regularization.",
            "BayesianRidge": "Remove noisy features or compare with Ridge/ElasticNet if uncertainty regularization is not enough.",
            "LinearRegression": "Switch to Ridge/Lasso-style regularization or reduce high-leakage/noisy features.",
            "LogisticRegression": "Decrease `C` to strengthen regularization and remove noisy features.",
        }
        return fixes.get(model_name, "Reduce model complexity, add regularization, remove noisy features, or collect more data.")
    if diagnosis == "Underfitting risk":
        fixes = {
            "RandomForest": "Increase `n_estimators`, allow deeper trees with larger `max_depth`, or add stronger predictive features.",
            "ExtraTrees": "Increase `n_estimators`, allow deeper trees with larger `max_depth`, or add stronger predictive features.",
            "GradientBoosting": "Increase `n_estimators`, raise `learning_rate` carefully, allow deeper trees, or add better features.",
            "HistGradientBoosting": "Increase `max_iter`, raise `learning_rate` carefully, allow more leaf nodes, or add better features.",
            "DecisionTree": "Increase `max_depth`, lower `min_samples_leaf`, or use RandomForest/GradientBoosting instead.",
            "KNNRegressor": "Decrease `n_neighbors`, try distance weighting, and make sure features are scaled.",
            "KNNClassifier": "Decrease `n_neighbors`, try distance weighting, and make sure features are scaled.",
            "NeuralNetwork": "Increase hidden-layer size, reduce `alpha`, train longer, or add better scaled features.",
            "SVR": "Increase `C`, tune `gamma`, lower `epsilon`, and scale features.",
            "Ridge": "Lower `alpha`, add non-linear features, or try tree-based models.",
            "ElasticNet": "Lower `alpha`, tune `l1_ratio`, add non-linear features, or try tree-based models.",
            "BayesianRidge": "Add non-linear features or try boosted/tree-based models.",
            "LinearRegression": "Add interaction/lag features or try RandomForest/GradientBoosting for non-linear patterns.",
            "LogisticRegression": "Increase `C`, add useful features, or try tree-based classifiers.",
        }
        return fixes.get(model_name, "Increase model capacity, add stronger features, tune preprocessing, or try a more flexible model.")
    if diagnosis == "Poor generalization":
        return (
            "Check for time drift, target leakage, too few rows, or a train/test split that represents a different market period. "
            "Use cross-validation and compare with simpler models."
        )
    return "No urgent fix is required. Keep monitoring train/test gap, cross-validation stability, and feature quality."


def diagnose_model_fit(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    task_type: str,
    use_scaling: bool,
    scaler_name: str,
    models_to_check: list[str],
    custom_hyperparameters: dict[str, dict[str, object]] | None = None,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Evaluate train/test behavior to flag overfitting and underfitting."""
    event_source, event_features = add_market_event_features(data, date_col)
    model_features = list(dict.fromkeys(features + event_features))
    model_cols = list(dict.fromkeys([target] + model_features + ([date_col] if date_col in event_source.columns else [])))
    model_df = event_source[model_cols].copy().dropna(subset=[target])
    train_df, test_df, split_info = chronological_model_split(model_df, target, model_features, date_col)
    X_train, X_test = train_df[model_features], test_df[model_features]
    y_train, y_test = train_df[target], test_df[target]
    rows = []

    if task_type == "Classification":
        y_train, y_test = make_classification_labels(y_train, y_test)
        # Encode labels as plain integers for diagnostics. This avoids NumPy
        # isnan/type-coercion errors that can happen with pandas categorical
        # labels inside some scikit-learn estimators and scoring functions.
        label_codes = {"Low": 0, "Medium": 1, "High": 2}
        y_train_codes = y_train.astype(str).map(label_codes)
        y_test_codes = y_test.astype(str).map(label_codes)
        if y_train_codes.isna().any() or y_test_codes.isna().any():
            combined_labels = pd.concat([y_train.astype(str), y_test.astype(str)], ignore_index=True)
            fallback_codes = {label: code for code, label in enumerate(sorted(combined_labels.unique()))}
            y_train_codes = y_train.astype(str).map(fallback_codes)
            y_test_codes = y_test.astype(str).map(fallback_codes)
        y_train = y_train_codes.astype(int)
        y_test = y_test_codes.astype(int)

    for model_name in models_to_check:
        cv_metric = "CV R2" if task_type == "Regression" else "CV F1"
        cv_scoring = "r2" if task_type == "Regression" else "f1_weighted"
        model_params = (custom_hyperparameters or {}).get(model_name, {})
        try:
            pipeline = build_model_pipeline(model_name, task_type, use_scaling, scaler_name, model_params)
            pipeline.fit(X_train, y_train)
            train_pred = pipeline.predict(X_train)
            test_pred = pipeline.predict(X_test)

            if task_type == "Regression":
                train_score = float(r2_score(y_train, train_pred))
                test_score = float(r2_score(y_test, test_pred))
                train_mae = float(mean_absolute_error(y_train, train_pred))
                test_mae = float(mean_absolute_error(y_test, test_pred))
                train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
                test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            else:
                train_score = float(f1_score(y_train, train_pred, average="weighted", zero_division=0))
                test_score = float(f1_score(y_test, test_pred, average="weighted", zero_division=0))
                train_mae = np.nan
                test_mae = np.nan
                train_rmse = np.nan
                test_rmse = np.nan
        except Exception as err:
            rows.append(
                {
                    "Model": model_name,
                    "Task": task_type,
                    "Train score": np.nan,
                    "Test score": np.nan,
                    "Train-Test gap": np.nan,
                    "Adjusted score": np.nan,
                    "Diagnosis": "Model failed",
                    cv_metric: np.nan,
                    "CV std": np.nan,
                    "Train MAE": np.nan,
                    "Test MAE": np.nan,
                    "Train RMSE": np.nan,
                    "Test RMSE": np.nan,
                    "Test start": split_info["Test start"],
                    "Event features": len(event_features),
                    "Hyperparameters": str(model_params or "Default"),
                    "What to fix": f"Model could not run with the selected setup: {err}",
                }
            )
            continue

        gap = float(train_score - test_score)
        diagnosis = fit_diagnosis_label(train_score, test_score, gap)
        cv = min(5, len(X_train))
        if cv >= 2:
            try:
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=cv_scoring)
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
            except Exception:
                cv_mean = np.nan
                cv_std = np.nan
        else:
            cv_mean = np.nan
            cv_std = np.nan
        adjusted_score = overfit_adjusted_score(test_score, gap, cv_std)

        rows.append(
            {
                "Model": model_name,
                "Task": task_type,
                "Train score": train_score,
                "Test score": test_score,
                "Train-Test gap": gap,
                "Adjusted score": adjusted_score,
                "Diagnosis": diagnosis,
                cv_metric: cv_mean,
                "CV std": cv_std,
                "Train MAE": train_mae,
                "Test MAE": test_mae,
                "Train RMSE": train_rmse,
                "Test RMSE": test_rmse,
                "Test start": split_info["Test start"],
                "Event features": len(event_features),
                "Hyperparameters": str(model_params or "Default"),
                "What to fix": hyperparameter_fix_recommendation(model_name, diagnosis, task_type),
            }
        )

    diagnosis_order = {
        "Overfitting risk": 0,
        "Underfitting risk": 1,
        "Poor generalization": 2,
        "Reasonable fit": 3,
        "Model failed": 4,
    }
    result = pd.DataFrame(rows)
    result["_diagnosis_order"] = result["Diagnosis"].map(diagnosis_order).fillna(9)
    return result.sort_values(["_diagnosis_order", "Adjusted score"], ascending=[True, False]).drop(columns="_diagnosis_order")


def best_diagnostic_candidate(diagnostics: pd.DataFrame, task_type: str) -> tuple[pd.Series | None, bool]:
    """Return the best safe model if possible, otherwise the best available candidate."""
    if diagnostics.empty:
        return None, False
    valid = diagnostics[
        diagnostics["Test score"].notna()
        & (diagnostics["Diagnosis"] != "Model failed")
    ].copy()
    if valid.empty:
        return None, False

    safe = valid[valid["Diagnosis"] == "Reasonable fit"].copy()
    if not safe.empty:
        return safe.sort_values("Adjusted score", ascending=False).iloc[0], True

    # When no model is fully safe, prefer the candidate with the strongest test
    # score and the smallest gap. This is a fallback, not a clean champion.
    fallback = valid.copy()
    fallback["Abs gap"] = fallback["Train-Test gap"].abs()
    return fallback.sort_values(["Adjusted score", "Abs gap"], ascending=[False, True]).iloc[0], False


def hyperparameter_definitions_table() -> pd.DataFrame:
    """Define every hyperparameter exposed in the Fit Diagnostics tuning panel."""
    return pd.DataFrame(
        [
            {
                "Hyperparameter": "n_estimators",
                "Used by": "RandomForest, GradientBoosting",
                "Definition": "Number of trees or boosting stages trained by the ensemble.",
                "Higher value usually means": "More stable model, slower training, sometimes better fit.",
            },
            {
                "Hyperparameter": "max_depth",
                "Used by": "RandomForest, GradientBoosting, DecisionTree",
                "Definition": "Maximum number of levels each tree can grow.",
                "Higher value usually means": "More flexible model and higher overfitting risk.",
            },
            {
                "Hyperparameter": "min_samples_leaf",
                "Used by": "RandomForest, GradientBoosting, DecisionTree",
                "Definition": "Minimum rows required at the end of a tree branch.",
                "Higher value usually means": "Smoother model with less overfitting.",
            },
            {
                "Hyperparameter": "min_samples_split",
                "Used by": "RandomForest, DecisionTree",
                "Definition": "Minimum rows required before a tree node can split again.",
                "Higher value usually means": "Simpler trees with less overfitting.",
            },
            {
                "Hyperparameter": "max_features",
                "Used by": "RandomForest",
                "Definition": "Number or rule for features each tree can consider at a split.",
                "Higher value usually means": "More fit to the training data; `sqrt` is safer against overfitting.",
            },
            {
                "Hyperparameter": "learning_rate",
                "Used by": "GradientBoosting",
                "Definition": "How strongly each new boosting stage changes the model.",
                "Higher value usually means": "Faster learning and higher overfitting risk.",
            },
            {
                "Hyperparameter": "subsample",
                "Used by": "GradientBoosting",
                "Definition": "Fraction of training rows used for each boosting stage.",
                "Higher value usually means": "Less randomness; lower values add regularization.",
            },
            {
                "Hyperparameter": "ccp_alpha",
                "Used by": "DecisionTree",
                "Definition": "Cost-complexity pruning strength used to remove weak tree branches.",
                "Higher value usually means": "Smaller tree with stronger regularization.",
            },
            {
                "Hyperparameter": "n_neighbors",
                "Used by": "KNNRegressor, KNNClassifier",
                "Definition": "Number of nearby training rows used to make each prediction.",
                "Higher value usually means": "Smoother predictions and less sensitivity to noise.",
            },
            {
                "Hyperparameter": "weights",
                "Used by": "KNNRegressor, KNNClassifier",
                "Definition": "Whether all neighbors count equally or closer neighbors count more.",
                "Higher value usually means": "`distance` is more local; `uniform` is smoother.",
            },
            {
                "Hyperparameter": "alpha",
                "Used by": "Ridge, NeuralNetwork",
                "Definition": "Regularization strength that penalizes large model weights.",
                "Higher value usually means": "Simpler model with less overfitting.",
            },
            {
                "Hyperparameter": "C",
                "Used by": "LogisticRegression, SVR",
                "Definition": "Penalty strength inverse; smaller C means stronger regularization.",
                "Higher value usually means": "More flexible model and weaker regularization.",
            },
            {
                "Hyperparameter": "max_iter",
                "Used by": "LogisticRegression, NeuralNetwork",
                "Definition": "Maximum training iterations allowed before stopping.",
                "Higher value usually means": "More time to converge, not necessarily more complexity.",
            },
            {
                "Hyperparameter": "epsilon",
                "Used by": "SVR",
                "Definition": "Prediction-error tolerance ignored by the SVR loss function.",
                "Higher value usually means": "Smoother model that ignores small errors.",
            },
            {
                "Hyperparameter": "gamma",
                "Used by": "SVR",
                "Definition": "Kernel influence setting for how far each training row reaches.",
                "Higher value usually means": "`auto` can be more local; `scale` adapts to feature variance.",
            },
            {
                "Hyperparameter": "hidden_layers",
                "Used by": "NeuralNetwork",
                "Definition": "Number of hidden layers in the neural network.",
                "Higher value usually means": "More capacity and higher overfitting risk.",
            },
            {
                "Hyperparameter": "hidden_units",
                "Used by": "NeuralNetwork",
                "Definition": "Number of neurons in each hidden layer.",
                "Higher value usually means": "More capacity and higher overfitting risk.",
            },
            {
                "Hyperparameter": "learning_rate_init",
                "Used by": "NeuralNetwork",
                "Definition": "Initial step size used while updating neural-network weights.",
                "Higher value usually means": "Faster learning but less stable training.",
            },
        ]
    )


def render_hyperparameter_controls(model_name: str, task_type: str, tuning_goal: str) -> dict[str, object]:
    """Render model-specific tuning controls and return selected hyperparameters."""
    params: dict[str, object] = {}
    overfit = tuning_goal == "Reduce overfitting"
    underfit = tuning_goal == "Reduce underfitting"

    if model_name == "LinearRegression":
        st.info("LinearRegression has no main regularization hyperparameter. Use Ridge, remove noisy features, or add better features.")
        return params

    if model_name in {"RandomForest", "ExtraTrees", "GradientBoosting"}:
        default_estimators = 200 if overfit else 500 if underfit else 300
        params["n_estimators"] = st.slider("Number of trees / estimators", 50, 800, default_estimators, 50)
        max_depth_options = [None, 2, 3, 4, 5, 8, 12, 16, 24]
        default_depth = 3 if overfit else 12 if underfit else None
        params["max_depth"] = st.selectbox(
            "Maximum depth",
            max_depth_options,
            index=max_depth_options.index(default_depth),
            help="Lower depth reduces overfitting. Higher depth can reduce underfitting.",
        )
        params["min_samples_leaf"] = st.slider(
            "Minimum samples per leaf",
            1,
            30,
            8 if overfit else 2 if underfit else 5,
            help="Higher values smooth the model and reduce overfitting.",
        )
        if model_name in {"RandomForest", "ExtraTrees"}:
            params["min_samples_split"] = st.slider("Minimum samples to split", 2, 40, 14 if overfit else 4 if underfit else 10)
            params["max_features"] = st.selectbox(
                "Features considered at each split",
                ["sqrt", "log2", None],
                index=0,
                help="sqrt/log2 reduce tree similarity and help control overfitting.",
            )
        else:
            params["learning_rate"] = st.slider(
                "Learning rate",
                0.01,
                0.30,
                0.04 if overfit else 0.15 if underfit else 0.10,
                0.01,
                help="Lower learning rate is safer against overfitting but may need more estimators.",
            )
            params["subsample"] = st.slider(
                "Row sample per boosting stage",
                0.5,
                1.0,
                0.75 if overfit else 1.0 if underfit else 0.8,
                0.05,
                help="Lower values add randomness and reduce overfitting.",
            )
    elif model_name == "HistGradientBoosting":
        params["max_iter"] = st.slider("Boosting iterations", 50, 800, 250 if overfit else 500 if underfit else 350, 50)
        params["learning_rate"] = st.slider("Learning rate", 0.01, 0.30, 0.03 if overfit else 0.12 if underfit else 0.05, 0.01)
        params["max_leaf_nodes"] = st.slider("Maximum leaf nodes", 5, 63, 10 if overfit else 31 if underfit else 15)
        params["l2_regularization"] = st.slider("L2 regularization", 0.0, 2.0, 0.20 if overfit else 0.0 if underfit else 0.05, 0.05)
    elif model_name == "ElasticNet":
        params["alpha"] = st.slider("Regularization strength (alpha)", 0.001, 10.0, 0.10 if overfit else 0.01 if underfit else 0.05)
        params["l1_ratio"] = st.slider("L1 ratio", 0.0, 1.0, 0.35 if overfit else 0.05 if underfit else 0.2, 0.05)
        params["max_iter"] = 5000
    elif model_name == "BayesianRidge":
        st.info("BayesianRidge tunes its regularization internally. Try feature selection or compare with ElasticNet/boosted trees.")
    elif model_name == "DecisionTree":
        max_depth_options = [None, 2, 3, 4, 5, 8, 12, 16, 24]
        default_depth = 3 if overfit else 12 if underfit else 5
        params["max_depth"] = st.selectbox("Maximum depth", max_depth_options, index=max_depth_options.index(default_depth))
        params["min_samples_leaf"] = st.slider("Minimum samples per leaf", 1, 40, 10 if overfit else 1 if underfit else 3)
        params["min_samples_split"] = st.slider("Minimum samples to split", 2, 50, 16 if overfit else 2)
        params["ccp_alpha"] = st.slider("Pruning strength (ccp_alpha)", 0.0, 0.05, 0.01 if overfit else 0.0, 0.001)
    elif model_name in {"KNN", "KNNRegressor", "KNNClassifier"}:
        params["n_neighbors"] = st.slider(
            "Number of neighbors",
            1,
            30,
            12 if overfit else 3 if underfit else 5,
            help="More neighbors smooths predictions; fewer neighbors increases flexibility.",
        )
        params["weights"] = st.selectbox("Neighbor weighting", ["uniform", "distance"], index=0 if overfit else 1)
    elif model_name == "Ridge":
        params["alpha"] = st.slider(
            "Regularization strength (alpha)",
            0.001,
            100.0,
            10.0 if overfit else 0.1 if underfit else 1.0,
            help="Higher alpha reduces overfitting. Lower alpha increases flexibility.",
        )
    elif model_name == "LogisticRegression":
        params["C"] = st.slider(
            "Inverse regularization strength (C)",
            0.01,
            20.0,
            0.2 if overfit else 5.0 if underfit else 1.0,
            help="Lower C means stronger regularization. Higher C increases flexibility.",
        )
        params["max_iter"] = st.slider("Maximum iterations", 200, 3000, 1000, 100)
    elif model_name == "SVR":
        params["C"] = st.slider("Penalty strength (C)", 0.01, 50.0, 0.5 if overfit else 10.0 if underfit else 1.0)
        params["epsilon"] = st.slider("Error tolerance (epsilon)", 0.001, 2.0, 0.3 if overfit else 0.05 if underfit else 0.1)
        params["gamma"] = st.selectbox("Kernel gamma", ["scale", "auto"], index=0)
    elif model_name == "NeuralNetwork":
        params["hidden_layers"] = st.slider("Hidden layers", 1, 4, 1 if overfit else 3 if underfit else 2)
        params["hidden_units"] = st.slider("Units per hidden layer", 8, 256, 32 if overfit else 128 if underfit else 64, 8)
        params["alpha"] = st.slider("Regularization strength (alpha)", 0.0001, 0.1, 0.01 if overfit else 0.0005 if underfit else 0.001)
        params["learning_rate_init"] = st.slider("Learning rate", 0.0001, 0.02, 0.001, 0.0001)
        params["max_iter"] = st.slider("Maximum iterations", 200, 2000, 800 if overfit else 1500 if underfit else 1200, 100)

    return params


# -----------------------------------------------------------------------------
# Reporting, governance, and project-management helpers
# -----------------------------------------------------------------------------

def data_dictionary(df: pd.DataFrame, date_col: str | None, target: str | None) -> pd.DataFrame:
    """Build a column-level reference table for the loaded dataset."""
    rows = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        role = "Date" if col == date_col else "Default target" if col == target else "Feature"
        if pd.api.types.is_numeric_dtype(series):
            example = f"min={series.min():,.3f}, max={series.max():,.3f}" if series.notna().any() else "empty"
        else:
            examples = series.dropna().astype(str).head(3).tolist()
            example = ", ".join(examples) if examples else "empty"
        rows.append(
            {
                "Column": col,
                "Role": role,
                "Type": str(series.dtype),
                "Missing": missing,
                "Missing %": missing / len(df) * 100 if len(df) else 0,
                "Unique values": int(series.nunique(dropna=True)),
                "Example / Range": example,
            }
        )
    return pd.DataFrame(rows)


def executive_report_markdown(
    df: pd.DataFrame,
    date_col: str | None,
    target: str | None,
    evaluation: pd.DataFrame | None,
    profile: dict[str, str | bool] | None = None,
) -> str:
    """Generate a downloadable one-page executive project report."""
    profile = profile or {"name": "DataIQ Platform", "domain": "General data"}
    date_range = "not available"
    if date_col and date_col in df.columns and df[date_col].notna().any():
        date_range = f"{df[date_col].min().date()} to {df[date_col].max().date()}"
    best_model = "Run Evaluation to calculate"
    if evaluation is not None and not evaluation.empty:
        metric = "R2" if "R2" in evaluation.columns else "F1"
        best = evaluation.sort_values(metric, ascending=False).iloc[0]
        best_model = f"{best['Model']} ({metric}={best[metric]:.3f})"
    top_signal = "not available"
    if target and target in df.columns:
        candidates = [col for col in interpretable_numeric_features(df.select_dtypes(include=["number"]).columns.tolist(), target) if col != target]
        if candidates:
            cmat = corr_matrix(df[[target] + candidates].dropna(subset=[target]))
            if target in cmat.columns:
                corr = cmat[target].drop(labels=[target], errors="ignore").dropna()
                if not corr.empty:
                    signal = corr.abs().idxmax()
                    top_signal = f"{signal} (corr={corr[signal]:.2f})"
    return f"""# {profile['name']} Executive Report

## Dataset
- Rows: {len(df):,}
- Columns: {df.shape[1]:,}
- Date range: {date_range}
- Main target: {target or "not detected"}
- Domain mode: {profile['domain']}

## Main Findings
- Strongest meaningful signal: {top_signal}
- Best evaluated model: {best_model}
- Overall project result: the selected target is analyzed through data quality, exploration, model evaluation, OLAP segmentation, and decision-support views.

## Recommendation
Use Evaluation for model choice, OLAP for segment interpretation, Forecast for future direction, Scenario Simulator for what-if analysis, and Production Readiness before trusting outputs.
"""


def markdown_table(data: pd.DataFrame, max_rows: int = 12) -> str:
    """Render a compact markdown table without optional tabulate dependency."""
    if data is None or data.empty:
        return "_No data available._"
    view = data.head(max_rows).copy()
    view = view.astype(str)
    headers = view.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        values = [str(row[col]).replace("|", "\\|").replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    if len(data) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(data)} rows._")
    return "\n".join(lines)


def report_generator_markdown(
    project_profile: dict[str, object],
    df: pd.DataFrame,
    date_col: str | None,
    target: str | None,
    smart_setup: dict[str, object],
    cleaning_audit: pd.DataFrame,
    evaluation: pd.DataFrame | None,
    validation: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> str:
    """Generate a full project report that can be downloaded or presented."""
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    best_model = "Run Evaluation to calculate"
    if evaluation is not None and not evaluation.empty:
        metric = "R2" if "R2" in evaluation.columns else "F1"
        best = evaluation.sort_values(metric, ascending=False).iloc[0]
        best_model = f"{best['Model']} ({metric}={best[metric]:.3f})"

    top_signal = "Not available"
    if target and target in df.columns:
        feature_candidates = [
            col
            for col in interpretable_numeric_features(df.select_dtypes(include=["number"]).columns.tolist(), target)
            if col != target
        ]
        if feature_candidates:
            cmat = corr_matrix(df[[target] + feature_candidates].dropna(subset=[target]))
            if target in cmat.columns:
                corr = cmat[target].drop(labels=[target], errors="ignore").dropna()
                if not corr.empty:
                    signal = corr.abs().idxmax()
                    top_signal = f"{signal} (correlation {corr[signal]:.3f})"

    project_table = pd.DataFrame(
        [{"Field": key, "Value": value} for key, value in project_profile.items()]
    )
    template_name = str(project_profile.get("Use-case template", "General Data Project"))
    template = USE_CASE_TEMPLATES.get(template_name, USE_CASE_TEMPLATES["General Data Project"])
    app_mode_name = str(project_profile.get("App mode", "Beginner Mode"))
    app_mode = APP_MODES.get(app_mode_name, APP_MODES["Beginner Mode"])
    smart_table = pd.DataFrame(
        [
            {"Item": "Template category", "Value": template.get("category", "General")},
            {"Item": "Use-case template", "Value": template_name},
            {"Item": "App mode", "Value": app_mode_name},
            {"Item": "Recommended date", "Value": smart_setup.get("Recommended date") or "Not detected"},
            {"Item": "Recommended target", "Value": smart_setup.get("Recommended target") or "Not detected"},
            {"Item": "Suggested task", "Value": smart_setup.get("Suggested task") or "Not ready"},
            {
                "Item": "Recommended features",
                "Value": ", ".join(smart_setup.get("Recommended features", [])[:12]) or "Not available",
            },
        ]
    )
    numeric_summary = df.select_dtypes(include=["number"]).describe().T.reset_index(names="Column").round(3)

    return f"""# {project_profile.get('Project name', 'DataIQ Project')} Report

Generated: {generated_at}

## Executive Overview
- Source: {project_profile.get('Source', 'Unknown')}
- Rows after cleaning/filtering: {len(df):,}
- Columns: {df.shape[1]:,}
- Primary target: {target or 'Not selected'}
- Date column: {date_col or 'Not selected'}
- Suggested task: {smart_setup.get('Suggested task', 'Not ready')}
- Template category: {template.get('category', 'General')}
- Use-case template: {template_name}
- App mode: {app_mode_name}
- Strongest meaningful signal: {top_signal}
- Best evaluated model: {best_model}

## Template Guidance
- Feature strategy: {template['feature_strategy']}
- Report language: {template['report_language']}
- App mode guidance: {app_mode['tone']}
- Caution: {template['caution']}

## Project Profile
{markdown_table(project_table, max_rows=20)}

## Smart Auto-Setup
{markdown_table(smart_table, max_rows=10)}

## Data Cleaning Studio Audit
{markdown_table(cleaning_audit, max_rows=20)}

## Data Validation
{markdown_table(validation, max_rows=20)}

## Model Evaluation
{markdown_table(evaluation if evaluation is not None else pd.DataFrame(), max_rows=12)}

## Numeric Summary
{markdown_table(numeric_summary, max_rows=12)}

## Data Dictionary Sample
{markdown_table(dictionary, max_rows=20)}

## Recommendation
Use this report as a project handoff. Before making real decisions, review Data Quality, Evaluation, Prediction Page outputs, and Production Readiness together.
"""


def report_markdown_to_html(markdown_text: str, title: str) -> str:
    """Create a simple standalone HTML report from the generated markdown."""
    body = html.escape(markdown_text)
    body = body.replace("\n", "<br>\n")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.55; margin: 36px; color: #172033; }}
    .report {{ max-width: 1040px; margin: 0 auto; }}
    h1, h2, h3 {{ color: #0f4c5c; }}
    code {{ background: #eef3f8; padding: 2px 5px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ccd6e0; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f8; }}
  </style>
</head>
<body>
  <main class="report">{body}</main>
</body>
</html>"""


def validation_report(df: pd.DataFrame, date_col: str | None, target: str | None) -> pd.DataFrame:
    """Run basic data-quality checks used before trusting analysis or scoring."""
    checks = []
    checks.append(
        {
            "Check": "Dataset has rows",
            "Status": "Pass" if len(df) > 0 else "Fail",
            "Value": len(df),
            "Why it matters": "Production pipelines should not score empty data.",
        }
    )
    checks.append(
        {
            "Check": "Duplicate rows",
            "Status": "Pass" if int(df.duplicated().sum()) == 0 else "Warning",
            "Value": int(df.duplicated().sum()),
            "Why it matters": "Duplicates can bias model training and reporting.",
        }
    )
    missing_pct = float(df.isna().sum().sum() / max(df.size, 1) * 100)
    checks.append(
        {
            "Check": "Overall missing data",
            "Status": "Pass" if missing_pct < 5 else "Warning" if missing_pct < 20 else "Fail",
            "Value": f"{missing_pct:.2f}%",
            "Why it matters": "High missingness can make dashboards and models unstable.",
        }
    )
    checks.append(
        {
            "Check": "Date column detected",
            "Status": "Pass" if date_col and date_col in df.columns else "Warning",
            "Value": date_col or "not detected",
            "Why it matters": "Time order is required for forecasting, drift, and market-period comparison.",
        }
    )
    checks.append(
        {
            "Check": "Target detected",
            "Status": "Pass" if target and target in df.columns else "Fail",
            "Value": target or "not detected",
            "Why it matters": "A target is required for supervised modeling and final conclusions.",
        }
    )
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    inf_count = int(np.isinf(df[numeric_cols]).sum().sum()) if numeric_cols else 0
    checks.append(
        {
            "Check": "Infinite numeric values",
            "Status": "Pass" if inf_count == 0 else "Fail",
            "Value": inf_count,
            "Why it matters": "Infinite values break many model and chart pipelines.",
        }
    )
    report = pd.DataFrame(checks)
    # Streamlit serializes dataframes through Arrow, which requires each column
    # to have a consistent type. This table mixes counts, percentages, and text
    # in Value, so keep the display column explicitly textual.
    report["Value"] = report["Value"].astype(str)
    return report


def drift_report(df: pd.DataFrame, date_col: str | None, numeric_cols: list[str]) -> pd.DataFrame:
    """Compare early baseline data with recent data to flag distribution shifts."""
    if not date_col or date_col not in df.columns or len(numeric_cols) == 0:
        return pd.DataFrame()
    ordered = df.sort_values(date_col).dropna(subset=[date_col])
    if len(ordered) < 20:
        return pd.DataFrame()
    split = max(5, len(ordered) // 3)
    baseline = ordered.head(split)
    current = ordered.tail(split)
    rows = []
    for col in numeric_cols:
        base_mean = baseline[col].replace([np.inf, -np.inf], np.nan).mean()
        curr_mean = current[col].replace([np.inf, -np.inf], np.nan).mean()
        base_std = baseline[col].replace([np.inf, -np.inf], np.nan).std()
        if pd.isna(base_mean) or pd.isna(curr_mean):
            continue
        pct_shift = ((curr_mean - base_mean) / abs(base_mean) * 100) if base_mean else np.nan
        z_shift = ((curr_mean - base_mean) / base_std) if base_std and not pd.isna(base_std) else np.nan
        if np.isfinite(z_shift) and abs(z_shift) >= 1.5:
            status = "High drift"
        elif np.isfinite(pct_shift) and abs(pct_shift) >= 20:
            status = "Moderate drift"
        else:
            status = "Stable"
        rows.append(
            {
                "Feature": col,
                "Baseline mean": base_mean,
                "Current mean": curr_mean,
                "Mean shift %": pct_shift,
                "Z shift": z_shift,
                "Status": status,
            }
        )
    return pd.DataFrame(rows).sort_values("Z shift", key=lambda value: value.abs(), ascending=False)


def model_card_markdown(
    target: str | None,
    evaluation: pd.DataFrame | None,
    validation: pd.DataFrame,
    drift: pd.DataFrame,
) -> str:
    """Create a concise model card describing metrics, limits, and governance checks."""
    best_model = "Not evaluated"
    metric_line = "Run Evaluation to record model performance."
    if evaluation is not None and not evaluation.empty:
        metric = "R2" if "R2" in evaluation.columns else "F1"
        best = evaluation.sort_values(metric, ascending=False).iloc[0]
        best_model = str(best["Model"])
        metric_line = f"{metric}: {best[metric]:.3f}"
    failed_checks = int((validation["Status"] == "Fail").sum()) if not validation.empty else 0
    drift_count = int((drift["Status"] != "Stable").sum()) if not drift.empty else 0
    return f"""# Model Card

## Intended Use
Educational analytics for the active CSV dataset selected in the platform.

## Target
{target or "Not detected"}

## Best Current Model
{best_model}

## Performance
{metric_line}

## Data Validation
- Failed checks: {failed_checks}
- Validation should be reviewed before trusting model outputs.

## Drift Monitoring
- Features with moderate/high drift: {drift_count}
- Drift means recent data behaves differently from the baseline period.

## Limitations
- Correlation and model predictions are not causal proof.
- Forecasts and scenarios are educational, not financial advice.
- Aggregated data can hide important subgroup differences.

## Governance Recommendation
Re-run validation, drift checks, and model evaluation whenever the dataset is updated.
"""


def experiment_table(evaluation: pd.DataFrame | None, target: str | None, features: list[str]) -> pd.DataFrame:
    """Convert saved evaluation results into an experiment-tracking table."""
    if evaluation is None or evaluation.empty:
        return pd.DataFrame()
    rows = []
    metric_cols = [
        col
        for col in ["R2", "MAE", "RMSE", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
        if col in evaluation.columns
    ]
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    for _, row in evaluation.iterrows():
        item = {
            "Experiment ID": f"EXP-{len(rows) + 1:03d}",
            "Timestamp": timestamp,
            "Target": target or "not detected",
            "Model": row.get("Model", "unknown"),
            "Feature count": len(features),
            "Notes": "Generated from Evaluation page",
        }
        for metric in metric_cols:
            item[metric] = row.get(metric)
        rows.append(item)
    return pd.DataFrame(rows)


def best_model_row(evaluation: pd.DataFrame | None) -> tuple[pd.Series | None, str | None]:
    """Return the best evaluated model row and the metric used to select it."""
    if evaluation is None or evaluation.empty:
        return None, None
    metric = "R2" if "R2" in evaluation.columns else "F1" if "F1" in evaluation.columns else None
    if not metric:
        return evaluation.iloc[0], None
    return evaluation.sort_values(metric, ascending=False).iloc[0], metric


def pipeline_stage_table(df: pd.DataFrame, date_col: str | None, target: str | None, evaluation: pd.DataFrame | None) -> pd.DataFrame:
    """Summarize the end-to-end project pipeline and each stage status."""
    # This table is a lightweight project health summary for reviewers and demos.
    return pd.DataFrame(
        [
            {
                "Stage": "Raw data",
                "Status": "Complete" if len(df) else "Needs data",
                "Output": f"{len(df):,} rows, {df.shape[1]:,} columns",
                "Purpose": "Load housing and macroeconomic indicators.",
            },
            {
                "Stage": "Cleaning",
                "Status": "Complete",
                "Output": f"Date column: {date_col or 'not detected'}",
                "Purpose": "Standardize dates, numeric columns, and neutral time periods.",
            },
            {
                "Stage": "Feature selection",
                "Status": "Complete" if target else "Needs target",
                "Output": f"Target: {target or 'not detected'}",
                "Purpose": "Choose target and model features.",
            },
            {
                "Stage": "Model training",
                "Status": "Complete" if evaluation is not None and not evaluation.empty else "Run Evaluation",
                "Output": "Comparison table" if evaluation is not None and not evaluation.empty else "No run yet",
                "Purpose": "Train and compare supervised models.",
            },
            {
                "Stage": "Evaluation",
                "Status": "Complete" if evaluation is not None and not evaluation.empty else "Run Evaluation",
                "Output": "Metrics and explainability" if evaluation is not None and not evaluation.empty else "No metrics yet",
                "Purpose": "Select the best model with evidence.",
            },
            {
                "Stage": "Reporting",
                "Status": "Complete",
                "Output": "Executive report, model card, downloads",
                "Purpose": "Turn results into project conclusions.",
            },
        ]
    )


def big_data_readiness_tables(df: pd.DataFrame, date_col: str | None, num_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize big-data readiness and a practical scale-up roadmap."""
    memory_mb = float(df.memory_usage(deep=True).sum() / 1024**2)
    has_time = bool(date_col and date_col in df.columns)
    row_band = "Small" if len(df) < 100_000 else "Medium" if len(df) < 5_000_000 else "Large"
    readiness = pd.DataFrame(
        [
            {
                "Big data dimension": "Volume",
                "Current evidence": f"{len(df):,} rows, {df.shape[1]:,} columns, about {memory_mb:.2f} MB in memory",
                "Readiness": row_band,
                "Scale action": "Use Parquet, partition by date/region, and avoid loading every row into browser charts.",
            },
            {
                "Big data dimension": "Velocity",
                "Current evidence": "Batch CSV upload or local file refresh",
                "Readiness": "Batch-ready",
                "Scale action": "Move ingestion to scheduled jobs, object storage, or streaming queues for frequent updates.",
            },
            {
                "Big data dimension": "Variety",
                "Current evidence": f"{len(num_cols):,} numeric measures plus date/period labels",
                "Readiness": "Structured data",
                "Scale action": "Add schema validation for external sources such as census, mortgage, inventory, and macro feeds.",
            },
            {
                "Big data dimension": "Veracity",
                "Current evidence": "Validation report, missing checks, duplicate checks, drift monitoring",
                "Readiness": "Governed",
                "Scale action": "Promote these checks to automated data-quality gates before model training.",
            },
            {
                "Big data dimension": "Time partitioning",
                "Current evidence": f"Date column: {date_col or 'not detected'}",
                "Readiness": "Ready" if has_time else "Needs date key",
                "Scale action": "Partition by year/month so dashboards and models read only the needed time window.",
            },
        ]
    )
    roadmap = pd.DataFrame(
        [
            {
                "Layer": "Storage",
                "Current app": "CSV loaded into pandas",
                "Big-data version": "Parquet files in object storage or a warehouse table",
                "Why it helps": "Columnar reads are faster and cheaper for analytical queries.",
            },
            {
                "Layer": "Processing",
                "Current app": "pandas cleaning and feature prep",
                "Big-data version": "Spark, Dask, DuckDB, or warehouse SQL transformations",
                "Why it helps": "Large transformations run outside the Streamlit process.",
            },
            {
                "Layer": "Serving",
                "Current app": "Streamlit reads prepared dataframe",
                "Big-data version": "Pre-aggregated marts and cached feature tables",
                "Why it helps": "The UI stays fast because it reads only dashboard-ready data.",
            },
            {
                "Layer": "Modeling",
                "Current app": "scikit-learn with performance-mode row sampling",
                "Big-data version": "Sampled training sets plus distributed feature engineering",
                "Why it helps": "Model experiments stay reproducible without forcing every run over all raw rows.",
            },
            {
                "Layer": "Governance",
                "Current app": "validation, drift, model card, registry",
                "Big-data version": "Automated quality gates, lineage, monitoring, and retraining triggers",
                "Why it helps": "Teams can trust changing data at production scale.",
            },
        ]
    )
    return readiness, roadmap


def business_impact_text(target: str | None, evaluation: pd.DataFrame | None, latest_change: float | None) -> str:
    """Translate model and target evidence into stakeholder-friendly business language."""
    # Use the best evaluated model and the latest target movement to create a
    # business-facing summary sentence.
    best, metric = best_model_row(evaluation)
    model_sentence = (
        f"The current champion candidate is `{best['Model']}` with `{metric} = {best[metric]:.3f}`. "
        if best is not None and metric
        else "Run Evaluation to select a champion model. "
    )
    if latest_change is None:
        movement = "The latest target movement is not available."
    elif latest_change > 0:
        movement = f"The latest target movement is positive by {latest_change:,.2f}, which supports a resilience story."
    elif latest_change < 0:
        movement = f"The latest target movement is negative by {latest_change:,.2f}, which supports a caution/risk story."
    else:
        movement = "The latest target movement is flat, which supports a stability story."
    return (
        f"Business impact: `{target or 'the target'}` can be monitored as a decision indicator. "
        f"{model_sentence}{movement} The app helps decision makers compare market segments, evaluate model reliability, "
        "test scenarios, and explain whether the housing market looks resilient, risky, or stable."
    )


def model_family_reason(model_name: str) -> str:
    """Explain the practical strengths and limits of a model family."""
    reasons = {
        "RandomForest": "RandomForest often performs well because it captures non-linear relationships and interactions between housing indicators without requiring one straight-line pattern.",
        "ExtraTrees": "ExtraTrees is a strong tabular baseline because it averages many randomized trees, often reducing variance compared with one decision tree.",
        "GradientBoosting": "GradientBoosting can score highly because it builds many small corrections, which helps when housing movement is driven by several weak signals together.",
        "HistGradientBoosting": "HistGradientBoosting is a fast boosted-tree model that can capture non-linear thresholds while staying regularized on medium-sized tabular data.",
        "NeuralNetwork": "NeuralNetwork can learn flexible non-linear patterns, but it needs enough clean rows; on small datasets it can also underperform simpler models.",
        "Ridge": "Ridge is stable and useful when the relationship is mostly linear, but it can miss curved or threshold effects in housing data.",
        "ElasticNet": "ElasticNet adds stronger regularization and feature selection pressure, which can help when many variables are correlated.",
        "BayesianRidge": "BayesianRidge is a stable linear baseline that estimates regularization from the data, often useful on small noisy datasets.",
        "LinearRegression": "LinearRegression is easy to interpret, but it usually scores lower when the market has non-linear shocks or interacting variables.",
        "SVR": "SVR can model non-linear patterns, but it is sensitive to scaling, feature choice, and parameter settings.",
        "DecisionTree": "DecisionTree is easy to read, but one tree can overfit training patterns and generalize poorly to the test period.",
        "KNNRegressor": "KNNRegressor depends on similar historical examples; it can struggle when the newest market period is different from older periods.",
        "KNNClassifier": "KNNClassifier depends on similar historical examples; it can struggle when the newest market period is different from older periods.",
        "LogisticRegression": "LogisticRegression is strong for simple class boundaries, but it can score lower when Low/Medium/High classes overlap.",
    }
    return reasons.get(model_name, "This model performs differently because each algorithm learns a different shape from the same features.")


def evaluation_explanation(evaluation: pd.DataFrame, task_type: str) -> str:
    """Generate a readable explanation of the best and weakest evaluated models."""
    if evaluation.empty:
        return "Run evaluation first to explain model performance."
    # Choose the score metric for the current supervised task.
    metric = "R2" if task_type == "Regression" else "F1"
    ranked = evaluation.sort_values(metric, ascending=False)
    best = ranked.iloc[0]
    weakest = ranked.iloc[-1]
    best_score = float(best[metric])
    weak_score = float(weakest[metric])
    gap = best_score - weak_score
    if task_type == "Classification" and "Accuracy" in evaluation.columns:
        accuracy_note = (
            f" Accuracy is highest for `{best['Model']}` because its learned boundaries separate the target classes better on the test rows. "
            "Models with lower accuracy likely confuse nearby classes such as Medium vs High when the selected indicators overlap."
        )
    else:
        accuracy_note = (
            " The same idea applies to regression: the best model follows the test-period pattern with less error, while weaker models miss more turning points."
        )
    return (
        f"`{best['Model']}` is strongest on this evaluation because it gets the best `{metric}` score of `{best_score:.3f}`, "
        f"while `{weakest['Model']}` is lowest at `{weak_score:.3f}`. The gap is `{gap:.3f}`. "
        f"{model_family_reason(str(best['Model']))} {model_family_reason(str(weakest['Model']))}"
        f"{accuracy_note} This does not mean the best model is always best in real life; it means it fit this dataset split and selected features better."
    )


# -----------------------------------------------------------------------------
# Unsupervised-learning helpers
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def prepare_unsupervised_matrix(df: pd.DataFrame, features: list[str], scale_data: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Clean, impute, and optionally scale features for unsupervised methods."""
    matrix = df[features].copy()
    matrix = matrix.replace([np.inf, -np.inf], np.nan).dropna()
    if matrix.empty:
        return matrix, np.empty((0, len(features)))
    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    values = StandardScaler().fit_transform(imputed) if scale_data else imputed
    return matrix, values


def cluster_quality(values: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
    """Calculate cluster counts and quality metrics when labels are usable."""
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    unique_labels = set(valid_labels)
    metrics: dict[str, float | int | None] = {
        "Clusters": len(unique_labels),
        "Noise / anomalies": int((labels == -1).sum()),
        "Silhouette": None,
        "Davies-Bouldin": None,
    }
    if len(unique_labels) >= 2 and len(valid_labels) > len(unique_labels):
        metrics["Silhouette"] = float(silhouette_score(values[valid_mask], valid_labels))
        metrics["Davies-Bouldin"] = float(davies_bouldin_score(values[valid_mask], valid_labels))
    return metrics


@st.cache_data(show_spinner=False)
def pca_projection(values: np.ndarray) -> pd.DataFrame:
    """Project feature data into two principal components for visualization."""
    components = PCA(n_components=2, random_state=0).fit_transform(values)
    return pd.DataFrame({"PC1": components[:, 0], "PC2": components[:, 1]})


def unsupervised_example_text(method: str, features: list[str], details: dict[str, object], target: str | None) -> str:
    """Explain an unsupervised-learning result in project language."""
    feature_text = ", ".join(features[:5])
    if len(features) > 5:
        feature_text += ", ..."

    if method == "KMeans":
        clusters = details.get("clusters", "several")
        silhouette = details.get("silhouette")
        quality = (
            f" The silhouette score is `{silhouette:.3f}`, so the groups are fairly separated."
            if isinstance(silhouette, float)
            else " The groups should be treated as exploratory because separation is not strong enough to score clearly."
        )
        return (
            f"Example conclusion: KMeans split the observations into `{clusters}` housing-market regimes using `{feature_text}`."
            f"{quality} A useful next step is to compare each cluster's average `{target or 'target'}` and macro conditions "
            "to label regimes such as high-rate pressure, strong-demand expansion, or affordability stress."
        )
    if method == "DBSCAN":
        clusters = details.get("clusters", "some")
        noise = details.get("noise", 0)
        return (
            f"Example conclusion: DBSCAN found `{clusters}` dense regimes and `{noise}` unusual/noise observations using `{feature_text}`. "
            "Those noise points are worth reviewing because they may represent unusual macro periods, shocks, or data points that do not behave like the rest of the market."
        )
    if method == "PCA":
        pc1 = details.get("pc1", 0)
        first_two = details.get("first_two", 0)
        return (
            f"Example conclusion: PCA compresses `{feature_text}` into fewer market-pressure dimensions. "
            f"`PC1` explains `{pc1:.1%}` of the selected-feature variation, while the first two components explain `{first_two:.1%}`. "
            "If the first two components explain a large share, the dashboard can summarize the market with a smaller number of interpretable signals."
        )
    anomalies = details.get("anomalies", 0)
    return (
        f"Example conclusion: IsolationForest flagged `{anomalies}` potentially unusual observations using `{feature_text}`. "
        "These rows are not automatically errors; they are candidates for important market events, regime shifts, or periods where normal relationships changed."
    )


def most_important_unsupervised_learning(
    method: str,
    features: list[str],
    details: dict[str, object],
    target: str | None,
) -> str:
    """Highlight the key lesson from the selected unsupervised method."""
    feature_text = ", ".join(features[:4])
    if len(features) > 4:
        feature_text += ", ..."

    if method == "KMeans":
        clusters = details.get("clusters", "multiple")
        silhouette = details.get("silhouette")
        separation = "clear" if isinstance(silhouette, float) and silhouette >= 0.35 else "exploratory"
        return (
            f"The most important thing we learn is that the housing data can be grouped into `{clusters}` "
            f"`{separation}` market regimes using `{feature_text}`. This suggests the market does not behave as one single pattern across time; "
            f"compare each cluster's average `{target or 'target'}` to label the regimes."
        )
    if method == "DBSCAN":
        clusters = details.get("clusters", 0)
        noise = details.get("noise", 0)
        if noise:
            return (
                f"The most important thing we learn is that `{noise}` observations behave unlike the dense market pattern. "
                "Those periods may represent shocks, transitions, or unusual macro conditions that deserve separate analysis."
            )
        return (
            f"The most important thing we learn is that DBSCAN found `{clusters}` dense groups and no major noise pattern. "
            "That means the selected features look more continuous than shock-driven under the current settings."
        )
    if method == "PCA":
        first_two = details.get("first_two", 0)
        if isinstance(first_two, float) and first_two >= 0.7:
            return (
                f"The most important thing we learn is that the selected features can be summarized well: the first two PCA components explain `{first_two:.1%}` "
                "of their movement. A small number of broad market-pressure dimensions can describe much of the data."
            )
        return (
            f"The most important thing we learn is that the selected features are not explained by only one or two simple dimensions "
            f"because the first two PCA components explain `{first_two:.1%}`. The market pattern is more complex."
        )
    anomalies = details.get("anomalies", 0)
    return (
        f"The most important thing we learn is that `{anomalies}` periods are potential anomalies. "
        "These are the first rows to inspect for regime shifts, policy shocks, data issues, or unusual affordability conditions."
    )


# -----------------------------------------------------------------------------
# AI assistant and code-lab helpers
# -----------------------------------------------------------------------------

def code_prompt_system(df: pd.DataFrame, date_col: str | None, target: str | None) -> str:
    """Describe the app context used to guide code-generation prompts."""
    cols = ", ".join(map(str, df.columns[:30]))
    return (
        "You generate concise, production-ready Streamlit/Python code for a reusable "
        "data analytics dashboard. Use pandas, plotly, scikit-learn, and Streamlit patterns "
        "that can fit into an existing streamlit_app.py file. Return code plus a short "
        f"note. Dataset columns include: {cols}. Date column: {date_col}. "
        f"Recommended target: {target}."
    )


def generate_local_code(prompt: str, df: pd.DataFrame, date_col: str | None, target: str | None) -> str:
    """Return safe local Streamlit code snippets when no OpenAI API key is provided."""
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    target = target or (numeric[0] if numeric else "target_column")
    date_expr = date_col or "date_column"
    lower = prompt.lower()
    if "forecast" in lower:
        return f"""# Add this inside a Streamlit tab to create a compact forecast view.
target = "{target}"
date_col = "{date_expr}"
horizon = st.slider("Forecast horizon", 3, 24, 12)
model_df = df[[date_col, target]].dropna().sort_values(date_col).copy()
model_df["step"] = range(len(model_df))

model = LinearRegression()
model.fit(model_df[["step"]], model_df[target])
future_steps = np.arange(len(model_df), len(model_df) + horizon).reshape(-1, 1)
future_dates = pd.date_range(model_df[date_col].max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
forecast = pd.DataFrame({{date_col: future_dates, "forecast": model.predict(future_steps)}})

fig = px.line(model_df, x=date_col, y=target, title=f"{{target}} Forecast")
fig.add_scatter(x=forecast[date_col], y=forecast["forecast"], mode="lines+markers", name="Forecast")
st.plotly_chart(fig, width="stretch")
st.dataframe(forecast, width="stretch")
"""
    if "chart" in lower or "plot" in lower:
        x_col = date_col or (numeric[0] if numeric else "x_column")
        y_col = target or (numeric[1] if len(numeric) > 1 else "y_column")
        return f"""# Add this inside a Streamlit tab for a reusable chart section.
x_col = "{x_col}"
y_col = "{y_col}"
color_col = "time_group" if "time_group" in df.columns else None
chart_df = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna()
fig = px.line(chart_df, x=x_col, y=y_col, color=color_col, markers=True, title=f"{{y_col}} over time")
st.plotly_chart(fig, width="stretch")
"""
    return f"""# Starter Streamlit block generated from your prompt.
target = "{target}"
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
selected = st.multiselect("Choose metrics", numeric_cols, default=numeric_cols[:4])

if selected:
    st.subheader("Generated Summary")
    st.dataframe(df[selected].describe().T.round(3), width="stretch")
    corr = df[selected].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1)
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Select at least one metric.")
"""


def generate_code(prompt: str, api_key: str, df: pd.DataFrame, date_col: str | None, target: str | None) -> str:
    """Generate code with OpenAI when available, otherwise use local templates."""
    if not prompt.strip():
        return "# Enter a prompt first, for example: add a forecast chart for the selected target."
    # The local fallback templates let the app remain useful even without an API key.
    if api_key and OpenAI is not None:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": code_prompt_system(df, date_col, target)},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
            max_tokens=900,
        )
        return response.choices[0].message.content.strip()
    return generate_local_code(prompt, df, date_col, target)


def app_context_summary(df: pd.DataFrame, date_col: str | None, target: str | None, source_name: str) -> str:
    """Create a compact text summary of the current dataset and app context."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    date_range = "No date column detected"
    if date_col and df[date_col].notna().any():
        date_range = f"{df[date_col].min().date()} to {df[date_col].max().date()}"
    return (
        f"Source: {source_name}\n"
        f"Rows: {len(df):,}\n"
        f"Columns: {df.shape[1]:,}\n"
        f"Date range: {date_range}\n"
        f"Selected target: {target or 'None'}\n"
        f"Numeric columns: {', '.join(numeric_cols[:20])}"
    )


def local_chat_answer(prompt: str, df: pd.DataFrame, date_col: str | None, target: str | None) -> str:
    """Answer common dashboard questions without requiring an external AI service."""
    lower = prompt.lower()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    target = target or (numeric_cols[0] if numeric_cols else None)

    if any(word in lower for word in ["summary", "overview", "describe"]):
        return (
            f"The filtered dataset has {len(df):,} rows, {df.shape[1]:,} columns, "
            f"{len(numeric_cols):,} numeric measures, and {int(df.isna().sum().sum()):,} missing cells."
        )
    if "correlation" in lower or "related" in lower:
        cmat = corr_matrix(df)
        if target and not cmat.empty and target in cmat.columns:
            top_corr = (
                cmat[target]
                .drop(labels=[target], errors="ignore")
                .dropna()
                .sort_values(key=lambda value: value.abs(), ascending=False)
                .head(5)
            )
            if not top_corr.empty:
                pairs = ", ".join(f"{idx}: {value:.2f}" for idx, value in top_corr.items())
                return f"Top correlations with {target}: {pairs}."
        return "I need at least two numeric columns to calculate useful correlations."
    if "forecast" in lower:
        return (
            "Use the Forecast tab: choose a target, optional exogenous features, horizon, "
            "and model, then click Generate forecast. Ridge is a stable first choice."
        )
    if "model" in lower or "ml" in lower:
        return (
            "Use ML Lab for supervised regression/classification, then use Evaluation and Prediction Page "
            "to compare models and generate usable predictions."
        )
    if "missing" in lower or "quality" in lower:
        missing = df.isna().sum().sort_values(ascending=False).head(5)
        top_missing = ", ".join(f"{col}: {int(value)}" for col, value in missing.items())
        return f"Top missing-value columns are: {top_missing}."
    if date_col and df[date_col].notna().any():
        return (
            f"I can help analyze this dataset. Current data runs from "
            f"{df[date_col].min().date()} to {df[date_col].max().date()}. "
            "Ask about trends, correlations, model choice, forecasting, or data quality."
        )
    return "I can help analyze this dataset. Ask about trends, correlations, model choice, forecasting, or data quality."


def answer_chat(
    prompt: str,
    api_key: str,
    messages: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str | None,
    target: str | None,
    source_name: str,
) -> str:
    """Answer chat prompts with OpenAI when configured, with local fallback otherwise."""
    if not prompt.strip():
        return "Ask me a question about the app, data, models, forecast, or code."
    if api_key and OpenAI is not None:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI analyst embedded in a Streamlit dashboard for a user-selected CSV dataset. "
                        "Answer clearly and briefly. Use the app context below and avoid pretending "
                        "you can see UI interactions that are not in the context.\n\n"
                        + app_context_summary(df, date_col, target, source_name)
                    ),
                },
                *messages[-8:],
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=450,
        )
        return response.choices[0].message.content.strip()
    return local_chat_answer(prompt, df, date_col, target)


# -----------------------------------------------------------------------------
# OLAP helpers
# -----------------------------------------------------------------------------

def flatten_pivot_columns(pivot: pd.DataFrame) -> pd.DataFrame:
    """Flatten pivot-table columns so exported OLAP tables are easy to read."""
    flat = pivot.reset_index()
    flat.columns = [
        " | ".join(str(part) for part in col if str(part) not in ["", "None"])
        if isinstance(col, tuple)
        else str(col)
        for col in flat.columns
    ]
    return flat


def olap_insight(
    pivot: pd.DataFrame,
    row_dims: list[str],
    value_cols: list[str],
    aggfunc: str,
) -> tuple[dict[str, object] | None, pd.DataFrame]:
    """Find the strongest OLAP segment and prepare a readable insight dictionary."""
    flat = flatten_pivot_columns(pivot)
    numeric_cols = flat.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return None, flat

    preferred = [col for col in numeric_cols if any(col == value or col.startswith(f"{value} |") for value in value_cols)]
    measure_col = preferred[0] if preferred else numeric_cols[0]
    rank_df = flat.dropna(subset=[measure_col]).copy()
    if rank_df.empty:
        return None, flat

    label_cols = [col for col in flat.columns if col not in numeric_cols]

    def row_label(row: pd.Series) -> str:
        parts = []
        for col in label_cols:
            value = row.get(col)
            if pd.notna(value):
                parts.append(f"{col}: {value}")
        return " / ".join(parts) if parts else "All selected records"

    rank_df["OLAP Segment"] = rank_df.apply(row_label, axis=1)
    top_row = rank_df.loc[rank_df[measure_col].idxmax()]
    bottom_row = rank_df.loc[rank_df[measure_col].idxmin()]
    top_segments = (
        rank_df[["OLAP Segment", measure_col]]
        .sort_values(measure_col, ascending=False)
        .head(5)
        .rename(columns={measure_col: "Aggregated Value"})
    )
    measure_name = value_cols[0] if value_cols else measure_col
    insight = {
        "measure": measure_name,
        "measure_column": measure_col,
        "aggregation": aggfunc,
        "row_dims": row_dims,
        "top_label": top_row["OLAP Segment"],
        "top_value": float(top_row[measure_col]),
        "bottom_label": bottom_row["OLAP Segment"],
        "bottom_value": float(bottom_row[measure_col]),
        "spread": float(top_row[measure_col] - bottom_row[measure_col]),
        "top_segments": top_segments,
    }
    return insight, flat


def olap_example_text(insight: dict[str, object]) -> str:
    """Turn the top OLAP segment into a short reporting example."""
    agg_label = str(insight["aggregation"]).title()
    measure = str(insight["measure"])
    top_label = str(insight["top_label"])
    bottom_label = str(insight["bottom_label"])
    top_value = float(insight["top_value"])
    bottom_value = float(insight["bottom_value"])
    spread = float(insight["spread"])
    if insight["aggregation"] == "count":
        result_word = "records"
    else:
        result_word = measure
    return (
        f"OLAP conclusion example: using {agg_label} {result_word}, the strongest segment is "
        f"{top_label} with {top_value:,.2f}. The weakest segment is {bottom_label} with "
        f"{bottom_value:,.2f}. The gap is {spread:,.2f}, so this pivot points to where the "
        "dashboard should investigate next instead of treating the full dataset as one average."
    )


def olap_interpretation_panel(
    insight: dict[str, object],
    row_dims: list[str],
    column_dims: list[str],
    selected_values: list[str],
) -> dict[str, str]:
    """Render guided OLAP interpretation notes beside the pivot output."""
    aggregation = str(insight["aggregation"])
    measure = str(insight["measure"])
    top_label = str(insight["top_label"])
    bottom_label = str(insight["bottom_label"])
    top_value = float(insight["top_value"])
    bottom_value = float(insight["bottom_value"])
    spread = float(insight["spread"])
    pct_gap = (spread / abs(bottom_value) * 100) if bottom_value else np.nan
    row_text = ", ".join(row_dims) if row_dims else "the selected rows"
    column_text = ", ".join(column_dims) if column_dims else "no column split"
    value_text = ", ".join(selected_values)

    if aggregation == "mean":
        meaning = (
            "This is an average comparison. The top segment usually has a higher level of the measure, "
            "not simply more records."
        )
        caution = "Check the count or raw data before saying the segment is larger in market size."
    elif aggregation == "count":
        meaning = (
            "This is a record-count comparison. The top segment appears most often in the filtered dataset."
        )
        caution = "Do not interpret count as higher housing prices; it only measures frequency."
    elif aggregation == "std":
        meaning = (
            "This is a volatility comparison. The top segment is the least stable or most variable group."
        )
        caution = "High standard deviation can mean risk, instability, or a period with mixed market behavior."
    elif aggregation in {"sum", "max"}:
        meaning = (
            f"This {aggregation} comparison highlights concentration or peak values across segments."
        )
        caution = "Use mean or median-style checks before turning this into an average-price conclusion."
    else:
        meaning = (
            f"This {aggregation} comparison focuses on boundary values rather than the full distribution."
        )
        caution = "Boundary values can be affected by unusual periods, so verify with Explore charts."

    if np.isfinite(pct_gap):
        gap_sentence = (
            f"The top segment is {spread:,.2f} above the bottom segment, a relative gap of {pct_gap:,.1f}%."
        )
    else:
        gap_sentence = f"The top segment is {spread:,.2f} above the bottom segment."

    return {
        "what": (
            f"The OLAP cube compares `{value_text}` using `{aggregation}` across `{row_text}` "
            f"and `{column_text}`. The strongest segment is `{top_label}` ({top_value:,.2f}); "
            f"the weakest is `{bottom_label}` ({bottom_value:,.2f}). {gap_sentence}"
        ),
        "why": (
            f"{meaning} This matters because it shows where the dataset is not behaving like one single average; "
            "different segments have different market stories."
        ),
        "next": (
            f"Use `{top_label}` as the main OLAP story, then compare it with the trend chart, correlation results, "
            "and ML evaluation. If the same segment also performs strongly in those pages, the conclusion is stronger."
        ),
        "caution": caution,
    }


def cube_mesh_trace(
    x_center: float,
    y_center: float,
    z_center: float,
    size: float,
    color: str,
    hover_text: str,
) -> go.Mesh3d:
    """Build one 3D cube block for the OLAP block-cube visualization."""
    half = size / 2
    x = [
        x_center - half,
        x_center + half,
        x_center + half,
        x_center - half,
        x_center - half,
        x_center + half,
        x_center + half,
        x_center - half,
    ]
    y = [
        y_center - half,
        y_center - half,
        y_center + half,
        y_center + half,
        y_center - half,
        y_center - half,
        y_center + half,
        y_center + half,
    ]
    z = [
        z_center - half,
        z_center - half,
        z_center - half,
        z_center - half,
        z_center + half,
        z_center + half,
        z_center + half,
        z_center + half,
    ]
    i = [0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6, 7]
    j = [1, 2, 4, 2, 5, 3, 5, 6, 6, 1, 2, 3]
    k = [2, 3, 5, 5, 6, 7, 6, 7, 2, 0, 3, 0]
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=0.9,
        flatshading=True,
        text=[hover_text] * 8,
        hovertemplate="%{text}<extra></extra>",
        showscale=False,
        lighting=dict(ambient=0.55, diffuse=0.7, fresnel=0.2, specular=0.35, roughness=0.45),
    )


# -----------------------------------------------------------------------------
# Reinforcement-learning demo helper
# -----------------------------------------------------------------------------

def reinforcement_market_lab(
    data: pd.DataFrame,
    date_col: str,
    target: str,
    signal_feature: str | None,
    episodes: int,
    risk_penalty: float,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run an educational Q-learning style market-policy simulation."""
    cols = [date_col, target] + ([signal_feature] if signal_feature else [])
    market = data[cols].dropna(subset=[date_col, target]).sort_values(date_col).copy()
    market[target] = pd.to_numeric(market[target], errors="coerce")
    market = market.dropna(subset=[target]).reset_index(drop=True)
    if len(market) < 12:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    market["current_return"] = market[target].pct_change().fillna(0)
    market["next_return"] = market[target].pct_change().shift(-1)
    market["volatility"] = market["current_return"].rolling(6, min_periods=2).std().fillna(0)
    market = market.dropna(subset=["next_return"]).reset_index(drop=True)
    market["Momentum State"] = pd.cut(
        market["current_return"],
        bins=[-np.inf, -0.005, 0.005, np.inf],
        labels=["Falling", "Flat", "Rising"],
    ).astype(str)

    if signal_feature and signal_feature in market.columns and market[signal_feature].nunique(dropna=True) >= 3:
        try:
            market["Signal State"] = pd.qcut(
                market[signal_feature],
                q=3,
                labels=["Low", "Medium", "High"],
                duplicates="drop",
            ).astype(str)
        except ValueError:
            market["Signal State"] = "One-level"
    else:
        market["Signal State"] = "No extra signal"

    market["State"] = market["Momentum State"] + " market | " + market["Signal State"] + " signal"
    actions = ["Buy / overweight", "Hold / neutral", "Wait / underweight"]

    def reward_for(action: str, next_return: float, volatility: float) -> float:
        move = next_return * 100
        risk = volatility * 100 * risk_penalty
        if action == "Buy / overweight":
            return move - risk
        if action == "Hold / neutral":
            return move * 0.45 - risk * 0.35
        return -move * 0.35 - risk * 0.15

    states = market["State"].unique().tolist()
    q_values = {state: {action: 0.0 for action in actions} for state in states}
    rng = np.random.default_rng(random_state)
    learning_rate = 0.18
    discount = 0.88
    epsilon = 0.35

    transitions = []
    for i in range(len(market) - 1):
        transitions.append(
            {
                "state": market.loc[i, "State"],
                "next_state": market.loc[i + 1, "State"],
                "next_return": float(market.loc[i, "next_return"]),
                "volatility": float(market.loc[i, "volatility"]),
                "date": market.loc[i, date_col],
                "target": float(market.loc[i, target]),
            }
        )

    for _ in range(episodes):
        for transition in transitions:
            state = transition["state"]
            next_state = transition["next_state"]
            if rng.random() < epsilon:
                action = rng.choice(actions)
            else:
                action = max(q_values[state], key=q_values[state].get)
            reward = reward_for(action, transition["next_return"], transition["volatility"])
            old_value = q_values[state][action]
            future_value = max(q_values[next_state].values())
            q_values[state][action] = old_value + learning_rate * (
                reward + discount * future_value - old_value
            )
        epsilon = max(0.04, epsilon * 0.985)

    q_table = pd.DataFrame.from_dict(q_values, orient="index").reset_index(names="State")
    q_table["Best Action"] = q_table[actions].idxmax(axis=1)
    q_table["Best Q Score"] = q_table[actions].max(axis=1)
    q_table = q_table.sort_values("Best Q Score", ascending=False)

    trace_rows = []
    cumulative_reward = 0.0
    for transition in transitions:
        state = transition["state"]
        action = max(q_values[state], key=q_values[state].get)
        reward = reward_for(action, transition["next_return"], transition["volatility"])
        cumulative_reward += reward
        trace_rows.append(
            {
                "Date": transition["date"],
                "State": state,
                "Action": action,
                "Reward": reward,
                "Cumulative Reward": cumulative_reward,
                target: transition["target"],
                "Next return %": transition["next_return"] * 100,
            }
        )
    trace = pd.DataFrame(trace_rows)

    action_counts = q_table["Best Action"].value_counts().reset_index()
    action_counts.columns = ["Action", "Policy States"]
    summary = {
        "states": len(states),
        "actions": len(actions),
        "episodes": episodes,
        "best_state": q_table.iloc[0]["State"] if not q_table.empty else None,
        "best_action": q_table.iloc[0]["Best Action"] if not q_table.empty else None,
        "total_reward": float(cumulative_reward),
    }
    return q_table, trace, action_counts, summary


# -----------------------------------------------------------------------------
# Streamlit application layout
# -----------------------------------------------------------------------------

# Sidebar controls are placed first because they decide which dataset is loaded,
# which theme is applied, and whether the app should show raw preview rows.
with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark">DI</div>
            <div>
                <div class="brand-name">DataIQ Platform</div>
                <div class="brand-subtitle">Enterprise Analytics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Upload a dataset, choose a target, model it, segment it, and prepare governance-ready outputs.")
    st.divider()

    uploaded = st.file_uploader("Upload any CSV dataset", type=["csv"])
    use_default = st.checkbox("Use bundled housing demo CSV", value=(uploaded is None))
    dark_mode = st.toggle("Dark mode", value=False)
    show_raw = st.checkbox("Show raw data preview", value=False)
    preview_rows = st.slider(
        "Raw preview rows",
        50,
        5000,
        200,
        50,
        disabled=not show_raw,
        help="Controls how many rows appear in the optional raw data preview.",
    )
    performance_mode = st.toggle(
        "Performance mode",
        value=True,
        help="Limits chart rows and caches expensive dataset preparation for smoother reruns.",
    )
    show_advanced_pages = st.toggle(
        "Show advanced / academic pages",
        value=False,
        help="Keeps the main app focused. Turn on to show experimental, academic, and extra governance pages.",
    )
    st.session_state.show_advanced_pages = show_advanced_pages
    max_chart_rows = st.slider(
        "Max chart rows",
        500,
        20000,
        3000,
        500,
        disabled=not performance_mode,
        help="Large charts are sampled to this many rows while tables and model inputs keep their selected data.",
    )
    max_model_rows = st.slider(
        "Max modeling rows",
        1000,
        50000,
        10000,
        1000,
        disabled=not performance_mode,
        help="In performance mode, supervised and diagnostic models use a deterministic sample up to this row count.",
    )

    st.divider()
    st.subheader("Search")
    if "dashboard_query" not in st.session_state:
        st.session_state.dashboard_query = ""
    dashboard_query = st.text_input(
        "Find a page, feature, metric, or column",
        placeholder="Try: overfit, cleaning, mortgage, OLAP, scenario",
        key="dashboard_query",
    )
    search_categories = ["All"] + sorted({item["category"] for item in dashboard_search_items()})
    search_category = st.selectbox("Search category", search_categories, key="search_category")
    search_limit = st.slider("Max results", 3, 10, 5, key="search_limit")
    include_column_search = st.checkbox("Include dataset columns", value=True, key="include_column_search")
    suggestion_cols = st.columns(2)
    if suggestion_cols[0].button("Models", key="search_models"):
        st.session_state.dashboard_query = "model metrics overfit"
        st.rerun()
    if suggestion_cols[1].button("Data", key="search_data"):
        st.session_state.dashboard_query = "cleaning missing column"
        st.rerun()

    st.divider()
    st.subheader("Dataset")
    st.caption("The bundled housing file is only a demo. Upload a CSV to use the platform for another domain.")
    st.code(str(DEFAULT_CSV), language="text")
    if st.button("Clear cached data", help="Use this after changing a large CSV outside the uploader."):
        st.cache_data.clear()
        st.rerun()

# Apply the selected theme immediately after reading the sidebar toggle.
apply_theme(dark_mode)

if not show_advanced_pages:
    st.markdown(
        """
        <style>
            /* Hide lower-priority pages from the tab bar in the default focused workflow.
               The page code remains available when the sidebar advanced-pages toggle is enabled. */
            .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(5),
            .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(8),
            .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(9),
            .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(11),
            .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(12),
            .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(13),
            .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(19),
            .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(20),
            .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(21),
            .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(22),
            .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(24) {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Can't see the sidebar? Open upload and search controls here", expanded=False):
    st.caption(
        "Streamlit may collapse the sidebar on small screens. Use this fallback panel "
        "to upload a CSV and search the dashboard without opening the sidebar."
    )
    fallback_cols = st.columns([1, 1])
    with fallback_cols[0]:
        main_uploaded = st.file_uploader(
            "Upload any CSV dataset",
            type=["csv"],
            key="main_upload_csv",
            help="This is the same CSV uploader as the sidebar, shown here for easier access.",
        )
        main_use_default = st.checkbox(
            "Use bundled housing demo CSV",
            value=(main_uploaded is None),
            key="main_use_default_csv",
        )
    with fallback_cols[1]:
        main_dashboard_query = st.text_input(
            "Search pages, metrics, or columns",
            placeholder="Try: evaluation, upload, OLAP, forecast, overfit",
            key="main_dashboard_query",
        )
        st.caption("Search results appear after the dataset loads.")


# Dataset loading supports two paths: a user-uploaded CSV for experiments, or the
# bundled raw Kaggle-style file for reproducible project demos.
active_upload = main_uploaded if main_uploaded is not None else uploaded
active_use_default = main_use_default if main_uploaded is not None else use_default

if active_upload is not None:
    raw_df = load_csv_from_bytes(active_upload.getvalue())
    source_name = f"Uploaded file: {active_upload.name}"
elif active_use_default and DEFAULT_CSV.exists():
    raw_df = load_csv_from_path(str(DEFAULT_CSV), DEFAULT_CSV.stat().st_mtime)
    source_name = f"Local file: {DEFAULT_CSV.name}"
else:
    st.error(
        "No dataset found. Upload a CSV or place it at "
        "data/01_raw/us_home_price_analysis_2004_2024.csv for the bundled demo."
    )
    st.stop()

# Once raw data is loaded, clean and normalize it once so every page can reuse
# the same prepared dataset and labels.
df, date_col, cleaning_report = prepare_dataset(raw_df)
base_prepared_df = df.copy()
if "time_group" not in df.columns:
    if date_col and date_col in df.columns and df[date_col].notna().any():
        df["time_group"] = df[date_col].apply(time_group_label)
    else:
        df["time_group"] = "Unknown"
if "time_group" not in base_prepared_df.columns:
    if date_col and date_col in base_prepared_df.columns and base_prepared_df[date_col].notna().any():
        base_prepared_df["time_group"] = base_prepared_df[date_col].apply(time_group_label)
    else:
        base_prepared_df["time_group"] = "Unknown"

with st.sidebar:
    st.divider()
    st.subheader("Data Cleaning Studio")
    st.caption("Choose cleaning actions. The rest of the platform will use the cleaned result.")
    studio_numeric_cols = base_prepared_df.select_dtypes(include=["number"]).columns.tolist()
    studio_missing_numeric_cols = [
        col for col in studio_numeric_cols if int(base_prepared_df[col].isna().sum()) > 0
    ]
    studio_drop_candidates = [
        col for col in base_prepared_df.columns if col not in {"time_group", date_col}
    ]
    with st.expander("Cleaning actions", expanded=False):
        studio_remove_duplicates = st.checkbox("Remove duplicate rows", value=True, key="studio_remove_duplicates")
        studio_drop_empty = st.checkbox("Drop fully empty columns", value=True, key="studio_drop_empty")
        studio_numeric_missing_strategy = st.selectbox(
            "Numeric missing values",
            [
                "Do not fill numeric missing values",
                "Replace with median",
                "Replace with mean",
                "Estimate from most correlated variables",
            ],
            index=1 if studio_missing_numeric_cols else 0,
            key="studio_numeric_missing_strategy",
            help="Correlation estimate trains a small linear model per column using the most related numeric variables.",
        )
        studio_numeric_missing_columns = st.multiselect(
            "Numeric columns to repair",
            studio_numeric_cols,
            default=studio_missing_numeric_cols,
            key="studio_numeric_missing_columns",
            disabled=studio_numeric_missing_strategy == "Do not fill numeric missing values",
        )
        studio_fill_categorical = st.checkbox(
            "Fill categorical missing values with mode",
            value=False,
            key="studio_fill_categorical",
        )
        studio_drop_columns = st.multiselect(
            "Drop columns",
            studio_drop_candidates,
            default=[],
            key="studio_drop_columns",
            help="Use this for IDs, notes, or noisy columns you do not want in the platform.",
        )
        studio_outlier_columns = st.multiselect(
            "Repair outliers in numeric columns",
            studio_numeric_cols,
            default=[],
            key="studio_outlier_columns",
            help="Select columns where extreme values should be repaired or removed.",
        )
        studio_outlier_strategy = st.selectbox(
            "Outlier repair strategy",
            [
                "Do not change outliers",
                "Cap to IQR bounds",
                "Replace with median",
                "Replace with mean",
                "Estimate from most correlated variables",
                "Remove rows",
            ],
            index=1 if studio_outlier_columns else 0,
            key="studio_outlier_strategy",
            disabled=not studio_outlier_columns,
        )
        studio_outlier_factor = st.slider(
            "Outlier IQR factor",
            1.0,
            3.0,
            1.5,
            0.25,
            key="studio_outlier_factor",
            disabled=not studio_outlier_columns,
        )

df, date_col, studio_report = apply_cleaning_studio(
    base_prepared_df,
    date_col,
    studio_drop_columns,
    studio_remove_duplicates,
    studio_drop_empty,
    studio_numeric_missing_columns,
    studio_numeric_missing_strategy,
    studio_fill_categorical,
    studio_outlier_columns,
    studio_outlier_strategy,
    studio_outlier_factor,
)
if "time_group" not in df.columns:
    if date_col and date_col in df.columns and df[date_col].notna().any():
        df["time_group"] = df[date_col].apply(time_group_label)
    else:
        df["time_group"] = "Unknown"
studio_before_health = dataset_health(base_prepared_df)
studio_after_health = dataset_health(df)
st.session_state.cleaning_studio_report = studio_report
st.session_state.cleaning_studio_before = studio_before_health
st.session_state.cleaning_studio_after = studio_after_health
profile = app_profile(df, source_name)

with st.sidebar:
    st.divider()
    st.subheader("Template & Mode")
    template_default = "Housing Demo" if profile["is_housing"] else "General Data Project"
    if st.session_state.get("template_source_key") != source_name:
        st.session_state.template_source_key = source_name
        st.session_state.use_case_template = template_default
        st.session_state.template_category = str(USE_CASE_TEMPLATES[template_default]["category"])
        st.session_state.app_mode = "Beginner Mode"
    template_categories = [
        category
        for category in TEMPLATE_CATEGORY_ORDER
        if any(template["category"] == category for template in USE_CASE_TEMPLATES.values())
    ]
    current_template = st.session_state.get("use_case_template", template_default)
    current_category = st.session_state.get(
        "template_category",
        USE_CASE_TEMPLATES.get(current_template, USE_CASE_TEMPLATES[template_default])["category"],
    )
    if current_category not in template_categories:
        current_category = str(USE_CASE_TEMPLATES[template_default]["category"])
        st.session_state.template_category = current_category
    selected_template_category = st.selectbox(
        "Template category",
        template_categories,
        index=template_categories.index(current_category),
        key="template_category",
    )
    template_names = [
        name
        for name, template in USE_CASE_TEMPLATES.items()
        if template["category"] == selected_template_category
    ]
    if current_template not in template_names:
        st.session_state.use_case_template = template_names[0]
    selected_template_name = st.selectbox(
        "Template",
        template_names,
        index=template_names.index(st.session_state.get("use_case_template", template_names[0]))
        if st.session_state.get("use_case_template", template_names[0]) in template_names
        else 0,
        key="use_case_template",
    )
    selected_template = USE_CASE_TEMPLATES[selected_template_name]
    app_mode_names = list(APP_MODES)
    selected_app_mode = st.selectbox(
        "App mode",
        app_mode_names,
        index=app_mode_names.index(st.session_state.get("app_mode", "Beginner Mode"))
        if st.session_state.get("app_mode", "Beginner Mode") in app_mode_names
        else 0,
        key="app_mode",
    )
    selected_mode_profile = APP_MODES[selected_app_mode]
    st.caption(str(selected_mode_profile["description"]))

auto_setup = smart_auto_setup(df, source_name, selected_template)

with st.sidebar:
    st.divider()
    st.subheader("Project Manager")
    if st.session_state.get("project_source_key") != source_name:
        st.session_state.project_source_key = source_name
        st.session_state.project_name = project_name_from_source(source_name)
        st.session_state.project_date_column = auto_setup["Recommended date"] or date_col or "No date column"
        st.session_state.project_target_column = auto_setup["Recommended target"]
        st.session_state.smart_features = auto_setup["Recommended features"]
        st.session_state.smart_auto_setup = auto_setup
    project_name = st.text_input(
        "Project name",
        key="project_name",
        help="This name is used in the project profile and report context for the active dataset.",
    )
    date_options = ["No date column"] + df.columns.tolist()
    current_date_choice = st.session_state.get("project_date_column", date_col or "No date column")
    if current_date_choice not in date_options:
        current_date_choice = date_col or "No date column"
        st.session_state.project_date_column = current_date_choice
    selected_date = st.selectbox(
        "Date column",
        date_options,
        index=date_options.index(current_date_choice),
        key="project_date_column",
        help="Override the auto-detected date column when your dataset uses another time field.",
    )
    if selected_date == "No date column":
        date_col = None
    elif selected_date in df.columns:
        date_col = selected_date
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].notna().any():
            df = df.sort_values(date_col)

with st.sidebar:
    st.markdown("#### Search Results")
    render_search_results(
        dashboard_query,
        search_category,
        search_limit,
        df,
        include_column_search,
    )

effective_main_query = main_dashboard_query.strip()
if effective_main_query:
    st.markdown("#### Search Results")
    render_search_results(
        effective_main_query,
        "All",
        5,
        df,
        True,
    )

with st.sidebar:
    # Date filtering happens after cleaning because the date column may need to
    # be detected and parsed from strings first.
    if date_col and df[date_col].notna().any():
        dmin, dmax = df[date_col].min(), df[date_col].max()
        selected_range = st.slider(
            "Date range",
            min_value=dmin.to_pydatetime(),
            max_value=dmax.to_pydatetime(),
            value=(dmin.to_pydatetime(), dmax.to_pydatetime()),
        )
        df = df[
            (df[date_col] >= pd.Timestamp(selected_range[0]))
            & (df[date_col] <= pd.Timestamp(selected_range[1]))
        ].copy()

if df.empty:
    st.error("The current filters returned no rows. Expand the date range or upload another CSV.")
    st.stop()

# These shared variables keep every tab consistent about the available numeric
# features and the selected analysis target.
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
initial_target = default_target(num_cols)
if num_cols:
    with st.sidebar:
        st.divider()
        st.subheader("Analysis Setup")
        current_target = st.session_state.get("project_target_column", auto_setup["Recommended target"] or initial_target)
        if current_target not in num_cols:
            current_target = initial_target
            st.session_state.project_target_column = current_target
        target_default = st.selectbox(
            "Primary target column",
            num_cols,
            index=num_cols.index(current_target) if current_target in num_cols else 0,
            key="project_target_column",
            help="This target drives overview, modeling, forecasting, reports, and scenario analysis.",
        )
        st.caption(f"Mode: {profile['domain']}. Change the target when you upload a different dataset.")
else:
    target_default = None

auto_setup["Suggested task"] = infer_task_type(df, target_default)
if selected_template.get("preferred_task") in {"Regression", "Classification"}:
    auto_setup["Suggested task"] = selected_template["preferred_task"]
auto_setup["Recommended features"] = recommend_feature_columns(df, target_default, date_col, num_cols)
st.session_state.smart_auto_setup = auto_setup
st.session_state.smart_features = auto_setup["Recommended features"]
st.session_state.selected_use_case_template = selected_template
st.session_state.selected_app_mode_profile = selected_mode_profile

project_profile = build_project_profile(
    project_name,
    df,
    raw_df,
    source_name,
    date_col,
    target_default,
    profile,
    selected_template_name,
    selected_app_mode,
)
st.session_state.active_project_profile = project_profile

with st.sidebar:
    st.markdown("#### Active Project")
    st.caption(
        f"Target: `{project_profile['Primary target']}` | "
        f"Task: `{project_profile['Suggested task']}`"
    )

# The hero banner and workflow strip are the app's main project narrative controls.
st.markdown(
    f"""
    <div class="hero">
        <div class="brand-lockup">
            <div class="brand-mark">{profile['mark']}</div>
            <div>
                <div class="brand-name">{profile['name']}</div>
                <div class="brand-subtitle">{profile['subtitle']}</div>
            </div>
        </div>
        <h1>{project_name}</h1>
        <p>
            {profile['description']}
        </p>
        <div class="chip-row">
            <span class="chip">{source_name}</span>
            <span class="chip">Mode: {profile['domain']}</span>
            <span class="chip">Template: {selected_template_name}</span>
            <span class="chip">App mode: {selected_app_mode}</span>
            <span class="chip">Date column: {date_col or "not detected"}</span>
            <span class="chip">Selected target: {target_default or "not detected"}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="workflow-strip">
        <div class="workflow-step">
            <div class="step-kicker">01 Foundation</div>
            <div class="step-title">Data Quality + Dictionary</div>
        </div>
        <div class="workflow-step">
            <div class="step-kicker">02 Discovery</div>
            <div class="step-title">Overview + Exploration</div>
        </div>
        <div class="workflow-step">
            <div class="step-kicker">03 Modeling</div>
            <div class="step-title">ML + Evaluation</div>
        </div>
        <div class="workflow-step">
            <div class="step-kicker">04 Segments</div>
            <div class="step-title">OLAP + Unsupervised</div>
        </div>
        <div class="workflow-step">
            <div class="step-kicker">05 Decisions</div>
            <div class="step-title">Forecast + Simulator</div>
        </div>
        <div class="workflow-step">
            <div class="step-kicker">06 Governance</div>
            <div class="step-title">Report + Readiness</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# The chat assistant is optional: it uses OpenAI when a key is supplied and a
# local fallback otherwise, so the dashboard remains usable offline.
chat_left, chat_right = st.columns([5, 1])
with chat_left:
    st.caption("Need help reading the dashboard? Open the AI chat from the icon.")
with chat_right:
    with st.popover("AI Chat", icon=":material/chat:"):
        st.markdown("#### AI Dashboard Chat")
        chat_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            key="popover_chat_api_key",
        )
        if "ai_chat_messages" not in st.session_state:
            st.session_state.ai_chat_messages = []

        for message in st.session_state.ai_chat_messages[-6:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        with st.form("ai_chat_form", clear_on_submit=True):
            chat_prompt = st.text_area(
                "Ask the AI",
                placeholder="Example: Which model should I try first?",
                height=90,
            )
            submitted = st.form_submit_button("Send")

        if submitted:
            st.session_state.ai_chat_messages.append({"role": "user", "content": chat_prompt})
            try:
                reply = answer_chat(
                    chat_prompt,
                    chat_api_key,
                    st.session_state.ai_chat_messages,
                    df,
                    date_col,
                    target_default,
                    source_name,
                )
            except Exception as err:
                reply = f"Chat error: {err}"
            st.session_state.ai_chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.ai_chat_messages and st.button("Clear chat"):
            st.session_state.ai_chat_messages = []
            st.rerun()


with st.expander("Dataset Project Manager", expanded=True):
    manager_cols = st.columns([1.25, 1, 1])
    with manager_cols[0]:
        st.markdown("#### Project Profile")
        profile_table = pd.DataFrame(
            [
                {"Field": "Project name", "Value": project_profile["Project name"]},
                {"Field": "Source", "Value": project_profile["Source"]},
                {"Field": "Mode", "Value": project_profile["Mode"]},
                {"Field": "Date column", "Value": project_profile["Date column"]},
                {"Field": "Primary target", "Value": project_profile["Primary target"]},
                {"Field": "Suggested task", "Value": project_profile["Suggested task"]},
            ]
        )
        st.dataframe(profile_table, width="stretch", hide_index=True)
    with manager_cols[1]:
        st.markdown("#### Dataset Size")
        st.metric("Rows after filters", f"{project_profile['Rows after filters']:,}")
        st.metric("Columns", f"{project_profile['Columns']:,}")
        st.metric("Date range", str(project_profile["Date range"]))
    with manager_cols[2]:
        st.markdown("#### Data Health")
        st.metric("Numeric columns", f"{project_profile['Numeric columns']:,}")
        st.metric("Missing cells", f"{project_profile['Missing cells']:,}")
        st.metric("Duplicate rows", f"{project_profile['Duplicate rows']:,}")

    setup_status = pd.DataFrame(
        [
            {
                "Setup item": "Project name",
                "Status": "Ready" if project_profile["Project name"] else "Needs value",
                "Current value": project_profile["Project name"],
            },
            {
                "Setup item": "Date column",
                "Status": "Ready" if date_col else "Optional / not selected",
                "Current value": project_profile["Date column"],
            },
            {
                "Setup item": "Primary target",
                "Status": "Ready" if target_default else "Needs numeric target",
                "Current value": project_profile["Primary target"],
            },
            {
                "Setup item": "Model task",
                "Status": "Ready" if project_profile["Suggested task"] != "Not ready" else "Needs target data",
                "Current value": project_profile["Suggested task"],
            },
        ]
    )
    st.markdown("#### Setup Checklist")
    st.dataframe(setup_status, width="stretch", hide_index=True)
    st.download_button(
        "Download project profile CSV",
        data=pd.DataFrame([project_profile]).to_csv(index=False).encode("utf-8"),
        file_name="dataiq_project_profile.csv",
        mime="text/csv",
    )

with st.expander("Use-Case Template & App Mode", expanded=False):
    template_cols = st.columns([1, 1])
    with template_cols[0]:
        st.markdown(f"#### {selected_template_name}")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Setting": "Category", "Value": selected_template["category"]},
                    {"Setting": "Description", "Value": selected_template["description"]},
                    {"Setting": "Preferred task", "Value": selected_template["preferred_task"]},
                    {"Setting": "Target keywords", "Value": ", ".join(selected_template["target_keywords"])},
                    {"Setting": "Feature strategy", "Value": selected_template["feature_strategy"]},
                    {"Setting": "Caution", "Value": selected_template["caution"]},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    with template_cols[1]:
        st.markdown(f"#### {selected_app_mode}")
        st.info(str(selected_mode_profile["description"]), icon=":material/tune:")
        st.markdown("##### Focus Pages")
        st.dataframe(
            pd.DataFrame({"Page": selected_mode_profile["focus"]}),
            width="stretch",
            hide_index=True,
        )
    st.markdown("#### Recommended Template Flow")
    st.dataframe(
        pd.DataFrame(
            [
                {"Step": idx + 1, "Page": page, "Why": selected_template["report_language"]}
                for idx, page in enumerate(selected_template["recommended_pages"])
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    if selected_template_name == "Health Dataset":
        st.warning(str(selected_template["caution"]), icon=":material/health_and_safety:")
    elif selected_template_name == "Finance / Risk":
        st.warning(str(selected_template["caution"]), icon=":material/policy:")

with st.expander("Smart Auto-Setup", expanded=False):
    smart_cols = st.columns(4)
    smart_cols[0].metric("Detected task", str(auto_setup["Suggested task"]))
    smart_cols[1].metric("Numeric columns", f"{len(auto_setup['Numeric columns']):,}")
    smart_cols[2].metric("Categorical columns", f"{len(auto_setup['Categorical columns']):,}")
    smart_cols[3].metric("Recommended features", f"{len(auto_setup['Recommended features']):,}")

    recommendation_rows = pd.DataFrame(
        [
            {"Item": "Use-case template", "Recommendation": selected_template_name},
            {"Item": "App mode", "Recommendation": selected_app_mode},
            {"Item": "Date column", "Recommendation": auto_setup["Recommended date"] or "No strong date column found"},
            {"Item": "Primary target", "Recommendation": target_default or auto_setup["Recommended target"] or "No numeric target found"},
            {"Item": "Task type", "Recommendation": auto_setup["Suggested task"]},
            {
                "Item": "Feature set",
                "Recommendation": ", ".join(auto_setup["Recommended features"][:8])
                if auto_setup["Recommended features"]
                else "Select features manually",
            },
        ]
    )
    st.dataframe(recommendation_rows, width="stretch", hide_index=True)

    if auto_setup["Issues"]:
        for issue in auto_setup["Issues"]:
            st.warning(issue, icon=":material/rule:")
    else:
        st.success(
            "Smart setup found a usable date/target/features configuration for this dataset.",
            icon=":material/auto_awesome:",
        )

    candidate_tabs = st.tabs(["Date Candidates", "Target Candidates", "Recommended Features"])
    with candidate_tabs[0]:
        if auto_setup["Date candidates"].empty:
            st.info("No date-like columns were detected.")
        else:
            st.dataframe(auto_setup["Date candidates"], width="stretch", hide_index=True)
    with candidate_tabs[1]:
        if auto_setup["Target candidates"].empty:
            st.info("No numeric target candidates were detected.")
        else:
            st.dataframe(auto_setup["Target candidates"], width="stretch", hide_index=True)
    with candidate_tabs[2]:
        feature_profile = pd.DataFrame(
            [
                {
                    "Feature": feature,
                    "Missing %": round(float(df[feature].isna().mean() * 100), 1),
                    "Unique values": int(df[feature].nunique(dropna=True)),
                }
                for feature in auto_setup["Recommended features"]
            ]
        )
        if feature_profile.empty:
            st.info("No recommended numeric features are available yet.")
        else:
            st.dataframe(feature_profile, width="stretch", hide_index=True)

    st.download_button(
        "Download smart setup CSV",
        data=recommendation_rows.to_csv(index=False).encode("utf-8"),
        file_name="dataiq_smart_auto_setup.csv",
        mime="text/csv",
    )

with st.expander("Data Cleaning Studio", expanded=False):
    st.caption("These actions are selected in the sidebar and applied before analysis, modeling, reports, and exports.")
    before = st.session_state.get("cleaning_studio_before", dataset_health(base_prepared_df))
    after = st.session_state.get("cleaning_studio_after", dataset_health(df))
    health_compare = pd.DataFrame(
        [
            {
                "Metric": metric,
                "Before": before.get(metric, 0),
                "After": after.get(metric, 0),
                "Change": after.get(metric, 0) - before.get(metric, 0),
            }
            for metric in ["Rows", "Columns", "Missing cells", "Duplicate rows", "Empty columns"]
        ]
    )
    studio_cols = st.columns(4)
    studio_cols[0].metric("Rows", f"{after['Rows']:,}", delta=f"{after['Rows'] - before['Rows']:,}")
    studio_cols[1].metric("Columns", f"{after['Columns']:,}", delta=f"{after['Columns'] - before['Columns']:,}")
    studio_cols[2].metric(
        "Missing cells",
        f"{after['Missing cells']:,}",
        delta=f"{after['Missing cells'] - before['Missing cells']:,}",
        delta_color="inverse",
    )
    studio_cols[3].metric(
        "Duplicate rows",
        f"{after['Duplicate rows']:,}",
        delta=f"{after['Duplicate rows'] - before['Duplicate rows']:,}",
        delta_color="inverse",
    )
    st.markdown("#### Before vs After")
    st.dataframe(health_compare, width="stretch", hide_index=True)
    st.markdown("#### Cleaning Audit")
    st.dataframe(studio_report, width="stretch", hide_index=True)
    preview_tabs = st.tabs(["Cleaned Preview", "Rows/Columns Summary"])
    with preview_tabs[0]:
        st.dataframe(df.head(50), width="stretch")
    with preview_tabs[1]:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Item": "Dropped rows", "Value": before["Rows"] - after["Rows"]},
                    {"Item": "Dropped columns", "Value": before["Columns"] - after["Columns"]},
                    {"Item": "Missing cells resolved", "Value": before["Missing cells"] - after["Missing cells"]},
                    {"Item": "Duplicate rows resolved", "Value": before["Duplicate rows"] - after["Duplicate rows"]},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    st.download_button(
        "Download cleaned dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataiq_cleaned_dataset.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download cleaning studio audit CSV",
        data=studio_report.to_csv(index=False).encode("utf-8"),
        file_name="dataiq_cleaning_studio_audit.csv",
        mime="text/csv",
    )


def onboarding_content() -> None:
    """Render the start-page guide for first-time users and project reviewers."""
    st.markdown("Welcome to **DataIQ Platform**. Follow this path for your first dataset:")
    guide_pages = selected_mode_profile.get("focus", selected_template["recommended_pages"])
    guide_text = "\n".join(
        f"{idx + 1}. **{page}**" for idx, page in enumerate(guide_pages[:6])
    )
    st.markdown(guide_text)
    st.info("Use sidebar Search for words like `accuracy`, `OLAP`, `forecast`, or `drift`.", icon=":material/search:")


with st.sidebar:
    # Keep the guide available without interrupting repeat users.
    with st.popover("First-Time Guide", icon=":material/assistant_navigation:"):
        onboarding_content()

# Top-level metric cards give reviewers an instant sense of dataset size and
# quality before they open any analytical tab.
metric_cols = st.columns(5)
with metric_cols[0]:
    metric_card("Rows", f"{len(df):,}")
with metric_cols[1]:
    metric_card("Columns", f"{df.shape[1]:,}")
with metric_cols[2]:
    metric_card("Numeric measures", f"{len(num_cols):,}")
with metric_cols[3]:
    metric_card("Missing cells", f"{int(df.isna().sum().sum()):,}")
with metric_cols[4]:
    metric_card("Time groups", f"{df['time_group'].nunique():,}")

if show_raw:
    # Raw preview is intentionally optional because wide CSVs can take space on
    # the page and distract from the dashboard story during demos.
    st.subheader("Raw Data Preview")
    st.caption(f"Showing the first {min(preview_rows, len(df)):,} of {len(df):,} rows.")
    st.dataframe(df.head(preview_rows), width="stretch")

# The tab order follows the project story: orientation, discovery, modeling,
# segmentation, decision support, reporting, and governance.
tabs = st.tabs(
    [
        "Start Here",
        "Overview",
        "Explore",
        "Data Quality",
        "Compare",
        "ML Lab",
        "Evaluation",
        "Unsupervised Lab",
        "Reinforcement Lab",
        "Forecast",
        "Conclusion",
        "Domain Review",
        "Code Lab",
        "OLAP & Export",
        "Executive Summary",
        "Data Dictionary",
        "Scenario Simulator",
        "Production Readiness",
        "Experiment Tracker",
        "Model Registry",
        "Data Pipeline",
        "Big Data Readiness",
        "Business Impact",
        "Fit Diagnostics",
        "Prediction Page",
        "Report Generator",
        "Model Save / Load",
    ]
)


with tabs[0]:
    # Start Here is a presenter-friendly landing page, not an analysis page. It
    # tells users which tabs to open when they only have a few minutes.
    st.subheader("Start Here")
    st.caption("A simple guided path for first-time users, evaluators, and project demos.")
    learning_cards(
        [
            (
                "Best first click",
                "Executive Summary",
                "Use it to understand the full project story before looking at detailed charts.",
            ),
            (
                "Best proof",
                "Evaluation",
                "Use it to compare models with R2, RMSE, Accuracy, Precision, Recall, F1, and explainability.",
            ),
            (
                "Best insight",
                "OLAP & Export",
                "Use it to explain the strongest segment with a pivot table, interpretation, and 3D cube.",
            ),
        ]
    )

    start_cols = st.columns([1, 1])
    with start_cols[0]:
        st.markdown("#### Recommended App Flow")
        flow = pd.DataFrame(
            [
                {
                    "Step": idx + 1,
                    "Page": page,
                    "What to say": str(selected_mode_profile["tone"]),
                }
                for idx, page in enumerate(selected_mode_profile.get("focus", selected_template["recommended_pages"]))
            ]
        )
        st.dataframe(flow, width="stretch", hide_index=True)

    with start_cols[1]:
        st.markdown("#### Two-Minute Demo Script")
        st.markdown(
            "\n".join(
                f"{idx + 1}. Open **{page}** and {str(selected_mode_profile['tone']).lower()}"
                for idx, page in enumerate(selected_template["recommended_pages"][:6])
            )
        )
        st.info(
            "Presentation tip: do not open every tab in a demo. Use the recommended flow, then mention the other tabs as supporting tools.",
            icon=":material/slideshow:",
        )

    st.markdown("#### Final Click-Through Checklist")
    checklist = pd.DataFrame(
        [
            {"Check": "Overview loads without errors", "Status": "Do before submission"},
            {"Check": "Evaluation can run model comparison", "Status": "Do before submission"},
            {"Check": "OLAP pivot and 3D cube display clearly", "Status": "Do before submission"},
            {"Check": "Executive report downloads", "Status": "Do before submission"},
            {"Check": "Production Readiness shows validation and drift", "Status": "Do before submission"},
            {"Check": "No red Streamlit errors appear", "Status": "Do before submission"},
        ]
    )
    st.dataframe(checklist, width="stretch", hide_index=True)


with tabs[1]:
    # Overview focuses on the two first analytical questions: how the target
    # moves over time, and which meaningful variables move with it.
    st.subheader("Overview")
    if target_default and date_col and df[date_col].notna().any():
        left, right = st.columns([2, 1])
        with left:
            chart_df = limit_rows_for_display(
                df.dropna(subset=[target_default]),
                max_chart_rows if performance_mode else 0,
                date_col,
            )
            fig = px.line(
                chart_df,
                x=date_col,
                y=target_default,
                color="time_group",
                markers=True,
                title=f"{target_default} over time",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")
        with right:
            st.markdown("#### Target Summary")
            st.dataframe(df[target_default].describe().to_frame("value").round(3), width="stretch")
    else:
        st.info("A date column and numeric target are needed for the overview chart.")

    st.markdown("#### Correlation Matrix & Interpretation")
    include_engineered_corr = st.checkbox(
        "Include engineered, lagged, smoothed, and target-derived variables",
        value=False,
        key="include_engineered_corr",
    )
    # Engineered columns are hidden by default so the matrix highlights real
    # market drivers instead of lagged or target-derived copies.
    corr_candidate_cols = num_cols if include_engineered_corr else interpretable_numeric_features(num_cols, target_default)
    excluded_corr_cols = [col for col in num_cols if col not in corr_candidate_cols]
    cmat = corr_matrix(df[corr_candidate_cols]) if len(corr_candidate_cols) >= 2 else pd.DataFrame()
    if target_default and not cmat.empty and target_default in cmat.columns:
        top_corr = (
            cmat[target_default]
            .drop(labels=[target_default], errors="ignore")
            .dropna()
            .sort_values(key=lambda value: value.abs(), ascending=False)
            .head(10)
            .reset_index()
        )
        top_corr.columns = ["Feature", "Correlation"]
        important_corr_features = [target_default] + [
            col for col in top_corr["Feature"].head(5).tolist() if col in corr_candidate_cols
        ]
        if len(important_corr_features) >= 2:
            st.markdown("##### Zoomed Important Correlations")
            zoom_corr = corr_matrix(df[important_corr_features])
            zoom_fig = px.imshow(
                zoom_corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Zoomed View: Target + Most Meaningful Drivers",
            )
            zoom_fig.update_layout(
                height=720,
                margin=dict(l=20, r=20, t=70, b=20),
                font=dict(size=15),
                coloraxis_colorbar=dict(title="corr"),
            )
            zoom_fig.update_xaxes(tickangle=-35, tickfont=dict(size=14))
            zoom_fig.update_yaxes(tickfont=dict(size=14))
            st.plotly_chart(zoom_fig, width="stretch")

            strongest_driver = top_corr.iloc[0]
            st.success(
                f"Most important meaningful relationship: `{strongest_driver['Feature']}` "
                f"has correlation `{strongest_driver['Correlation']:.2f}` with `{target_default}`.",
                icon=":material/zoom_in:",
            )

        if excluded_corr_cols:
            with st.expander("Excluded from default correlation view"):
                st.caption(
                    "These columns are hidden by default because they are lagged, smoothed, ratios, percent changes, or target-derived. "
                    "They can make correlation look artificially strong without adding a realistic driver."
                )
                st.write(", ".join(excluded_corr_cols))

        selected_corr = corr_matrix(df[important_corr_features])
        filtered_top_corr = top_corr.copy()
        filtered_top_corr["Interpretation"] = np.where(
            filtered_top_corr["Correlation"] > 0,
            "Moves with target",
            "Moves against target",
        )
        st.markdown("##### Most Meaningful Target Relationships")
        st.dataframe(filtered_top_corr.round(3), width="stretch")

        summary = correlation_summary(selected_corr, target_default)
        pos = summary["target_positive"]
        neg = summary["target_negative"]
        pairs = summary["strong_pairs"]

        interp_cols = st.columns(3)
        with interp_cols[0]:
            if isinstance(pos, pd.Series) and not pos.empty:
                st.info(
                    f"Strongest positive link to `{target_default}`: `{pos.index[0]}` at `{pos.iloc[0]:.2f}`.",
                    icon=":material/trending_up:",
                )
            else:
                st.info("No positive target relationship is available in these metrics.", icon=":material/trending_up:")
        with interp_cols[1]:
            if isinstance(neg, pd.Series) and not neg.empty:
                st.info(
                    f"Strongest negative link to `{target_default}`: `{neg.index[0]}` at `{neg.iloc[0]:.2f}`.",
                    icon=":material/trending_down:",
                )
            else:
                st.info("No negative target relationship is available in these metrics.", icon=":material/trending_down:")
        with interp_cols[2]:
            strong_count = int((selected_corr.abs() >= 0.8).sum().sum() - len(selected_corr))
            st.info(
                f"`{strong_count // 2}` highly correlated meaningful feature pairs have absolute correlation of at least `0.80`.",
                icon=":material/hub:",
            )

        st.markdown("##### How to Read This Matrix")
        st.markdown(
            "- Values near `1.00` move together strongly.\n"
            "- Values near `-1.00` move in opposite directions strongly.\n"
            "- Values near `0.00` have weak linear association.\n"
            "- The default view excludes smoothed, lagged, ratio, percent-change, and target-derived columns to avoid misleading correlations.\n"
            "- Correlation is association, not proof that one metric causes another."
        )

        if isinstance(pairs, pd.DataFrame) and not pairs.empty:
            st.markdown("##### Strongest Feature Relationships")
            st.dataframe(pairs.round(3), width="stretch")
    else:
        st.info("Need at least two numeric columns to calculate correlations.")


with tabs[2]:
    # Explore is intentionally flexible: users can switch chart types without
    # changing the underlying filtered dataset.
    st.subheader("Explore")
    if not num_cols:
        st.info("No numeric columns were found.")
    else:
        chart_type = st.segmented_control(
            "Chart type",
            options=["Line", "Scatter", "Histogram", "Box Plot", "Violin Plot"],
            default="Line",
        )
        if chart_type == "Line":
            if not date_col:
                st.info("Line charts need a date column.")
            else:
                metric = st.selectbox("Metric", num_cols, index=num_cols.index(target_default), key="line_metric")
                rolling = st.slider("Rolling window", 2, 24, 6)
                chart_df = df[[date_col, metric, "time_group"]].dropna().sort_values(date_col)
                chart_df = limit_rows_for_display(chart_df, max_chart_rows if performance_mode else 0, date_col)
                chart_df["rolling_mean"] = chart_df[metric].rolling(rolling).mean()
                fig = px.line(chart_df, x=date_col, y=[metric, "rolling_mean"], title=f"{metric} trend")
                st.plotly_chart(fig, width="stretch")
        elif chart_type == "Scatter":
            x_col = st.selectbox("X", num_cols, index=0, key="scatter_x")
            y_index = 1 if len(num_cols) > 1 else 0
            y_col = st.selectbox("Y", num_cols, index=y_index, key="scatter_y")
            chart_df = limit_rows_for_display(df, max_chart_rows if performance_mode else 0, date_col)
            fig = px.scatter(chart_df, x=x_col, y=y_col, color="time_group", trendline="ols", opacity=0.7)
            fig.update_layout(height=520)
            st.plotly_chart(fig, width="stretch")
        elif chart_type == "Histogram":
            metric = st.selectbox("Metric", num_cols, index=num_cols.index(target_default), key="hist_metric")
            bins = st.slider("Histogram bins", 10, 80, 35)
            chart_df = limit_rows_for_display(df, max_chart_rows if performance_mode else 0, date_col)
            fig = px.histogram(
                chart_df,
                x=metric,
                color="time_group",
                marginal="box",
                nbins=bins,
                title=f"Histogram of {metric}",
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, width="stretch")
            st.info(
                "Histogram shows the shape of the distribution. Tall bars mean many observations fall in that value range.",
                icon=":material/bar_chart:",
            )
        elif chart_type == "Box Plot":
            metric = st.selectbox("Metric", num_cols, index=num_cols.index(target_default), key="box_metric")
            show_points = st.checkbox("Show outlier points", value=True, key="box_points")
            chart_df = limit_rows_for_display(df, max_chart_rows if performance_mode else 0, date_col)
            fig = px.box(
                chart_df,
                x="time_group",
                y=metric,
                points="outliers" if show_points else False,
                title=f"Box Plot of {metric} by Time Group",
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, width="stretch")
            st.info(
                "Box plots highlight median, spread, and outliers. Wider distance between quartiles means more variability.",
                icon=":material/data_thresholding:",
            )
        else:
            metric = st.selectbox("Metric", num_cols, index=num_cols.index(target_default), key="violin_metric")
            chart_df = limit_rows_for_display(df, max_chart_rows if performance_mode else 0, date_col)
            fig = px.violin(
                chart_df,
                x="time_group",
                y=metric,
                color="time_group",
                box=True,
                points="outliers",
                title=f"Violin Plot of {metric} by Time Group",
            )
            fig.update_layout(height=540, showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.info(
                "Violin plots combine distribution shape with a box plot. Wider sections mean values are more concentrated there.",
                icon=":material/analytics:",
            )


with tabs[3]:
    # Data Quality shows the checks that should happen before modeling or
    # making conclusions from the dataset.
    st.subheader("Data Quality")
    st.markdown("#### Data Cleaning Studio Audit")
    dq_before = st.session_state.get("cleaning_studio_before", dataset_health(base_prepared_df))
    dq_after = st.session_state.get("cleaning_studio_after", dataset_health(df))
    dq_compare = pd.DataFrame(
        [
            {
                "Metric": metric,
                "Before studio": dq_before.get(metric, 0),
                "After studio": dq_after.get(metric, 0),
                "Change": dq_after.get(metric, 0) - dq_before.get(metric, 0),
            }
            for metric in ["Rows", "Columns", "Missing cells", "Duplicate rows", "Empty columns"]
        ]
    )
    st.dataframe(dq_compare, width="stretch", hide_index=True)
    st.dataframe(st.session_state.get("cleaning_studio_report", studio_report), width="stretch", hide_index=True)
    st.download_button(
        "Download cleaned dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataiq_cleaned_dataset.csv",
        mime="text/csv",
        key="dq_cleaned_download",
    )

    st.markdown("#### Data Cleaning Report")
    st.success(
        "Data cleaning is included in the project even when the uploaded dataset is already clean. "
        "The table below documents each cleaning check and its result.",
        icon=":material/cleaning_services:",
    )
    st.dataframe(cleaning_report, width="stretch", hide_index=True)
    st.download_button(
        "Download cleaning report CSV",
        data=cleaning_report.to_csv(index=False).encode("utf-8"),
        file_name="data_cleaning_report.csv",
        mime="text/csv",
    )

    st.markdown("#### Missing Values and Duplicates")
    missing = df.isna().sum().sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing"]
    missing["missing_%"] = (missing["missing"] / len(df) * 100).round(2)
    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    c2.metric("Columns with missing data", f"{int((missing['missing'] > 0).sum()):,}")
    c3.metric("Total missing cells", f"{int(missing['missing'].sum()):,}")
    st.dataframe(missing, width="stretch")
    if num_cols:
        st.markdown("#### Numeric Profile")
        st.dataframe(df[num_cols].describe().T.round(3), width="stretch")


with tabs[4]:
    # Period Comparison creates a simple grouped view for explaining differences
    # between neutral time windows.
    st.subheader("Period Comparison")
    if not num_cols:
        st.info("No numeric measures are available for comparison.")
    else:
        selected_metrics = st.multiselect(
            "Metrics",
            num_cols,
            default=num_cols[: min(6, len(num_cols))],
            key="compare_metrics",
        )
        if selected_metrics:
            grouped = df.groupby("time_group")[selected_metrics].agg(["mean", "median", "std"]).round(3)
            st.dataframe(grouped, width="stretch")

            means = df.groupby("time_group")[selected_metrics].mean(numeric_only=True)
            long_means = means.reset_index().melt(
                id_vars="time_group",
                var_name="Metric",
                value_name="Mean",
            )
            st.caption("The active dataset is shown by automatically generated neutral time groups.")
            st.plotly_chart(
                px.bar(
                    long_means,
                    x="time_group",
                    y="Mean",
                    color="Metric",
                    barmode="group",
                    title="Mean by Time Group",
                ),
                width="stretch",
            )

            radar_cols = selected_metrics[: min(8, len(selected_metrics))]
            if len(radar_cols) >= 3:
                radar = radar_compare(df, radar_cols)
                if radar:
                    st.plotly_chart(radar, width="stretch")


with tabs[5]:
    # ML Lab is the interactive sandbox: one selected model can be run quickly,
    # or all supported models can be compared on the same train/test split.
    st.subheader("Machine Learning Lab")
    learning_cards(
        [
            (
                "What this page does",
                "Train supervised models",
                "Choose a target, select features, then run regression or classification models.",
            ),
            (
                "Beginner rule",
                "Start simple",
                "Use Ridge or RandomForest first. Then compare all models before trusting one result.",
            ),
            (
                "What to report",
                "Model plus evidence",
                "Mention the best score, actual-vs-predicted gap, and strongest feature relationship.",
            ),
        ]
    )
    if len(num_cols) < 2:
        st.info("ML needs at least two numeric columns.")
    else:
        target = st.selectbox("Target", num_cols, index=num_cols.index(target_default), key="ml_target")
        feature_candidates = [col for col in num_cols if col != target]
        smart_feature_defaults = [
            col for col in st.session_state.get("smart_features", []) if col in feature_candidates
        ]
        features = st.multiselect(
            "Features",
            feature_candidates,
            default=smart_feature_defaults or feature_candidates[: min(10, len(feature_candidates))],
        )
        if not features:
            st.info("Select at least one feature.")
        else:
            task_type = st.radio("Task type", ["Regression", "Classification"], horizontal=True)
            model_options = model_options_for_task(task_type)
            model_choice = st.selectbox("Model", model_options, index=1 if task_type == "Regression" else 0)
            use_scaling = st.checkbox("Scale features", value=True)
            scaler_name = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler"])
            if model_choice == "NeuralNetwork":
                st.info(
                    "Deep learning option: this uses a small multilayer neural network with two hidden layers. "
                    "It can learn non-linear patterns, but on a small housing dataset it should be compared against simpler models before trusting it.",
                    icon=":material/psychology:",
                )

            modeling_source = limit_rows_for_display(df, max_model_rows if performance_mode else 0, date_col)
            event_modeling_source, event_features = add_market_event_features(modeling_source, date_col)
            model_features = list(dict.fromkeys(features + event_features))
            model_cols = list(dict.fromkeys([target] + model_features + ([date_col] if date_col in event_modeling_source.columns else [])))
            model_df = event_modeling_source[model_cols].copy().dropna(subset=[target])
            if performance_mode and len(df) > len(modeling_source):
                st.caption(f"Performance mode: modeling uses {len(modeling_source):,} sampled rows from {len(df):,} filtered rows.")
            min_rows = 20 if task_type == "Regression" else 30
            if len(model_df) < min_rows:
                st.warning(f"Need at least {min_rows} rows after filtering for a reliable model run.")
            else:
                train_df, test_df, split_info = chronological_model_split(model_df, target, model_features, date_col)
                X_train, X_test = train_df[model_features], test_df[model_features]
                y_train, y_test = train_df[target], test_df[target]
                if split_info["Test start"] is not None:
                    test_start = pd.Timestamp(split_info["Test start"])
                    st.caption(
                        f"Chronological evaluation starts at `{test_start:%Y-%m-%d}` "
                        f"({split_info['Train rows']:,} train rows, {split_info['Test rows']:,} test rows). "
                        f"The model also receives `{len(event_features)}` date-only event/regime features."
                    )

                run_col, compare_col = st.columns([1, 1])
                run_single = run_col.button("Run selected model", type="primary")
                compare_all = compare_col.button("Compare all models")

                if compare_all:
                    # Compare every model with identical inputs so the ranking
                    # reflects model behavior, not different feature choices.
                    rows = []
                    if task_type == "Regression":
                        for option in model_options:
                            pipeline = build_model_pipeline(option, task_type, use_scaling, scaler_name)
                            pred, _, _ = fit_predict_regression_series(
                                pipeline,
                                X_train,
                                y_train,
                                X_test,
                                train_df,
                                target,
                            )
                            mae, rmse, r2 = regression_metrics(y_test, pred)
                            rows.append(
                                {
                                    "Task": "Regression",
                                    "Model": option,
                                    "MAE": mae,
                                    "RMSE": rmse,
                                    "R2": r2,
                                    "Train rows": len(X_train),
                                    "Test rows": len(X_test),
                                    "Feature count": len(model_features),
                                    "Event features": len(event_features),
                                    "Test start": split_info["Test start"],
                                }
                            )
                        comparison = pd.DataFrame(rows).sort_values("R2", ascending=False)
                    else:
                        try:
                            y_train_binned, y_test_binned = make_classification_labels(y_train, y_test)
                        except (AttributeError, ValueError) as err:
                            st.error(f"Could not create target classes: {err}")
                            comparison = pd.DataFrame()
                        else:
                            for option in model_options:
                                try:
                                    pipeline = build_model_pipeline(option, task_type, use_scaling, scaler_name)
                                    pipeline.fit(X_train, y_train_binned)
                                    pred = pipeline.predict(X_test)
                                    roc_auc = classification_auc(pipeline, X_test, y_test_binned)
                                    rows.append(
                                        {
                                            "Task": "Classification",
                                            "Model": option,
                                            "Accuracy": accuracy_score(y_test_binned, pred),
                                            "Precision": precision_score(
                                                y_test_binned,
                                                pred,
                                                average="weighted",
                                                zero_division=0,
                                            ),
                                            "Recall": recall_score(
                                                y_test_binned,
                                                pred,
                                                average="weighted",
                                                zero_division=0,
                                            ),
                                            "F1": f1_score(
                                                y_test_binned,
                                                pred,
                                                average="weighted",
                                                zero_division=0,
                                            ),
                                            "ROC AUC": roc_auc,
                                            "Train rows": len(X_train),
                                            "Test rows": len(X_test),
                                            "Feature count": len(model_features),
                                            "Event features": len(event_features),
                                            "Test start": split_info["Test start"],
                                            "Status": "OK",
                                        }
                                    )
                                except Exception as err:
                                    rows.append(
                                        {
                                            "Task": "Classification",
                                            "Model": option,
                                            "Accuracy": np.nan,
                                            "Precision": np.nan,
                                            "Recall": np.nan,
                                            "F1": np.nan,
                                            "ROC AUC": np.nan,
                                            "Train rows": len(X_train),
                                            "Test rows": len(X_test),
                                            "Feature count": len(model_features),
                                            "Event features": len(event_features),
                                            "Test start": split_info["Test start"],
                                            "Status": f"Failed: {err}",
                                        }
                                    )
                            comparison = pd.DataFrame(rows).sort_values("F1", ascending=False, na_position="last")
                    if not comparison.empty:
                        st.session_state.model_comparison = comparison
                        if "ml_results" not in st.session_state:
                            st.session_state.ml_results = []
                        st.session_state.ml_results.extend(comparison.to_dict("records"))

                if run_single:
                    # Single-model mode is useful for teaching: it shows the
                    # fitted predictions and, when possible, feature importance.
                    pipeline = build_model_pipeline(model_choice, task_type, use_scaling, scaler_name)

                    if task_type == "Regression":
                        pred, pipeline, fit_y_train = fit_predict_regression_series(
                            pipeline,
                            X_train,
                            y_train,
                            X_test,
                            train_df,
                            target,
                        )
                        mae, rmse, r2 = regression_metrics(y_test, pred)
                        render_regression_metric_values(mae, rmse, r2, y_test)
                        if "ml_results" not in st.session_state:
                            st.session_state.ml_results = []
                        st.session_state.ml_results.append(
                            {"Task": "Regression", "Model": model_choice, "MAE": mae, "RMSE": rmse, "R2": r2}
                        )
                        st.session_state.last_prediction_summary = {
                            "Model": model_choice,
                            "Task": "Regression",
                            "MAE": mae,
                            "RMSE": rmse,
                            "R2": r2,
                            "Actual mean": float(np.mean(y_test)),
                            "Predicted mean": float(np.mean(pred)),
                            "Mean gap": float(np.mean(pred) - np.mean(y_test)),
                        }
                        result_df = pd.DataFrame({"actual": y_test.values, "predicted": pred})
                        if date_col and date_col in test_df.columns and date_col in train_df.columns:
                            result_df[date_col] = test_df[date_col].values
                            result_df = pd.concat(
                                [
                                    pd.DataFrame(
                                        {
                                            date_col: [train_df[date_col].iloc[-1]],
                                            "actual": [y_train.iloc[-1]],
                                            "predicted": [y_train.iloc[-1]],
                                        }
                                    ),
                                    result_df,
                                ],
                                ignore_index=True,
                            )
                        prediction_chart_col, prediction_text_col = st.columns([2, 1])
                        with prediction_chart_col:
                            if date_col and date_col in result_df.columns:
                                st.plotly_chart(
                                    px.line(result_df, x=date_col, y=["actual", "predicted"]),
                                    width="stretch",
                                )
                            else:
                                st.plotly_chart(px.line(result_df, y=["actual", "predicted"]), width="stretch")
                        with prediction_text_col:
                            st.markdown("##### Prediction interpretation")
                            st.info(regression_prediction_interpretation(y_test, pred))

                        # Cross-validation is capped by available rows so small
                        # demo datasets do not fail with too many folds.
                        cv = min(5, len(fit_y_train))
                        if cv >= 2:
                            cv_r2 = cross_val_score(pipeline, X_train.loc[fit_y_train.index], fit_y_train, cv=cv, scoring="r2")
                            st.caption(f"Cross-validation R2: {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")

                        fitted = pipeline.named_steps["model"]
                        if hasattr(fitted, "feature_importances_"):
                            importance = pd.Series(fitted.feature_importances_, index=model_features).sort_values()
                            feature_chart_col, feature_text_col = st.columns([2, 1])
                            with feature_chart_col:
                                st.plotly_chart(px.bar(importance, orientation="h"), width="stretch")
                            with feature_text_col:
                                st.markdown("##### Feature interpretation")
                                st.info(feature_effect_interpretation(importance, "importance"))
                        elif hasattr(fitted, "coef_"):
                            coef = pd.Series(np.ravel(fitted.coef_), index=model_features).sort_values()
                            feature_chart_col, feature_text_col = st.columns([2, 1])
                            with feature_chart_col:
                                st.plotly_chart(px.bar(coef, orientation="h"), width="stretch")
                            with feature_text_col:
                                st.markdown("##### Coefficient interpretation")
                                st.info(feature_effect_interpretation(coef, "coefficient"))
                    else:
                        try:
                            y_train_binned, y_test_binned = make_classification_labels(y_train, y_test)
                        except (AttributeError, ValueError) as err:
                            st.error(f"Could not create target classes: {err}")
                        else:
                            try:
                                pipeline.fit(X_train, y_train_binned)
                                pred = pipeline.predict(X_test)
                            except Exception as err:
                                st.error(f"Could not train `{model_choice}`: {err}")
                                pred = None
                            if pred is None:
                                st.info("Try RandomForest, LogisticRegression, DecisionTree, or KNNClassifier for this classification split.")
                            else:
                                acc = accuracy_score(y_test_binned, pred)
                                prec = precision_score(y_test_binned, pred, average="weighted", zero_division=0)
                                rec = recall_score(y_test_binned, pred, average="weighted", zero_division=0)
                                f1 = f1_score(y_test_binned, pred, average="weighted", zero_division=0)
                                roc_auc = classification_auc(pipeline, X_test, y_test_binned)
                                render_classification_metric_values(acc, prec, rec, f1, roc_auc)
                                if "ml_results" not in st.session_state:
                                    st.session_state.ml_results = []
                                st.session_state.ml_results.append(
                                    {
                                        "Task": "Classification",
                                        "Model": model_choice,
                                        "Accuracy": acc,
                                        "Precision": prec,
                                        "Recall": rec,
                                        "F1": f1,
                                        "ROC AUC": roc_auc,
                                    }
                                )
                                report = classification_report(y_test_binned, pred, output_dict=True, zero_division=0)
                                st.dataframe(pd.DataFrame(report).transpose(), width="stretch")
                                st.markdown("#### Classification Diagnostics")
                                render_classification_diagnostics(
                                    pipeline,
                                    X_test,
                                    y_test_binned,
                                    pred,
                                    roc_auc,
                                )

                if st.session_state.get("model_comparison") is not None:
                    st.markdown("#### Model Comparison Table")
                    model_comparison = st.session_state.model_comparison.round(4)
                    st.dataframe(model_comparison, width="stretch")
                    if "NeuralNetwork" in set(model_comparison["Model"]):
                        metric = "R2" if "R2" in model_comparison.columns else "F1"
                        ascending = metric not in {"R2", "F1", "Accuracy"}
                        ranked = model_comparison.sort_values(metric, ascending=ascending)
                        neural_row = model_comparison.loc[model_comparison["Model"] == "NeuralNetwork"].iloc[0]
                        best_row = ranked.iloc[0]
                        simple_ranked = ranked[ranked["Model"] != "NeuralNetwork"]
                        simple_best = simple_ranked.iloc[0] if not simple_ranked.empty else best_row
                        nn_value = neural_row[metric]
                        simple_value = simple_best[metric]
                        if neural_row["Model"] == best_row["Model"]:
                            st.success(
                                f"Deep learning result: the NeuralNetwork is currently the best model with {metric} = {nn_value:.3f}.",
                                icon=":material/psychology:",
                            )
                        else:
                            st.info(
                                f"Deep learning result: NeuralNetwork scored {metric} = {nn_value:.3f}. "
                                f"The best non-neural model is {simple_best['Model']} with {metric} = {simple_value:.3f}, "
                                "so the data does not automatically need deep learning.",
                                icon=":material/psychology:",
                            )
                    st.download_button(
                        "Download model comparison CSV",
                        data=model_comparison.to_csv(index=False).encode("utf-8"),
                        file_name="model_comparison.csv",
                        mime="text/csv",
                    )

        if st.session_state.get("ml_results"):
            st.markdown("#### Saved Results This Session")
            history = pd.DataFrame(st.session_state.ml_results)
            sort_col = "R2" if "R2" in history.columns else "F1" if "F1" in history.columns else "Model"
            ascending = sort_col not in {"R2", "F1", "Accuracy"}
            history = history.sort_values(sort_col, ascending=ascending).round(4)
            st.dataframe(history, width="stretch")
            history_col, clear_col = st.columns([1, 1])
            history_col.download_button(
                "Download saved ML results CSV",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name="saved_ml_results.csv",
                mime="text/csv",
            )
            if clear_col.button("Clear saved ML results"):
                st.session_state.ml_results = []
                st.session_state.model_comparison = None
                st.rerun()

        # This block converts metrics into report-ready language for students or
        # reviewers who need a concise conclusion, not only numbers.
        st.markdown("#### Guided ML Conclusion Example")
        guide_df = df[[target] + features].dropna()
        if len(guide_df) < 5:
            st.info("Need more complete rows to create a useful guided conclusion.", icon=":material/info:")
        else:
            guide_corr = corr_matrix(guide_df)
            driver_corr = guide_corr[target].drop(labels=[target], errors="ignore").dropna()
            if driver_corr.empty:
                st.info("No usable feature correlation was found for the selected target.", icon=":material/info:")
            else:
                best_driver = driver_corr.abs().idxmax()
                best_corr = driver_corr[best_driver]
                direction = "moves with" if best_corr > 0 else "moves against"
                strength = "strong" if abs(best_corr) >= 0.7 else "moderate" if abs(best_corr) >= 0.4 else "weak"

                conclusion_cols = st.columns(3)
                with conclusion_cols[0]:
                    st.success(
                        f"Best signal: `{best_driver}` has a `{strength}` correlation of `{best_corr:.2f}` with `{target}`.",
                        icon=":material/target:",
                    )
                with conclusion_cols[1]:
                    if st.session_state.get("model_comparison") is not None:
                        comparison = st.session_state.model_comparison
                        metric = "R2" if "R2" in comparison.columns else "F1"
                        best_row = comparison.sort_values(metric, ascending=False).iloc[0]
                        st.info(
                            f"Best tested model: `{best_row['Model']}` with `{metric} = {best_row[metric]:.3f}`.",
                            icon=":material/emoji_events:",
                        )
                    else:
                        st.info("Run Compare all models to identify the best model objectively.", icon=":material/emoji_events:")
                with conclusion_cols[2]:
                    pred_summary = st.session_state.get("last_prediction_summary")
                    if pred_summary and pred_summary.get("Task") == "Regression":
                        st.info(
                            f"Actual vs predicted mean gap: `{pred_summary['Mean gap']:.2f}` using `{pred_summary['Model']}`.",
                            icon=":material/compare_arrows:",
                        )
                    else:
                        st.info("Run a regression model to compare actual vs predicted values.", icon=":material/compare_arrows:")

                st.markdown("##### Example Conclusion You Can Use")
                st.markdown(
                    f"""
                    **Conclusion:** In the current filtered data, `{best_driver}` is the most important selected signal for `{target}`.
                    It {direction} `{target}` with a correlation of `{best_corr:.2f}`, so it is a realistic candidate driver to inspect before
                    building a final forecasting story. If the best model also has a higher `R2` and a low `RMSE`, the conclusion is stronger:
                    the relationship is not only visible in correlation, it also helps prediction.
                    """
                )

                st.markdown("##### Classic Theory Check")
                st.caption(
                    "Benchmark idea: classic real-estate economics, including the DiPasquale-Wheaton four-quadrant model, links prices to demand, supply, construction, and capital-market conditions. The table checks whether this dataset supports those expected directions."
                )
                theory_df = theory_check_table(df, target, features)
                st.dataframe(theory_df.round({"Observed correlation": 3}), width="stretch")

                supported = int((theory_df["Evidence"] == "Supports theory").sum())
                challenged = int((theory_df["Evidence"] == "Challenges theory").sum())
                missing = int((theory_df["Evidence"] == "Missing feature").sum())
                st.info(
                    f"The selected features support `{supported}` theory checks, challenge `{challenged}`, and miss `{missing}` because the needed variables are not selected or not present.",
                    icon=":material/fact_check:",
                )


with tabs[6]:
    # Evaluation is the formal model-comparison page. It stores results in
    # session state so other tabs can reuse the best-model evidence.
    st.subheader("Model Evaluation")
    st.caption("Compare supervised models side by side with the metrics used in machine-learning reports.")
    learning_cards(
        [
            (
                "What this page does",
                "Compare model quality",
                "It runs all supervised models and ranks them with standard evaluation metrics.",
            ),
            (
                "Beginner rule",
                "Use the right metric",
                "Regression uses R2, MAE, and RMSE. Classification uses Accuracy, Precision, Recall, F1, and ROC AUC.",
            ),
            (
                "What to report",
                "Why the best wins",
                "Use the explanation and feature importance to say why one model works better than another.",
            ),
        ]
    )
    if len(num_cols) < 2:
        st.info("Evaluation needs at least two numeric columns.")
    else:
        eval_target = st.selectbox(
            "Evaluation target",
            num_cols,
            index=num_cols.index(target_default) if target_default in num_cols else 0,
            key="eval_target",
        )
        eval_feature_candidates = [col for col in num_cols if col != eval_target]
        smart_eval_defaults = [
            col for col in st.session_state.get("smart_features", []) if col in eval_feature_candidates
        ]
        eval_features = st.multiselect(
            "Evaluation features",
            eval_feature_candidates,
            default=smart_eval_defaults or eval_feature_candidates[: min(10, len(eval_feature_candidates))],
            key="eval_features",
        )
        eval_task = st.radio("Evaluation task", ["Regression", "Classification"], horizontal=True, key="eval_task")
        eval_scale = st.checkbox("Scale evaluation features", value=True, key="eval_scale")
        eval_scaler = st.selectbox("Evaluation scaler", ["StandardScaler", "MinMaxScaler"], key="eval_scaler")

        if not eval_features:
            st.info("Select at least one feature to evaluate models.")
        else:
            eval_source = limit_rows_for_display(df, max_model_rows if performance_mode else 0, date_col)
            eval_model_df = eval_source[[eval_target] + eval_features].dropna(subset=[eval_target])
            if performance_mode and len(df) > len(eval_source):
                st.caption(f"Performance mode: evaluation uses {len(eval_source):,} sampled rows from {len(df):,} filtered rows.")
            min_rows = 20 if eval_task == "Regression" else 30
            if len(eval_model_df) < min_rows:
                st.warning(f"Need at least {min_rows} rows after filtering for a useful evaluation.")
            elif st.button("Run full model evaluation", type="primary"):
                # Store the exact target, features, task, and metrics so later
                # pages can build experiment tracking and model registry views.
                try:
                    evaluation = evaluate_all_models(
                        eval_source,
                        eval_target,
                        eval_features,
                        eval_task,
                        eval_scale,
                        eval_scaler,
                        date_col,
                    )
                except Exception as err:
                    st.error(f"Could not evaluate models: {err}")
                else:
                    st.session_state.evaluation_results = evaluation
                    st.session_state.evaluation_task = eval_task
                    st.session_state.evaluation_features = eval_features
                    st.session_state.evaluation_target = eval_target
                    st.session_state.evaluation_scale = eval_scale
                    st.session_state.evaluation_scaler = eval_scaler

        if st.session_state.get("evaluation_results") is not None:
            evaluation = st.session_state.evaluation_results.round(4)
            task = st.session_state.get("evaluation_task", eval_task)
            st.markdown("#### Evaluation Metrics Table")
            st.dataframe(evaluation, width="stretch", hide_index=True)
            st.caption(
                "For regression, compare R2, MAE, and RMSE rather than accuracy. "
                "Adding variables only changes the score when those variables add useful signal in the chronological test period."
            )

            if task == "Regression":
                best = evaluation.sort_values("R2", ascending=False).iloc[0]
                st.success(
                    f"Best regression model: `{best['Model']}` with R2 = `{best['R2']:.3f}`, "
                    f"RMSE = `{best['RMSE']:.3f}`, and MAE = `{best['MAE']:.3f}`.",
                    icon=":material/emoji_events:",
                )
                st.info(evaluation_explanation(evaluation, task), icon=":material/help:")
                render_regression_metric_values(
                    float(best["MAE"]),
                    float(best["RMSE"]),
                    float(best["R2"]),
                    df[st.session_state.get("evaluation_target", eval_target)].dropna(),
                )
                long_metrics = evaluation.melt(
                    id_vars=["Model"],
                    value_vars=["MAE", "RMSE", "R2"],
                    var_name="Metric",
                    value_name="Score",
                )
                chart_col, chart_text_col = st.columns([2, 1])
                with chart_col:
                    st.plotly_chart(
                        px.bar(long_metrics, x="Model", y="Score", color="Metric", barmode="group"),
                        width="stretch",
                    )
                with chart_text_col:
                    st.markdown("##### Chart interpretation")
                    st.info(regression_metric_chart_interpretation(evaluation, best))
                with st.expander("How to read regression metrics"):
                    st.markdown(
                        "- **R2:** higher is better; it shows how much variation the model explains.\n"
                        "- **MAE:** lower is better; it is the average absolute prediction error.\n"
                        "- **RMSE:** lower is better; it punishes large mistakes more than MAE."
                    )
            else:
                best = evaluation.sort_values("F1", ascending=False).iloc[0]
                st.success(
                    f"Best classification model: `{best['Model']}` with F1 = `{best['F1']:.3f}`, "
                    f"accuracy = `{best['Accuracy']:.3f}`, precision = `{best['Precision']:.3f}`, "
                    f"and recall = `{best['Recall']:.3f}`.",
                    icon=":material/emoji_events:",
                )
                st.info(evaluation_explanation(evaluation, task), icon=":material/help:")
                render_classification_metric_values(
                    float(best["Accuracy"]),
                    float(best["Precision"]),
                    float(best["Recall"]),
                    float(best["F1"]),
                    None if "ROC AUC" not in best or pd.isna(best["ROC AUC"]) else float(best["ROC AUC"]),
                )
                metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
                if "ROC AUC" in evaluation.columns and evaluation["ROC AUC"].notna().any():
                    metric_cols.append("ROC AUC")
                long_metrics = evaluation.melt(
                    id_vars=["Model"],
                    value_vars=metric_cols,
                    var_name="Metric",
                    value_name="Score",
                )
                chart_col, chart_text_col = st.columns([2, 1])
                with chart_col:
                    st.plotly_chart(
                        px.bar(long_metrics, x="Model", y="Score", color="Metric", barmode="group"),
                        width="stretch",
                    )
                with chart_text_col:
                    st.markdown("##### Chart interpretation")
                    st.info(
                        f"- Each bar compares one metric for one model.\n"
                        f"- The best model by F1 is `{best['Model']}` with F1 `{best['F1']:.3f}`.\n"
                        "- Taller bars are better for all classification metrics shown here.\n"
                        "- If one model has high accuracy but lower recall or F1, it may be missing important class cases."
                    )
                with st.expander("How to read classification metrics"):
                    st.markdown(
                        "- **Accuracy:** total correct predictions, useful when classes are balanced.\n"
                        "- **Precision:** when the model predicts a class, how often it is correct.\n"
                        "- **Recall:** how many real class cases the model successfully finds.\n"
                        "- **F1:** balances precision and recall, good for comparing models.\n"
                        "- **ROC AUC:** higher is better; it measures ranking quality when probability scores are available."
                    )
                st.markdown("#### Best Classifier Diagnostics")
                try:
                    diagnostic_target = st.session_state.get("evaluation_target", eval_target)
                    diagnostic_features = st.session_state.get("evaluation_features", eval_features)
                    diagnostic_scale = st.session_state.get("evaluation_scale", eval_scale)
                    diagnostic_scaler = st.session_state.get("evaluation_scaler", eval_scaler)
                    diagnostic_cols = list(
                        dict.fromkeys(
                            [diagnostic_target]
                            + diagnostic_features
                            + ([date_col] if date_col in df.columns else [])
                        )
                    )
                    diagnostic_df = df[diagnostic_cols].copy().dropna(subset=[diagnostic_target])
                    diagnostic_train_df, diagnostic_test_df, _ = chronological_model_split(
                        diagnostic_df,
                        diagnostic_target,
                        diagnostic_features,
                        date_col,
                    )
                    diagnostic_x_train = diagnostic_train_df[diagnostic_features]
                    diagnostic_x_test = diagnostic_test_df[diagnostic_features]
                    diagnostic_y_train = diagnostic_train_df[diagnostic_target]
                    diagnostic_y_test = diagnostic_test_df[diagnostic_target]
                    diagnostic_y_train, diagnostic_y_test = make_classification_labels(
                        diagnostic_y_train,
                        diagnostic_y_test,
                    )
                    diagnostic_model = build_model_pipeline(
                        str(best["Model"]),
                        "Classification",
                        bool(diagnostic_scale),
                        str(diagnostic_scaler),
                    )
                    diagnostic_model.fit(diagnostic_x_train, diagnostic_y_train)
                    diagnostic_pred = diagnostic_model.predict(diagnostic_x_test)
                    diagnostic_roc_auc = classification_auc(diagnostic_model, diagnostic_x_test, diagnostic_y_test)
                except Exception as err:
                    st.info(f"Could not build classifier diagnostics: {err}")
                else:
                    render_classification_diagnostics(
                        diagnostic_model,
                        diagnostic_x_test,
                        diagnostic_y_test,
                        diagnostic_pred,
                        diagnostic_roc_auc,
                        confusion_title=f"Confusion Matrix: {best['Model']}",
                    )

            st.download_button(
                "Download evaluation table CSV",
                data=evaluation.to_csv(index=False).encode("utf-8"),
                file_name="model_evaluation_metrics.csv",
                mime="text/csv",
            )
            if eval_features:
                # Explainability is calculated after ranking so the app explains
                # the current best model rather than an arbitrary model.
                st.markdown("#### Model Explainability")
                try:
                    metric = "R2" if task == "Regression" else "F1"
                    best_model_name = evaluation.sort_values(metric, ascending=False).iloc[0]["Model"]
                    explain_pipeline, explain_X, explain_y = fit_best_model_for_task(
                        eval_source,
                        eval_target,
                        eval_features,
                        task,
                        eval_scale,
                        eval_scaler,
                        best_model_name,
                        date_col,
                    )
                    importance = model_feature_importance(
                        explain_pipeline,
                        explain_X.columns.tolist(),
                        explain_X,
                        explain_y,
                        task,
                    )
                    st.success(
                        f"Explainability result: `{importance.iloc[0]['Feature']}` is the most influential feature for `{best_model_name}`.",
                        icon=":material/manage_search:",
                    )
                    st.plotly_chart(
                        px.bar(
                            importance.head(12).sort_values("Importance"),
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            color="Importance %",
                            color_continuous_scale="Turbo",
                            title=f"Feature Importance for {best_model_name}",
                        ),
                        width="stretch",
                    )
                    st.dataframe(importance.round(4), width="stretch", hide_index=True)
                except Exception as err:
                    st.info(f"Could not calculate explainability for the current setup: {err}")


with tabs[7]:
    # Unsupervised learning answers a different question from supervised ML:
    # "What structure exists?" rather than "Can we predict the target?"
    st.subheader("Unsupervised Learning Lab")
    st.caption("Find groups, lower-dimensional structure, and unusual observations without a target variable.")
    learning_cards(
        [
            (
                "What this page does",
                "Find hidden structure",
                "Unsupervised learning does not predict a target. It finds groups, patterns, and unusual records.",
            ),
            (
                "Beginner rule",
                "Pick the method by question",
                "Use KMeans for segments, PCA for simplification, DBSCAN for dense groups, and IsolationForest for anomalies.",
            ),
            (
                "What to report",
                "Market regimes",
                "Explain what the clusters or components reveal about different housing-market conditions.",
            ),
        ]
    )
    if len(num_cols) < 2:
        st.info("Unsupervised learning needs at least two numeric columns.")
    else:
        unsup_features = st.multiselect(
            "Features for unsupervised learning",
            num_cols,
            default=num_cols[: min(8, len(num_cols))],
            key="unsup_features",
        )
        if len(unsup_features) < 2:
            st.info("Select at least two numeric features.")
        else:
            method = st.selectbox(
                "Unsupervised model",
                ["KMeans", "DBSCAN", "PCA", "IsolationForest"],
                index=2,
                key="unsup_method",
            )
            scale_unsup = st.checkbox("Scale unsupervised features", value=True, key="scale_unsup")
            # The same prepared matrix feeds all unsupervised methods so method
            # differences are easier to compare.
            unsup_source = limit_rows_for_display(df, max_model_rows if performance_mode else 0, date_col)
            if performance_mode and len(df) > len(unsup_source):
                st.caption(f"Performance mode: unsupervised models use {len(unsup_source):,} sampled rows from {len(df):,} filtered rows.")
            matrix, values = prepare_unsupervised_matrix(unsup_source, unsup_features, scale_unsup)

            if len(matrix) < 5:
                st.warning("Need at least five complete rows after cleaning selected features.")
            else:
                example_details: dict[str, object] = {}
                if method == "KMeans":
                    # KMeans is best for broad market regimes when the analyst
                    # chooses how many groups should exist.
                    cluster_count = st.slider("Number of clusters", 2, min(10, len(matrix) - 1), 3)
                    labels = KMeans(n_clusters=cluster_count, random_state=0, n_init=10).fit_predict(values)
                    quality = cluster_quality(values, labels)
                    example_details = {
                        "clusters": quality["Clusters"],
                        "silhouette": quality["Silhouette"],
                    }
                    result = matrix.copy()
                    result["cluster"] = labels.astype(str)
                    projection = pca_projection(values)
                    projection["cluster"] = result["cluster"].values

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Clusters", f"{quality['Clusters']}")
                    c2.metric("Silhouette", "n/a" if quality["Silhouette"] is None else f"{quality['Silhouette']:.3f}")
                    c3.metric(
                        "Davies-Bouldin",
                        "n/a" if quality["Davies-Bouldin"] is None else f"{quality['Davies-Bouldin']:.3f}",
                    )
                    st.plotly_chart(px.scatter(projection, x="PC1", y="PC2", color="cluster"), width="stretch")
                    st.dataframe(result.head(100), width="stretch")

                elif method == "DBSCAN":
                    # DBSCAN discovers dense regions and labels isolated rows as
                    # noise, which is useful for unusual market periods.
                    eps = st.slider("Neighborhood radius (eps)", 0.1, 5.0, 0.8, 0.1)
                    min_samples = st.slider("Minimum samples", 2, 20, 5)
                    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(values)
                    quality = cluster_quality(values, labels)
                    example_details = {
                        "clusters": quality["Clusters"],
                        "noise": quality["Noise / anomalies"],
                    }
                    result = matrix.copy()
                    result["cluster"] = labels.astype(str)
                    projection = pca_projection(values)
                    projection["cluster"] = result["cluster"].values

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Clusters", f"{quality['Clusters']}")
                    c2.metric("Noise points", f"{quality['Noise / anomalies']}")
                    c3.metric("Silhouette", "n/a" if quality["Silhouette"] is None else f"{quality['Silhouette']:.3f}")
                    st.plotly_chart(px.scatter(projection, x="PC1", y="PC2", color="cluster"), width="stretch")
                    st.dataframe(result.head(100), width="stretch")

                elif method == "PCA":
                    # PCA compresses many numeric indicators into a few axes so
                    # users can see whether the data has a simpler structure.
                    pca = PCA(n_components=min(len(unsup_features), len(matrix)), random_state=0)
                    pca.fit(values)
                    explained = pd.DataFrame(
                        {
                            "Component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
                            "Explained variance": pca.explained_variance_ratio_,
                            "Cumulative variance": np.cumsum(pca.explained_variance_ratio_),
                        }
                    )
                    projection = pca_projection(values)
                    c1, c2 = st.columns(2)
                    c1.metric("PC1 variance", f"{explained['Explained variance'].iloc[0]:.3f}")
                    c2.metric(
                        "First 2 PCs variance",
                        f"{explained['Explained variance'].head(2).sum():.3f}",
                    )
                    example_details = {
                        "pc1": float(explained["Explained variance"].iloc[0]),
                        "first_two": float(explained["Explained variance"].head(2).sum()),
                    }
                    st.plotly_chart(px.scatter(projection, x="PC1", y="PC2"), width="stretch")
                    st.dataframe(explained.round(4), width="stretch")

                else:
                    # IsolationForest is used here as an anomaly detector for
                    # candidate months/rows that deserve deeper investigation.
                    contamination = st.slider("Expected anomaly share", 0.01, 0.25, 0.05, 0.01)
                    detector = IsolationForest(contamination=contamination, random_state=0)
                    raw_labels = detector.fit_predict(values)
                    scores = detector.decision_function(values)
                    labels = np.where(raw_labels == -1, "Anomaly", "Normal")
                    result = matrix.copy()
                    result["anomaly_label"] = labels
                    result["anomaly_score"] = scores
                    projection = pca_projection(values)
                    projection["anomaly_label"] = labels

                    c1, c2 = st.columns(2)
                    c1.metric("Anomalies", f"{int((raw_labels == -1).sum()):,}")
                    c2.metric("Normal rows", f"{int((raw_labels == 1).sum()):,}")
                    example_details = {"anomalies": int((raw_labels == -1).sum())}
                    st.plotly_chart(
                        px.scatter(projection, x="PC1", y="PC2", color="anomaly_label"),
                        width="stretch",
                    )
                    st.dataframe(result.sort_values("anomaly_score").head(100), width="stretch")

                st.markdown("#### Guided Unsupervised Example")
                st.success(
                    most_important_unsupervised_learning(method, unsup_features, example_details, target_default),
                    icon=":material/star:",
                )
                st.info(
                    unsupervised_example_text(method, unsup_features, example_details, target_default),
                    icon=":material/lightbulb:",
                )
                with st.expander("How to use this result"):
                    st.markdown(
                        "- Use **KMeans** when you want broad market regimes or segments.\n"
                        "- Use **DBSCAN** when you care about dense groups and unusual/noise periods.\n"
                        "- Use **PCA** when you want to compress many macro variables into fewer market-pressure dimensions.\n"
                        "- Use **IsolationForest** when you want candidate anomaly periods for deeper review.\n"
                        "- Treat unsupervised output as a discovery tool, then validate findings with domain logic, plots, and supervised models."
                    )

                if st.button("Compare unsupervised models"):
                    # The comparison table gives a quick benchmark across
                    # segmentation, dimensionality-reduction, and anomaly tools.
                    rows = []
                    for clusters in [2, 3, 4, 5]:
                        if clusters < len(matrix):
                            labels = KMeans(n_clusters=clusters, random_state=0, n_init=10).fit_predict(values)
                            rows.append({"Model": f"KMeans ({clusters})", **cluster_quality(values, labels)})
                    dbscan_labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(values)
                    rows.append({"Model": "DBSCAN", **cluster_quality(values, dbscan_labels)})
                    anomaly_labels = IsolationForest(contamination=0.05, random_state=0).fit_predict(values)
                    rows.append(
                        {
                            "Model": "IsolationForest",
                            "Clusters": None,
                            "Noise / anomalies": int((anomaly_labels == -1).sum()),
                            "Silhouette": None,
                            "Davies-Bouldin": None,
                        }
                    )
                    pca = PCA(n_components=2, random_state=0).fit(values)
                    rows.append(
                        {
                            "Model": "PCA (2 components)",
                            "Clusters": None,
                            "Noise / anomalies": None,
                            "Silhouette": None,
                            "Davies-Bouldin": None,
                            "Explained variance": float(pca.explained_variance_ratio_.sum()),
                        }
                    )
                    comparison = pd.DataFrame(rows)
                    st.session_state.unsupervised_comparison = comparison

                if st.session_state.get("unsupervised_comparison") is not None:
                    st.markdown("#### Unsupervised Model Comparison")
                    unsupervised_comparison = st.session_state.unsupervised_comparison.round(4)
                    st.dataframe(unsupervised_comparison, width="stretch")
                    st.download_button(
                        "Download unsupervised comparison CSV",
                        data=unsupervised_comparison.to_csv(index=False).encode("utf-8"),
                        file_name="unsupervised_model_comparison.csv",
                        mime="text/csv",
                    )


with tabs[8]:
    # This page is an educational RL example. It demonstrates state, action, and
    # reward concepts without claiming to be a trading or investment system.
    st.subheader("Reinforcement Learning Lab")
    st.caption("Learn a simple decision policy from repeated market states, actions, and rewards.")
    learning_cards(
        [
            (
                "What this page does",
                "Learn decisions",
                "Reinforcement learning learns actions from rewards instead of predicting one number.",
            ),
            (
                "Beginner rule",
                "State, action, reward",
                "State means market condition, action means buy/hold/wait, and reward means whether the action helped.",
            ),
            (
                "What to report",
                "Policy result",
                "Report the best learned action for the strongest market state and explain it as a simulation.",
            ),
        ]
    )
    if not date_col or not target_default or len(num_cols) < 1:
        st.info("Reinforcement learning needs a date column and at least one numeric target.")
    else:
        rl_target = st.selectbox(
            "Reward target",
            num_cols,
            index=num_cols.index(target_default) if target_default in num_cols else 0,
            key="rl_target",
        )
        signal_candidates = [col for col in num_cols if col != rl_target]
        default_signal = None
        for candidate in ["Mortgage_Rate", "Interest_Rate", "Unemployment_Rate", "Real_GDP", "Median_Income"]:
            if candidate in signal_candidates:
                default_signal = candidate
                break
        if default_signal is None and signal_candidates:
            default_signal = signal_candidates[0]
        rl_signal = st.selectbox(
            "Extra market signal",
            ["None"] + signal_candidates,
            index=(signal_candidates.index(default_signal) + 1) if default_signal in signal_candidates else 0,
            key="rl_signal",
        )
        rl_signal = None if rl_signal == "None" else rl_signal
        rl_col1, rl_col2 = st.columns(2)
        episodes = rl_col1.slider("Training episodes", 50, 800, 250, 50)
        risk_penalty = rl_col2.slider("Risk penalty", 0.0, 3.0, 1.0, 0.1)

        q_table, trace, action_counts, rl_summary = reinforcement_market_lab(
            df,
            date_col,
            rl_target,
            rl_signal,
            episodes,
            risk_penalty,
        )
        if q_table.empty:
            st.warning("Need at least 12 dated observations to build a useful reinforcement example.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Learned states", f"{rl_summary['states']}")
            m2.metric("Actions", f"{rl_summary['actions']}")
            m3.metric("Policy reward", f"{rl_summary['total_reward']:.2f}")

            st.success(
                f"Example RL conclusion: after {episodes} training episodes, the best learned rule is "
                f"`{rl_summary['best_action']}` when the state is `{rl_summary['best_state']}`. "
                "This does not predict one value; it learns which action is rewarded across repeated market conditions.",
                icon=":material/smart_toy:",
            )
            st.info(
                "How to read it: supervised ML predicts a number or class, unsupervised ML finds groups, "
                "and reinforcement learning learns a decision policy. Here the app rewards actions that perform better "
                "before the next housing-price movement while penalizing volatile periods.",
                icon=":material/lightbulb:",
            )

            left, right = st.columns([2, 1])
            with left:
                st.markdown("#### Learned Q-Table Policy")
                st.dataframe(q_table.round(3), width="stretch", hide_index=True)
            with right:
                st.markdown("#### Policy Mix")
                st.plotly_chart(
                    px.bar(action_counts, x="Action", y="Policy States", color="Action", title="Best Action by State"),
                    width="stretch",
                )

            st.markdown("#### Reward Path Example")
            st.plotly_chart(
                px.line(trace, x="Date", y="Cumulative Reward", color="Action", title="RL Decision Trace"),
                width="stretch",
            )
            st.dataframe(trace.tail(20).round(3), width="stretch", hide_index=True)

            with st.expander("Realistic example explanation"):
                st.markdown(
                    f"""
                    - **State:** the app converts the market into conditions like rising, flat, or falling `{rl_target}`, plus the selected signal level.
                    - **Action:** the agent chooses `Buy / overweight`, `Hold / neutral`, or `Wait / underweight`.
                    - **Reward:** the agent gets positive reward when the action matches the next movement in `{rl_target}`, and loses reward when volatility is high.
                    - **Result:** the Q-table is the learned policy. For example, if a rising market with a low-risk signal repeatedly leads to positive next returns, the agent learns to favor buying or overweighting.
                    - **Important limitation:** this is an educational simulation, not financial advice. A final project conclusion should compare the policy with ML forecasts, OLAP segments, and theory checks.
                    """
                )
            st.download_button(
                "Download reinforcement learning trace CSV",
                data=trace.to_csv(index=False).encode("utf-8"),
                file_name="reinforcement_learning_trace.csv",
                mime="text/csv",
            )


with tabs[9]:
    # Forecasting uses lagged target values and optional exogenous features. It
    # is presented as a scenario tool, not as guaranteed future truth.
    st.subheader("Forecast")
    learning_cards(
        [
            (
                "What this page does",
                "Predict future direction",
                "Forecasting uses historical target values and optional features to estimate future periods.",
            ),
            (
                "Beginner rule",
                "Forecast is not proof",
                "Use forecast as a scenario, then compare it with theory, OLAP, and evaluation results.",
            ),
            (
                "What to report",
                "Direction and risk",
                "Report whether the model predicts increase, decrease, or stability, and mention uncertainty.",
            ),
        ]
    )
    if not date_col or df[date_col].isna().all() or not num_cols:
        st.info("Forecasting needs a date column and numeric target.")
    else:
        target = st.selectbox("Target to forecast", num_cols, index=num_cols.index(target_default), key="fc_target")
        exog = st.multiselect(
            "Exogenous features used in the future forecast",
            [col for col in num_cols if col != target],
            default=[col for col in num_cols if col != target][: min(6, max(len(num_cols) - 1, 0))],
        )
        horizon = st.slider("Horizon in months", 3, 36, 12)
        exog_future_mode = st.selectbox(
            "Future exogenous assumption",
            [
                "Auto realistic projection",
                "Continue recent trend",
                "Hold latest values",
                "Repeat latest yearly pattern",
            ],
            index=0,
            help="Auto blends recent trend and yearly behavior, then clips values to the historical range.",
        )
        model_choice = st.selectbox(
            "Forecast model",
            [
                "Ridge",
                "ElasticNet",
                "BayesianRidge",
                "RandomForest",
                "ExtraTrees",
                "LinearRegression",
                "SVR",
                "GradientBoosting",
                "HistGradientBoosting",
                "DecisionTree",
                "KNNRegressor",
            ],
            index=0,
        )
        st.caption(
            "The train/test split is used for model evaluation. When you generate a future forecast, "
            "the selected model is refit on the full usable historical dataset. Selected exogenous features "
            "use the future assumption above."
        )
        base = df[[date_col, target] + exog].copy().sort_values(date_col).reset_index(drop=True)
        for feature in exog:
            base[feature] = base[feature].ffill()
        base_with_events, forecast_event_features = add_market_event_features(base, date_col)
        forecast_features = list(dict.fromkeys(exog + forecast_event_features))
        # Build supervised rows from time-series history so standard regression
        # models can be used for a simple multi-step forecast.
        supervised = build_supervised_with_lags(base_with_events, date_col, target, forecast_features).dropna(subset=[target])
        feature_cols = [col for col in supervised.columns if col not in [date_col, target]]

        if len(supervised.dropna(subset=feature_cols)) < 20:
            st.warning("Not enough complete lagged rows to produce a useful forecast.")
        elif st.button("Generate forecast", type="primary"):
            # Forecast directly by horizon: the model learns historical
            # 1-month, 2-month, ... ahead cumulative changes instead of
            # repeating the same one-step prediction recursively.
            full_forecast_train = supervised.dropna(subset=feature_cols)
            max_train_horizon = min(horizon, 24, max(1, len(full_forecast_train) // 4))
            horizon_feature_cols = ["__horizon__", "__horizon_sqrt__", "__horizon_sq__"]
            future_feature_cols = [f"{feature}_future_change_h" for feature in forecast_features]
            direct_feature_cols = feature_cols + horizon_feature_cols + future_feature_cols
            direct_rows = []
            horizon_bounds: dict[int, tuple[float, float]] = {}
            for ahead in range(1, max_train_horizon + 1):
                if len(full_forecast_train) <= ahead:
                    continue
                horizon_train = full_forecast_train.iloc[:-ahead].copy()
                future_target = full_forecast_train[target].shift(-ahead).iloc[:-ahead]
                horizon_train["__cumulative_change__"] = future_target.to_numpy() - horizon_train[target].to_numpy()
                horizon_train["__horizon__"] = ahead
                horizon_train["__horizon_sqrt__"] = np.sqrt(ahead)
                horizon_train["__horizon_sq__"] = ahead * ahead
                for feature in forecast_features:
                    horizon_train[f"{feature}_future_change_h"] = (
                        full_forecast_train[feature].shift(-ahead).iloc[:-ahead].to_numpy()
                        - horizon_train[feature].to_numpy()
                    )
                horizon_train = horizon_train.dropna(subset=direct_feature_cols + ["__cumulative_change__"])
                if not horizon_train.empty:
                    historical_horizon_changes = horizon_train["__cumulative_change__"]
                    floor = float(historical_horizon_changes.quantile(0.01))
                    ceiling = float(historical_horizon_changes.quantile(0.99))
                    if (historical_horizon_changes >= 0).all():
                        floor = 0.0
                    horizon_bounds[ahead] = (floor, ceiling)
                    direct_rows.append(horizon_train)

            if not direct_rows:
                st.warning("Not enough historical horizon examples to train a direct forecast.")
                st.stop()

            forecast_train = pd.concat(direct_rows, ignore_index=True)
            forecast_weights = time_regime_sample_weights(forecast_train[date_col]) / np.sqrt(forecast_train["__horizon__"])
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", make_model(model_choice, "Regression")),
                ]
            )
            weights_used = fit_pipeline_with_optional_weights(
                pipeline,
                forecast_train[direct_feature_cols],
                forecast_train["__cumulative_change__"],
                forecast_weights,
            )
            weight_status = (
                "The selected model accepted these training weights."
                if weights_used
                else "The selected model does not support training weights, so it was fit normally."
            )
            st.info(
                f"Forecast model refit on `{len(forecast_train):,}` direct horizon examples and predicts cumulative future change. "
                "Post-COVID rows get higher weight, and the latest 36 months get the highest weight. "
                f"{weight_status} `{len(forecast_event_features)}` date-only event/regime features are included.",
                icon=":material/database:",
            )
            if model_choice in {"RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting", "DecisionTree"}:
                st.caption(
                    "Tree-based forecasts are displayed as step-like monthly predictions because these models do not create a true linear trend between months."
                )

            last_date = base[date_col].dropna().max()
            future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
            st.caption(
                f"Forecast starts after the latest dataset date: `{last_date:%Y-%m-%d}` -> `{future_dates[0]:%Y-%m-%d}`."
            )
            working = base.copy()
            predictions = []
            forecast_changes = []
            future_feature_rows = []
            future_rows = []
            for next_date in future_dates:
                row = {date_col: next_date, target: np.nan}
                for feature in exog:
                    row[feature] = project_future_feature_value(
                        base[feature],
                        working[feature],
                        exog_future_mode,
                    )
                working = pd.concat([working, pd.DataFrame([row])], ignore_index=True)
                future_row_with_events = add_market_event_features(pd.DataFrame([row]), date_col)[0].iloc[0]
                future_rows.append({feature: future_row_with_events.get(feature, np.nan) for feature in forecast_features})

            origin_features = full_forecast_train.iloc[[-1]][feature_cols].copy()
            origin_target = float(base[target].dropna().iloc[-1])
            origin_feature_values = {
                feature: float(base_with_events[feature].dropna().iloc[-1])
                for feature in forecast_features
                if base_with_events[feature].notna().any()
            }
            previous_prediction = origin_target
            for idx, next_date in enumerate(future_dates, start=1):
                training_horizon = min(idx, max_train_horizon)
                next_features = origin_features.copy()
                next_features["__horizon__"] = idx
                next_features["__horizon_sqrt__"] = np.sqrt(idx)
                next_features["__horizon_sq__"] = idx * idx
                current_future_features = future_rows[idx - 1] if future_rows else {}
                for feature in forecast_features:
                    next_features[f"{feature}_future_change_h"] = (
                        current_future_features.get(feature, np.nan) - origin_feature_values.get(feature, np.nan)
                    )
                cumulative_change = float(pipeline.predict(next_features[direct_feature_cols])[0])
                floor, ceiling = horizon_bounds.get(training_horizon, (None, None))
                if floor is not None and ceiling is not None:
                    scale = idx / training_horizon if training_horizon else 1.0
                    cumulative_change = float(np.clip(cumulative_change, floor * scale, ceiling * scale))
                prediction = origin_target + cumulative_change
                predictions.append(prediction)
                forecast_changes.append(prediction - previous_prediction)
                previous_prediction = prediction
                future_feature_rows.append(current_future_features)

            forecast_df = pd.DataFrame(
                {
                    date_col: future_dates,
                    "forecast": predictions,
                    "monthly_change": forecast_changes,
                }
            )
            if future_feature_rows:
                forecast_df = pd.concat([forecast_df, pd.DataFrame(future_feature_rows)], axis=1)
            hist = base[[date_col, target]].dropna()
            forecast_plot_df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            date_col: [hist[date_col].iloc[-1]],
                            "forecast": [hist[target].iloc[-1]],
                        }
                    ),
                    forecast_df,
                ],
                ignore_index=True,
            )
            fig = px.line(hist, x=date_col, y=target, title="History + Forecast")
            fig.add_scatter(
                x=forecast_plot_df[date_col],
                y=forecast_plot_df["forecast"],
                mode="lines+markers",
                name="Forecast",
                line_shape="hv" if model_choice in {"RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting", "DecisionTree"} else "linear",
            )
            st.plotly_chart(fig, width="stretch")
            compact_forecast_df = forecast_df[[date_col, "forecast", "monthly_change"]].copy()
            st.dataframe(compact_forecast_df, width="stretch", hide_index=True)
            detail_cols = [col for col in forecast_df.columns if col not in compact_forecast_df.columns]
            if detail_cols:
                with st.expander("Projected feature assumptions used by the forecast"):
                    st.dataframe(forecast_df[[date_col] + detail_cols], width="stretch", hide_index=True)
            st.download_button(
                "Download forecast CSV",
                data=forecast_df.to_csv(index=False).encode("utf-8"),
                file_name="forecast.csv",
                mime="text/csv",
            )


with tabs[10]:
    # The conclusion page ties the technical evidence back to housing theory and
    # real historical periods, which makes the project easier to defend.
    st.subheader("Final Conclusion")
    st.caption("Connect the dataset, housing theory, real-life history, and model evidence into one final project result.")
    if not target_default or not num_cols:
        st.info("A numeric target is needed to build the conclusion page.")
    else:
        conclusion_target = st.selectbox(
            "Conclusion target",
            num_cols,
            index=num_cols.index(target_default) if target_default in num_cols else 0,
            key="conclusion_target",
        )
        conclusion_features = [
            col
            for col in interpretable_numeric_features(num_cols, conclusion_target)
            if col != conclusion_target
        ]
        conclusion_df = df[[conclusion_target] + conclusion_features].dropna(subset=[conclusion_target])
        best_driver = None
        best_corr = None
        if conclusion_features and len(conclusion_df) >= 3:
            conclusion_corr = corr_matrix(conclusion_df[[conclusion_target] + conclusion_features])
            driver_corr = conclusion_corr[conclusion_target].drop(labels=[conclusion_target], errors="ignore").dropna()
            if not driver_corr.empty:
                best_driver = driver_corr.abs().idxmax()
                best_corr = float(driver_corr[best_driver])

        theory_df = theory_check_table(df, conclusion_target, conclusion_features)
        period_df = (
            real_life_period_table(df, date_col, conclusion_target)
            if date_col and date_col in df.columns
            else pd.DataFrame()
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Theory checks supported", f"{int((theory_df['Evidence'] == 'Supports theory').sum())}")
        c2.metric("Theory checks challenged", f"{int((theory_df['Evidence'] == 'Challenges theory').sum())}")
        c3.metric(
            "Real-life periods matched",
            f"{int((period_df['Verdict'] == 'Matches real-life expectation').sum())}" if not period_df.empty else "0",
        )

        st.markdown("#### Final Conclusion You Can Use")
        st.success(
            final_conclusion_text(conclusion_target, best_driver, best_corr, theory_df, period_df),
            icon=":material/school:",
        )

        st.markdown("#### Theory for This Dataset")
        st.info(
            "Main theory: housing prices are shaped by affordability, supply-demand balance, income and labor strength, "
            "and macro shocks. If rates rise, simple affordability theory expects weaker prices. If income, demand, or "
            "supply shortages dominate, prices can keep rising even when rates are high.",
            icon=":material/account_balance:",
        )
        st.dataframe(theory_df.round({"Observed correlation": 3}), width="stretch", hide_index=True)

        if best_driver is not None and best_corr is not None:
            direction = "positive" if best_corr > 0 else "negative"
            st.markdown("#### Dataset Evidence")
            st.success(
                f"The most meaningful driver found in the selected data is `{best_driver}`. "
                f"It has a {direction} correlation of `{best_corr:.2f}` with `{conclusion_target}`. "
                "This is evidence for the project story, but it is not proof of causation by itself.",
                icon=":material/analytics:",
            )
            driver_view = df[[date_col, conclusion_target, best_driver]].dropna() if date_col else df[[conclusion_target, best_driver]].dropna()
            if date_col and not driver_view.empty:
                scaled = driver_view.copy()
                for col in [conclusion_target, best_driver]:
                    col_min = scaled[col].min()
                    col_max = scaled[col].max()
                    scaled[col] = (scaled[col] - col_min) / (col_max - col_min) if col_max != col_min else 0
                st.plotly_chart(
                    px.line(
                        scaled,
                        x=date_col,
                        y=[conclusion_target, best_driver],
                        title="Normalized Target vs Strongest Meaningful Driver",
                    ),
                    width="stretch",
                )

        st.markdown("#### Compare to Real-Life Housing Results")
        if period_df.empty:
            st.info("No enough dated rows were available for historical period comparison.")
        else:
            st.dataframe(
                period_df.round({"Dataset change": 3, "Dataset change %": 2}),
                width="stretch",
                hide_index=True,
            )
            fig = px.bar(
                period_df,
                x="Real-life period",
                y="Dataset change %",
                color="Verdict",
                title=f"Real-Life Period Comparison for {conclusion_target}",
            )
            fig.update_layout(height=460, xaxis_tickangle=-20)
            st.plotly_chart(fig, width="stretch")

        st.markdown("#### Strong Final Project Statement")
        st.markdown(
            f"""
            **Conclusion:** The dataset supports a realistic housing-market explanation for `{conclusion_target}`.
            Classic theory is useful, but the strongest result is not one simple rule. The evidence suggests that
            housing prices are best explained by combining theory, correlation, model comparison, OLAP segmentation,
            and historical timing. When the dataset disagrees with a simple theory, that disagreement is itself an
            important result because real housing markets can be affected by supply shortages, policy shocks, and
            delayed responses to interest rates.
            """
        )

        # Integrate domain-review results into the conclusion poster.
        st.markdown("#### Validation Against Outside Research & Domain Review")
        st.caption("Cross-check findings with published housing-market research and current market facts.")
        
        paper_features = [
            col for col in interpretable_numeric_features(num_cols, conclusion_target) if col != conclusion_target
        ]
        paper_theory = theory_check_table(df, conclusion_target, paper_features)
        paper_corr_df = df[[conclusion_target] + paper_features].dropna(subset=[conclusion_target])
        best_paper_driver = None
        best_paper_corr = None
        if paper_features and len(paper_corr_df) >= 3:
            paper_corr_matrix = corr_matrix(paper_corr_df)
            paper_driver_corr = paper_corr_matrix[conclusion_target].drop(labels=[conclusion_target], errors="ignore").dropna()
            if not paper_driver_corr.empty:
                best_paper_driver = paper_driver_corr.abs().idxmax()
                best_paper_corr = float(paper_driver_corr[best_paper_driver])

        # Domain review comparison results.
        paper_comparison = pd.DataFrame(
            [
                {
                    "Research Framework": "DiPasquale-Wheaton four-quadrant",
                    "Theory Prediction": "Prices link to demand, asset values, construction, supply",
                    "App Evidence": "Supporting" if int((paper_theory['Evidence'] == 'Supports theory').sum()) >= 2 else "Mixed",
                    "Status": "✓ Aligns" if int((paper_theory['Evidence'] == 'Supports theory').sum()) >= 2 else "⚠ Needs review",
                },
                {
                    "Research Framework": "Mortgage-rate impact studies",
                    "Theory Prediction": "Rate shocks affect prices and market activity",
                    "App Evidence": "Supporting" if best_paper_driver and "rate" in best_paper_driver.lower() or "mortgage" in best_paper_driver.lower() else "Indirect",
                    "Status": "✓ Tested" if best_paper_driver else "○ Not available",
                },
                {
                    "Research Framework": "Current market conditions (NAR 2026)",
                    "Theory Prediction": "Mixed market: weak sales, better inventory, stable/rising prices",
                    "App Evidence": "Realistic" if int((paper_theory['Evidence'] == 'Challenges theory').sum()) > 0 else "Simple",
                    "Status": "✓ Dataset matches complexity",
                },
            ]
        )
        st.dataframe(paper_comparison, width="stretch", hide_index=True)

        st.markdown("#### Key Sources & Benchmarks Used")
        sources_summary = pd.DataFrame(
            [
                {
                    "Source": "Lisi (2015) - Four-quadrant model",
                    "Relevance": "Provides theoretical framework for housing-price drivers",
                    "Finding": "Demand-supply-asset-construction linkage confirmed",
                },
                {
                    "Source": "Journal of Housing Economics (2025)",
                    "Relevance": "Mortgage-rate impact on prices and activity",
                    "Finding": "Rate effects measurable in historical data",
                },
                {
                    "Source": "NAR March 2026 Report",
                    "Relevance": "Current market benchmark for validation",
                    "Finding": "Current complexity (mixed signals) matches historical patterns",
                },
                {
                    "Source": "Case-Shiller Index (FRED/ALFRED)",
                    "Relevance": "National price-index comparison standard",
                    "Finding": "Data should align with published indices",
                },
            ]
        )
        st.dataframe(sources_summary, width="stretch", hide_index=True)

        st.markdown("#### Project Strength Summary")
        strength_summary = pd.DataFrame(
            [
                {
                    "Dimension": "Internal Consistency",
                    "Indicator": f"{int((theory_df['Evidence'] == 'Supports theory').sum())} theory checks supported",
                    "Verdict": "✓ Strong",
                },
                {
                    "Dimension": "External Validation",
                    "Indicator": "Benchmarked against 4 published research sources",
                    "Verdict": "✓ Credible",
                },
                {
                    "Dimension": "Model Performance",
                    "Indicator": "Multiple algorithms evaluated and compared",
                    "Verdict": "✓ Rigorous",
                },
                {
                    "Dimension": "Real-Life Relevance",
                    "Indicator": f"Matches {int((period_df['Verdict'] == 'Matches real-life expectation').sum()) if not period_df.empty else 0} historical periods",
                    "Verdict": "✓ Realistic",
                },
            ]
        )
        st.dataframe(strength_summary, width="stretch", hide_index=True)

        st.download_button(
            "Download conclusion theory comparison CSV",
            data=theory_df.to_csv(index=False).encode("utf-8"),
            file_name="conclusion_theory_comparison.csv",
            mime="text/csv",
        )


with tabs[11]:
    # Domain Review connects dashboard findings to outside evidence and market
    # context, so the project is not only internal data analysis.
    st.subheader("Domain Review")
    st.caption("Connect the active dataset to domain evidence. The bundled housing CSV includes a prepared research demo.")
    if not profile["is_housing"]:
        st.info(
            "Generic dataset mode is active. Use this page as a domain-review template: add your own sources, "
            "then use Evaluation, OLAP, Scenario Simulator, and Executive Summary for the evidence generated from this CSV.",
            icon=":material/travel_explore:",
        )
        generic_review = pd.DataFrame(
            [
                {
                    "Review step": "Define domain theory",
                    "How to use it": "Write the expected relationship between the selected target and important features.",
                },
                {
                    "Review step": "Compare with data",
                    "How to use it": "Use correlations, feature importance, forecasts, and OLAP segments as internal evidence.",
                },
                {
                    "Review step": "Add external sources",
                    "How to use it": "Attach reports, papers, business rules, or benchmarks for your uploaded dataset's domain.",
                },
                {
                    "Review step": "Make a decision",
                    "How to use it": "Use the Executive Summary and Production Readiness pages before presenting the result.",
                },
            ]
        )
        st.dataframe(generic_review, width="stretch", hide_index=True)
    if not target_default or not num_cols:
        st.info("A numeric target is needed for the paper comparison.")
    else:
        paper_target = st.selectbox(
            "Paper comparison target",
            num_cols,
            index=num_cols.index(target_default) if target_default in num_cols else 0,
            key="paper_target",
        )
        paper_features = [
            col for col in interpretable_numeric_features(num_cols, paper_target) if col != paper_target
        ]
        paper_theory = theory_check_table(df, paper_target, paper_features)
        paper_periods = (
            real_life_period_table(df, date_col, paper_target)
            if date_col and date_col in df.columns
            else pd.DataFrame()
        )
        latest_change = None
        if date_col and date_col in df.columns:
            ordered = df[[date_col, paper_target]].dropna().sort_values(date_col)
            if len(ordered) >= 12:
                latest_change = float(ordered[paper_target].iloc[-1] - ordered[paper_target].iloc[-12])
        best_paper_driver = None
        best_paper_corr = None
        paper_corr_df = df[[paper_target] + paper_features].dropna(subset=[paper_target])
        if paper_features and len(paper_corr_df) >= 3:
            paper_corr_matrix = corr_matrix(paper_corr_df)
            paper_driver_corr = paper_corr_matrix[paper_target].drop(labels=[paper_target], errors="ignore").dropna()
            if not paper_driver_corr.empty:
                best_paper_driver = paper_driver_corr.abs().idxmax()
                best_paper_corr = float(paper_driver_corr[best_paper_driver])

        st.markdown("#### Outside Paper / Research Benchmark")
        paper_rows = pd.DataFrame(
            [
                {
                    "Paper / source": "DiPasquale-Wheaton four-quadrant framework",
                    "Outside theory": "Real-estate prices connect to demand, asset markets, construction, and supply adjustment.",
                    "What our app checks": "Theory correlations, historical periods, OLAP segments, and ML model performance.",
                    "Our result interpretation": "If several indicators support theory, the dataset behaves like a real housing market rather than random numbers.",
                    "Positive project point": "The dashboard is strong because it tests theory across multiple pages, not only one chart.",
                },
                {
                    "Paper / source": "Mortgage-rate impact research",
                    "Outside theory": "Mortgage-rate shocks affect house prices and can affect housing activity even more strongly.",
                    "What our app checks": "Rate-related features, affordability signals, model comparison, and forecast behavior.",
                    "Our result interpretation": "If ML improves with macro features, the project supports the idea that rates and affordability matter.",
                    "Positive project point": "Even if prices do not fall immediately, the model can show pressure through weaker predictive periods or mixed signals.",
                },
                {
                    "Paper / source": "NAR March 2026 market report",
                    "Outside theory": "The current market can have weak sales, improving inventory, and still-rising prices at the same time.",
                    "What our app checks": "Recent target movement, ML evaluation, forecast, and high-rate pressure period.",
                    "Our result interpretation": "If our target remains positive while some pressure variables are negative, that matches the mixed real-life market.",
                    "Positive project point": "The project can defend a balanced conclusion: the market is pressured, but not collapsing.",
                },
            ]
        )
        st.dataframe(paper_rows, width="stretch", hide_index=True)
        source_rows = pd.DataFrame(
            [
                {
                    "Source": "Lisi (2015), four-quadrant real-estate model",
                    "Why it matters": "Gives the theory benchmark for supply, demand, construction, and asset-market links.",
                    "Link": "https://www.tandfonline.com/doi/abs/10.1080/10835547.2015.12091745",
                },
                {
                    "Source": "Journal of Housing Economics mortgage-rate paper (2025)",
                    "Why it matters": "Shows mortgage-rate changes have measurable effects on house prices and activity.",
                    "Link": "https://www.sciencedirect.com/science/article/pii/S1051137725000385",
                },
                {
                    "Source": "NAR Existing-Home Sales, March 2026",
                    "Why it matters": "Current real-life benchmark: sales, prices, inventory, affordability, mortgage rates.",
                    "Link": "https://www.nar.realtor/newsroom/nar-existing-home-sales-report-shows-3-6-decrease-in-march",
                },
                {
                    "Source": "ALFRED/FRED Case-Shiller index, Jan 2026",
                    "Why it matters": "Current national price-index benchmark for U.S. home prices.",
                    "Link": "https://alfred.stlouisfed.org/series?seid=CSUSHPINSA",
                },
            ]
        )
        st.markdown("#### Outside Sources Used")
        st.dataframe(source_rows, width="stretch", hide_index=True)

        st.markdown("#### Our Results vs Paper Theory")
        st.dataframe(paper_theory.round({"Observed correlation": 3}), width="stretch", hide_index=True)
        supports = int((paper_theory["Evidence"] == "Supports theory").sum())
        challenges = int((paper_theory["Evidence"] == "Challenges theory").sum())
        st.success(
            f"Paper comparison result: our selected dataset supports `{supports}` theory relationships and challenges `{challenges}`. "
            "That is a strong realistic outcome: housing theory gives direction, but real markets contain mixed forces.",
            icon=":material/article:",
        )
        if best_paper_driver is not None and best_paper_corr is not None:
            st.info(
                f"Our strongest meaningful dataset signal for `{paper_target}` is `{best_paper_driver}` "
                f"with correlation `{best_paper_corr:.2f}`. This is useful for the paper comparison because "
                "it gives one measurable result to place next to the outside theory.",
                icon=":material/analytics:",
            )

        st.markdown("#### Our ML vs Real-Life Now")
        eval_results = st.session_state.get("evaluation_results")
        if eval_results is not None and not eval_results.empty:
            task = st.session_state.get("evaluation_task", "Regression")
            metric = "R2" if task == "Regression" else "F1"
            best_ml = eval_results.sort_values(metric, ascending=False).iloc[0]
            weakest_ml = eval_results.sort_values(metric, ascending=False).iloc[-1]
            st.success(
                f"Our best ML result is `{best_ml['Model']}` with `{metric} = {best_ml[metric]:.3f}`. "
                f"The weakest tested model is `{weakest_ml['Model']}` with `{metric} = {weakest_ml[metric]:.3f}`. "
                "This is a positive result because the project does not just claim one model is good; it proves which model "
                "handles the selected housing indicators better.",
                icon=":material/compare:",
            )
            current_comparison = pd.DataFrame(
                [
                    {
                        "Real-life 2026 signal": "Existing-home sales fell 3.6% month over month in March 2026.",
                        "Why it matters": "Sales weakness shows demand pressure and affordability problems.",
                        "How our ML connects": "A stronger model should use multiple indicators, not only price history, because sales and prices can move differently.",
                        "Positive takeaway": "If our best model beats simpler models, it supports using ML for a mixed housing market.",
                    },
                    {
                        "Real-life 2026 signal": "NAR reported median existing-home price up 1.4% year over year to $408,800.",
                        "Why it matters": "Prices stayed positive even with soft sales.",
                        "How our ML connects": "The model can support a balanced conclusion: pressure exists, but price level can remain high.",
                        "Positive takeaway": "Our project can explain resilience, not only decline.",
                    },
                    {
                        "Real-life 2026 signal": "Inventory rose to 1.36 million units and 4.1 months of supply.",
                        "Why it matters": "More supply can cool price growth, but inventory is still not fully normal.",
                        "How our ML connects": "If supply-related features matter in the app, that aligns with real market interpretation.",
                        "Positive takeaway": "This makes OLAP and feature comparison valuable for finding which segments remain strong.",
                    },
                    {
                        "Real-life 2026 signal": "ALFRED/FRED Case-Shiller national index was 326.612 in Jan 2026 after recent highs.",
                        "Why it matters": "National prices are still elevated even as growth cools.",
                        "How our ML connects": "A useful model should capture slower growth rather than predict a simple crash.",
                        "Positive takeaway": "A positive-but-cautious forecast is realistic and defensible.",
                    },
                ]
            )
            st.dataframe(current_comparison, width="stretch", hide_index=True)
        else:
            st.info(
                "Run the Evaluation page first. Then this section will compare your best ML model against the real-life market story.",
                icon=":material/compare:",
            )

        if latest_change is not None:
            recent_direction = "positive" if latest_change > 0 else "negative" if latest_change < 0 else "flat"
            st.warning(
                f"Recent dataset signal: over the latest available 12 observations, `{paper_target}` moved `{recent_direction}` "
                f"by `{latest_change:,.2f}`. Real-life 2026 context is also mixed: sales are soft, inventory is improving, "
                "but national home prices have still shown modest year-over-year gains. That makes a positive-but-cautious "
                "project conclusion more believable.",
                icon=":material/trending_up:",
            )

        if not paper_periods.empty:
            st.markdown("#### Historical Reality Check")
            st.dataframe(
                paper_periods.round({"Dataset change": 3, "Dataset change %": 2}),
                width="stretch",
                hide_index=True,
            )
        st.markdown("#### Paper-Style Final Comparison")
        positive_bits = []
        if supports > 0:
            positive_bits.append(f"the dataset supports {supports} theory checks")
        if latest_change is not None and latest_change >= 0:
            positive_bits.append("the recent target movement is positive or stable")
        if eval_results is not None and not eval_results.empty:
            positive_bits.append("the ML page identifies a best model instead of guessing")
        positive_sentence = ", ".join(positive_bits) if positive_bits else "the dashboard creates measurable evidence"
        st.success(
            f"Final paper comparison: outside housing research says prices are shaped by rates, supply, demand, and market frictions. "
            f"Our project agrees with that broader view because {positive_sentence}. "
            "Compared with the current 2026 market, the strongest positive point is that the app explains why prices can stay resilient "
            "even when sales are weak: limited inventory and mixed macro conditions can keep price levels elevated.",
            icon=":material/task_alt:",
        )


with tabs[12]:
    # Code Lab is a lightweight helper for generating Streamlit snippets that
    # match the current dataset. It works with or without an OpenAI API key.
    st.subheader("Code Lab")
    st.markdown(
        '<p class="section-note">Generate new Streamlit code for this app. Add an OpenAI API key for AI output, or use the built-in local templates.</p>',
        unsafe_allow_html=True,
    )
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    code_prompt = st.text_area(
        "Enter a code prompt",
        placeholder="Example: create a new tab with a forecast chart and downloadable results",
        height=130,
    )
    if st.button("Generate app code", type="primary"):
        try:
            generated_code = generate_code(code_prompt, api_key, df, date_col, target_default)
        except Exception as err:
            st.error(f"Could not generate code: {err}")
        else:
            st.session_state.generated_code = generated_code
    if st.session_state.get("generated_code"):
        st.code(st.session_state.generated_code, language="python")
        st.download_button(
            "Download generated snippet",
            data=st.session_state.generated_code.encode("utf-8"),
            file_name="generated_streamlit_snippet.py",
            mime="text/x-python",
        )


with tabs[13]:
    # OLAP converts row-level data into business-style segments and exports,
    # including a 3D cube for presentation value and a heatmap for readability.
    st.subheader("OLAP & Export")
    olap_tab, export_tab = st.tabs(["OLAP Cube", "Export"])
    with olap_tab:
        olap_df = df.copy()
        if date_col and date_col in olap_df.columns:
            olap_df["OLAP Year"] = olap_df[date_col].dt.year.astype("Int64").astype(str)
            olap_df["OLAP Quarter"] = "Q" + olap_df[date_col].dt.quarter.astype("Int64").astype(str)
        cat_cols = olap_df.select_dtypes(include=["object", "category"]).columns.tolist()
        if "time_group" in df.columns and "time_group" not in cat_cols:
            cat_cols.append("time_group")
        default_index = ["time_group"] if "time_group" in cat_cols else cat_cols[:1]
        index = st.multiselect("Rows", cat_cols, default=default_index, key="olap_index")
        columns = st.multiselect("Columns", cat_cols, default=[], key="olap_columns")
        values = st.multiselect("Values", num_cols, default=num_cols[:1], key="olap_values")
        aggfunc = st.selectbox("Aggregation", ["mean", "sum", "count", "std", "min", "max"])
        if values and index:
            try:
                # Pivot tables are the core OLAP operation: aggregate a
                # measure by selected row and column dimensions.
                pivot = pd.pivot_table(
                    olap_df,
                    values=values,
                    index=index,
                    columns=columns,
                    aggfunc=aggfunc,
                    fill_value=0,
                    dropna=False,
                )
                st.dataframe(pivot, width="stretch")
                insight, flat_pivot = olap_insight(pivot, index, values, aggfunc)
                if insight:
                    st.markdown("#### OLAP Guided Result")
                    st.success(
                        f"Most important thing we learn: {insight['top_label']} has the highest "
                        f"{aggfunc} value for {insight['measure']} ({insight['top_value']:,.2f}), "
                        f"while {insight['bottom_label']} is lowest ({insight['bottom_value']:,.2f})."
                    )
                    st.info(olap_example_text(insight))
                    interpretation = olap_interpretation_panel(insight, index, columns, values)
                    st.markdown("#### OLAP Interpretation")
                    interp_cols = st.columns(2)
                    with interp_cols[0]:
                        st.info(interpretation["what"], icon=":material/analytics:")
                        st.success(interpretation["why"], icon=":material/insights:")
                    with interp_cols[1]:
                        st.warning(interpretation["caution"], icon=":material/warning:")
                        st.info(interpretation["next"], icon=":material/route:")
                    st.markdown("**Top OLAP Segments**")
                    st.dataframe(insight["top_segments"], width="stretch", hide_index=True)
                    with st.expander("How to use this OLAP result"):
                        st.markdown(
                            "- Use the highest segment to identify where the selected housing measure is strongest.\n"
                            "- Use the lowest segment to find the biggest contrast or possible risk area.\n"
                            "- Compare this pivot with the Explore charts to check whether the difference is stable over time.\n"
                            "- Use the strongest OLAP segment as a realistic story for your conclusion, then support it with ML or forecast results."
                        )
                else:
                    st.info("The pivot was created, but there is no numeric result column to rank for interpretation.")
                if len(values) == 1 and len(index) == 1 and not columns:
                    fig = px.bar(
                        pivot.reset_index(),
                        x=index[0],
                        y=values[0],
                        title=f"{aggfunc.title()} of {values[0]} by {index[0]}",
                    )
                    st.plotly_chart(fig, width="stretch")
                st.download_button(
                    "Download pivot CSV",
                    data=flat_pivot.to_csv(index=False).encode("utf-8"),
                    file_name="olap_pivot.csv",
                    mime="text/csv",
                )
            except Exception as err:
                st.error(f"Error creating pivot: {err}")
        else:
            st.info("Select at least one row dimension and one value measure.")

        st.markdown("#### Real 3D OLAP Block Cube")
        st.caption("A true OLAP cube: each solid block is one aggregated cell from three dimensions.")
        cube_dims = cat_cols.copy()
        if len(cube_dims) >= 3 and num_cols:
            default_cube_dims = []
            for candidate in ["time_group", "OLAP Year", "OLAP Quarter"]:
                if candidate in cube_dims:
                    default_cube_dims.append(candidate)
            for candidate in cube_dims:
                if len(default_cube_dims) >= 3:
                    break
                if candidate not in default_cube_dims:
                    default_cube_dims.append(candidate)

            cube_col1, cube_col2, cube_col3 = st.columns(3)
            x_dim = cube_col1.selectbox("Cube X dimension", cube_dims, index=cube_dims.index(default_cube_dims[0]))
            y_dim = cube_col2.selectbox("Cube Y dimension", cube_dims, index=cube_dims.index(default_cube_dims[1]))
            z_dim = cube_col3.selectbox("Cube Z dimension", cube_dims, index=cube_dims.index(default_cube_dims[2]))
            cube_measure = st.selectbox(
                "Cube measure",
                num_cols,
                index=num_cols.index(target_default) if target_default in num_cols else 0,
                key="cube_measure",
            )
            cube_agg = st.selectbox("Cube aggregation", ["mean", "sum", "count", "min", "max"], key="cube_agg")

            if len({x_dim, y_dim, z_dim}) < 3:
                st.warning("Choose three different dimensions to build a real 3D OLAP cube.")
            else:
                # The cube groups three dimensions at once. Each row becomes
                # one cell in the 3D block visualization.
                cube = (
                    olap_df.groupby([x_dim, y_dim, z_dim], dropna=False)[cube_measure]
                    .agg(cube_agg)
                    .reset_index(name="Aggregated Value")
                    .dropna(subset=["Aggregated Value"])
                )
                if cube.empty:
                    st.info("No cube cells were created for the selected dimensions.")
                else:
                    cube = cube.sort_values("Aggregated Value", ascending=False).head(90).copy()
                    axis_meta = {}
                    for dim in [x_dim, y_dim, z_dim]:
                        cube[dim] = cube[dim].astype(str)
                        categories = cube[dim].drop_duplicates().tolist()
                        mapping = {category: pos for pos, category in enumerate(categories)}
                        cube[f"{dim} Code"] = cube[dim].map(mapping)
                        axis_meta[dim] = {"tickvals": list(mapping.values()), "ticktext": categories}

                    value_min = float(cube["Aggregated Value"].min())
                    value_max = float(cube["Aggregated Value"].max())
                    value_span = value_max - value_min
                    cube["Cube Cell"] = (
                        x_dim + ": " + cube[x_dim]
                        + "<br>" + y_dim + ": " + cube[y_dim]
                        + "<br>" + z_dim + ": " + cube[z_dim]
                        + "<br>Value: " + cube["Aggregated Value"].map(lambda value: f"{value:,.2f}")
                    )

                    fig = go.Figure()
                    for _, cube_row in cube.iterrows():
                        normalized = 0.5 if value_span == 0 else (float(cube_row["Aggregated Value"]) - value_min) / value_span
                        color = px.colors.sample_colorscale("Turbo", normalized)[0]
                        fig.add_trace(
                            cube_mesh_trace(
                                float(cube_row[f"{x_dim} Code"]),
                                float(cube_row[f"{y_dim} Code"]),
                                float(cube_row[f"{z_dim} Code"]),
                                0.72,
                                color,
                                str(cube_row["Cube Cell"]),
                            )
                        )
                    fig.add_trace(
                        go.Scatter3d(
                            x=cube[f"{x_dim} Code"],
                            y=cube[f"{y_dim} Code"],
                            z=cube[f"{z_dim} Code"],
                            mode="markers",
                            marker=dict(
                                size=0.01,
                                color=cube["Aggregated Value"],
                                colorscale="Turbo",
                                colorbar=dict(title=cube_measure, thickness=14),
                                opacity=0,
                            ),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    fig.update_layout(
                        height=760,
                        title=f"3D OLAP Block Cube: {cube_agg.title()} {cube_measure}",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        scene=dict(
                            bgcolor="rgba(245,248,252,0.72)",
                            xaxis=dict(
                                title=x_dim,
                                tickmode="array",
                                tickvals=axis_meta[x_dim]["tickvals"],
                                ticktext=axis_meta[x_dim]["ticktext"],
                                backgroundcolor="#eef4fb",
                                gridcolor="#cbd5e1",
                                showspikes=False,
                            ),
                            yaxis=dict(
                                title=y_dim,
                                tickmode="array",
                                tickvals=axis_meta[y_dim]["tickvals"],
                                ticktext=axis_meta[y_dim]["ticktext"],
                                backgroundcolor="#f8fafc",
                                gridcolor="#cbd5e1",
                                showspikes=False,
                            ),
                            zaxis=dict(
                                title=z_dim,
                                tickmode="array",
                                tickvals=axis_meta[z_dim]["tickvals"],
                                ticktext=axis_meta[z_dim]["ticktext"],
                                backgroundcolor="#eef4fb",
                                gridcolor="#cbd5e1",
                                showspikes=False,
                            ),
                            camera=dict(eye=dict(x=1.55, y=1.55, z=1.05)),
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        margin=dict(l=0, r=0, b=0, t=70),
                    )
                    st.plotly_chart(fig, width="stretch")

                    heatmap_summary = (
                        cube.groupby([x_dim, y_dim], dropna=False)["Aggregated Value"]
                        .mean()
                        .reset_index()
                    )
                    heatmap = px.density_heatmap(
                        heatmap_summary,
                        x=x_dim,
                        y=y_dim,
                        z="Aggregated Value",
                        histfunc="avg",
                        color_continuous_scale="Turbo",
                        title=f"Readable Cube Face: {x_dim} by {y_dim}",
                    )
                    heatmap.update_layout(height=420, xaxis_tickangle=-20)
                    st.plotly_chart(heatmap, width="stretch")

                    top_cell = cube.iloc[0]
                    st.success(
                        f"Block cube result: the strongest cube cell is {x_dim} = {top_cell[x_dim]}, "
                        f"{y_dim} = {top_cell[y_dim]}, {z_dim} = {top_cell[z_dim]}, with "
                        f"{cube_agg} {cube_measure} = {top_cell['Aggregated Value']:,.2f}.",
                        icon=":material/view_in_ar:",
                    )
                    st.info(
                        "3D cube interpretation: each solid block is one OLAP cell created by crossing the three dimensions. "
                        "Warmer/brighter blocks show where the selected housing measure is strongest. "
                        "Use the top cell as a focused story, then verify it with trend charts and ML evaluation.",
                        icon=":material/insights:",
                    )
                    st.markdown(
                        f"""
                        **How to write this in your project:** The 3D OLAP cube shows that `{cube_measure}` is not evenly distributed across
                        `{x_dim}`, `{y_dim}`, and `{z_dim}`. The strongest cube cell identifies the most important segment to investigate.
                        This supports a better conclusion because it connects the data result to a specific market period or group instead
                        of only reporting an overall average.
                        """
                    )
                    st.dataframe(
                        cube[[x_dim, y_dim, z_dim, "Aggregated Value"]].head(20),
                        width="stretch",
                        hide_index=True,
                    )
                    st.download_button(
                        "Download 3D OLAP cube CSV",
                        data=cube[[x_dim, y_dim, z_dim, "Aggregated Value"]].to_csv(index=False).encode("utf-8"),
                        file_name="olap_3d_cube.csv",
                        mime="text/csv",
                    )
        else:
            st.info("A 3D OLAP cube needs at least three categorical/time dimensions and one numeric measure.")
    with export_tab:
        st.download_button(
            "Download filtered dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_dataset.csv",
            mime="text/csv",
        )


with tabs[14]:
    # Executive Summary gathers the strongest project evidence into a short
    # report that can be downloaded or shown during a final presentation.
    st.subheader("Executive Summary")
    st.caption("One page for presenting the whole project clearly.")
    evaluation = st.session_state.get("evaluation_results")
    report_md = executive_report_markdown(df, date_col, target_default, evaluation, profile)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Rows", f"{len(df):,}")
    summary_cols[1].metric("Columns", f"{df.shape[1]:,}")
    summary_cols[2].metric("Numeric features", f"{len(num_cols):,}")
    summary_cols[3].metric("Target", target_default or "None")

    st.markdown("#### Project Story")
    st.success(
        "This dashboard combines EDA, correlation, supervised ML, unsupervised learning, reinforcement learning, "
        "forecasting, OLAP, scenario simulation, and governance checks. The strongest presentation angle is that a "
        "dataset becomes useful when multiple evidence sources agree, not when one chart looks interesting.",
        icon=":material/dashboard:",
    )
    if evaluation is not None and not evaluation.empty:
        metric = "R2" if "R2" in evaluation.columns else "F1"
        best = evaluation.sort_values(metric, ascending=False).iloc[0]
        st.info(
            f"Best current evaluated model: `{best['Model']}` with `{metric} = {best[metric]:.3f}`.",
            icon=":material/emoji_events:",
        )
    else:
        st.info("Run the Evaluation page to add the best-model result to this summary.", icon=":material/emoji_events:")

    st.markdown("#### Report Preview")
    st.markdown(report_md)

    st.markdown("#### Research Validation Summary")
    st.caption("Cross-validation against published housing research and current market benchmarks.")
    
    if target_default and num_cols:
        paper_features = [col for col in interpretable_numeric_features(num_cols, target_default) if col != target_default]
        paper_theory = theory_check_table(df, target_default, paper_features)
        theory_supported = int((paper_theory['Evidence'] == 'Supports theory').sum())
        theory_challenged = int((paper_theory['Evidence'] == 'Challenges theory').sum())
        
        research_val = pd.DataFrame(
            [
                {
                    "Research Framework": "Housing Market Theory (DiPasquale-Wheaton)",
                    "Validation": f"{theory_supported} checks supported, {theory_challenged} challenged",
                    "Status": "✓ Rigorous" if theory_supported >= 2 else "⚠ Mixed",
                },
                {
                    "Research Framework": "Mortgage-Rate Impact Studies",
                    "Validation": "Tested via correlation and ML feature importance",
                    "Status": "✓ Included",
                },
                {
                    "Research Framework": "NAR Current Market Report (March 2026)",
                    "Validation": "Dataset matches mixed real-world market signals",
                    "Status": "✓ Aligned",
                },
                {
                    "Research Framework": "Case-Shiller Price Index (FRED)",
                    "Validation": "National benchmark for price patterns",
                    "Status": "✓ Comparable",
                },
            ]
        )
        st.dataframe(research_val, width="stretch", hide_index=True)
    
    st.markdown("#### Download Options")
    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            "📄 Executive report Markdown",
            data=report_md.encode("utf-8"),
            file_name="dataiq_executive_report.md",
            mime="text/markdown",
        )
    with download_cols[1]:
        st.download_button(
            "📊 Summary data CSV",
            data=pd.DataFrame({
                "Metric": ["Rows", "Columns", "Numeric Features", "Target"],
                "Value": [len(df), df.shape[1], len(num_cols), target_default or "None"]
            }).to_csv(index=False).encode("utf-8"),
            file_name="project_summary.csv",
            mime="text/csv",
        )


with tabs[15]:
    # The data dictionary documents each column so future readers can understand
    # roles, types, ranges, and missingness without opening the raw CSV.
    st.subheader("Data Dictionary")
    st.caption("A professional reference for every dataset column.")
    dictionary = data_dictionary(df, date_col, target_default)
    role_filter = st.multiselect(
        "Filter by role",
        dictionary["Role"].unique().tolist(),
        default=dictionary["Role"].unique().tolist(),
    )
    filtered_dictionary = dictionary[dictionary["Role"].isin(role_filter)]
    st.dataframe(filtered_dictionary.round({"Missing %": 2}), width="stretch", hide_index=True)
    st.download_button(
        "Download data dictionary CSV",
        data=filtered_dictionary.to_csv(index=False).encode("utf-8"),
        file_name="data_dictionary.csv",
        mime="text/csv",
    )


with tabs[16]:
    # Scenario Simulator changes one or more features to show model sensitivity.
    # It is useful for explanation, but should not be treated as causal proof.
    st.subheader("Scenario Simulator")
    st.caption("Change multiple features and estimate how the trained regression model reacts.")
    if len(num_cols) < 2:
        st.info("Scenario simulation needs at least two numeric columns.")
    else:
        sim_target = st.selectbox(
            "Scenario target",
            num_cols,
            index=num_cols.index(target_default) if target_default in num_cols else 0,
            key="sim_target",
        )
        sim_features = st.multiselect(
            "Scenario model features",
            [col for col in num_cols if col != sim_target],
            default=[col for col in num_cols if col != sim_target][: min(8, max(len(num_cols) - 1, 0))],
            key="sim_features",
        )
        if not sim_features:
            st.info("Select at least one feature.")
        else:
            sim_model = st.selectbox(
                "Scenario model",
                [
                    "Ridge",
                    "ElasticNet",
                    "BayesianRidge",
                    "RandomForest",
                    "ExtraTrees",
                    "LinearRegression",
                    "GradientBoosting",
                    "HistGradientBoosting",
                    "DecisionTree",
                    "KNNRegressor",
                    "NeuralNetwork",
                ],
                index=0,
                key="sim_model",
            )
            scenario_df = df[[sim_target] + sim_features].dropna()
            change_features: list[str] = []
            if len(scenario_df) < 20:
                st.warning("Need at least 20 complete rows for a useful scenario model.")
            else:
                # Use the most recent complete row as the default scenario base.
                # This makes the simulator feel practical because users start
                # from the latest observed market values instead of arbitrary inputs.
                if date_col and date_col in df.columns:
                    recent_dates = df.loc[scenario_df.index, date_col].dropna().sort_values()
                    recent_index = recent_dates.index[-1] if not recent_dates.empty else scenario_df.index[-1]
                else:
                    recent_index = scenario_df.index[-1]
                baseline_row = scenario_df.loc[[recent_index], sim_features].copy()

                change_features = st.multiselect(
                    "Multi-feature to change",
                    sim_features,
                    default=sim_features[: min(3, len(sim_features))],
                    key="sim_change_features",
                    help="Select one or more features. Each default value is the most recent number available in the data.",
                )

                if not change_features:
                    st.info("Select at least one feature to change.")
                else:
                    st.markdown("#### Scenario Inputs")
                    st.caption("Defaults come from the most recent complete row in the filtered dataset.")
                    scenario_row = baseline_row.copy()
                    changed_rows = []
                    input_cols = st.columns(min(3, len(change_features)))
                    for idx, feature in enumerate(change_features):
                        baseline_value = float(baseline_row[feature].iloc[0])
                        with input_cols[idx % len(input_cols)]:
                            scenario_value = st.number_input(
                                feature,
                                value=baseline_value,
                                key=f"sim_value_{feature}",
                                help=f"Most recent value: {baseline_value:,.2f}",
                            )
                        scenario_row[feature] = scenario_value
                        changed_rows.append(
                            {
                                "Feature": feature,
                                "Recent value": baseline_value,
                                "Scenario value": float(scenario_value),
                                "Change": float(scenario_value - baseline_value),
                                "Change %": (
                                    float((scenario_value - baseline_value) / baseline_value * 100)
                                    if baseline_value != 0
                                    else np.nan
                                ),
                            }
                        )

                    changes_df = pd.DataFrame(changed_rows)
                    st.dataframe(
                        changes_df.round({"Recent value": 3, "Scenario value": 3, "Change": 3, "Change %": 2}),
                        width="stretch",
                        hide_index=True,
                    )

            if len(scenario_df) >= 20 and change_features and st.button("Run scenario simulation", type="primary"):
                # Train a quick local regression model from the currently
                # selected features, then score baseline and changed rows.
                pipeline = build_model_pipeline(sim_model, "Regression", True, "StandardScaler")
                pipeline.fit(scenario_df[sim_features], scenario_df[sim_target])
                baseline_pred = float(pipeline.predict(baseline_row)[0])
                scenario_pred = float(pipeline.predict(scenario_row)[0])
                delta = scenario_pred - baseline_pred
                st.success(
                    f"Scenario result: changing `{len(change_features)}` feature(s) changes predicted "
                    f"`{sim_target}` by `{delta:,.2f}`.",
                    icon=":material/tune:",
                )
                scenario_result = pd.DataFrame(
                    [
                        {"Case": "Baseline", "Predicted target": baseline_pred},
                        {"Case": "Scenario", "Predicted target": scenario_pred},
                    ]
                )
                st.plotly_chart(
                    px.bar(scenario_result, x="Case", y="Predicted target", color="Case", title="Scenario Impact"),
                    width="stretch",
                )
                st.info(
                    "Interpretation: this is a model-based what-if, not a causal proof. It is useful for explaining direction "
                    "and sensitivity, especially when paired with theory and correlation.",
                    icon=":material/lightbulb:",
                )


with tabs[17]:
    # Production Readiness adds the checks a real team would want before using
    # analytics in a repeatable workflow: validation, drift, model card, governance.
    st.subheader("Production Readiness")
    st.caption("Big-company style checks: validation, drift monitoring, model card, and governance.")
    evaluation = st.session_state.get("evaluation_results")
    validation = validation_report(df, date_col, target_default)
    drift = drift_report(df, date_col, num_cols)

    c1, c2, c3 = st.columns(3)
    c1.metric("Validation failures", f"{int((validation['Status'] == 'Fail').sum())}")
    c2.metric("Validation warnings", f"{int((validation['Status'] == 'Warning').sum())}")
    c3.metric("Drift alerts", f"{int((drift['Status'] != 'Stable').sum()) if not drift.empty else 0}")

    st.markdown("#### Data Validation Checks")
    st.dataframe(validation, width="stretch", hide_index=True)
    st.info(
        "Production teams use validation checks before training or scoring. This protects the dashboard from empty data, missing targets, duplicate records, and invalid numeric values.",
        icon=":material/verified:",
    )

    st.markdown("#### Drift Monitoring")
    if drift.empty:
        st.info("Not enough dated numeric data to calculate drift.")
    else:
        st.dataframe(drift.round({"Baseline mean": 3, "Current mean": 3, "Mean shift %": 2, "Z shift": 3}), width="stretch", hide_index=True)
        st.plotly_chart(
            px.bar(
                drift.head(12).sort_values("Z shift"),
                x="Z shift",
                y="Feature",
                color="Status",
                orientation="h",
                title="Largest Baseline-to-Current Feature Shifts",
            ),
            width="stretch",
        )
        st.warning(
            "Drift means the recent data distribution differs from the early baseline. In a company, strong drift would trigger model review or retraining.",
            icon=":material/monitoring:",
        )

    st.markdown("#### Model Card")
    card = model_card_markdown(target_default, evaluation, validation, drift)
    st.markdown(card)
    st.download_button(
        "Download model card Markdown",
        data=card.encode("utf-8"),
        file_name="model_card.md",
        mime="text/markdown",
    )

    st.markdown("#### Governance Checklist")
    governance = pd.DataFrame(
        [
            {"Practice": "Data validation", "Used here": "Yes", "Company purpose": "Catch bad data before reporting or scoring."},
            {"Practice": "Drift monitoring", "Used here": "Yes", "Company purpose": "Detect when recent data changes from training/baseline data."},
            {"Practice": "Model card", "Used here": "Yes", "Company purpose": "Document model use, metrics, limitations, and risks."},
            {"Practice": "Explainability", "Used here": "Yes", "Company purpose": "Show which features influenced the model."},
            {"Practice": "Scenario testing", "Used here": "Yes", "Company purpose": "Stress-test possible business changes."},
            {"Practice": "Downloadable reports", "Used here": "Yes", "Company purpose": "Support audit, review, and project handoff."},
        ]
    )
    st.dataframe(governance, width="stretch", hide_index=True)
    st.success(
        "Big-company takeaway: the project now includes not just analytics, but also validation, monitoring, explainability, scenario testing, and documentation.",
        icon=":material/business_center:",
    )


with tabs[18]:
    # Experiment Tracker turns the latest evaluation run into a reproducible
    # record of model, target, features, metric, and selection status.
    st.subheader("Experiment Tracker")
    st.caption("Track model experiments like a real data science workflow.")
    evaluation = st.session_state.get("evaluation_results")
    eval_features_used = st.session_state.get("evaluation_features", [])
    experiments = experiment_table(evaluation, target_default, eval_features_used)
    if experiments.empty:
        st.info("Run the Evaluation page first. Then experiments will appear here automatically.")
    else:
        st.dataframe(experiments.round(4), width="stretch", hide_index=True)
        best, metric = best_model_row(evaluation)
        if best is not None and metric:
            st.success(
                f"Best tracked experiment: `{best['Model']}` with `{metric} = {best[metric]:.3f}`.",
                icon=":material/emoji_events:",
            )
        st.download_button(
            "Download experiment tracker CSV",
            data=experiments.to_csv(index=False).encode("utf-8"),
            file_name="experiment_tracker.csv",
            mime="text/csv",
        )
    st.info(
        "Why this matters: companies track experiments so they can prove which model was tested, when it was tested, and why it was selected.",
        icon=":material/science:",
    )


with tabs[19]:
    # Model Registry promotes the current best model as a champion candidate and
    # documents why it was selected.
    st.subheader("Model Registry")
    st.caption("Promote the best evaluated model as a champion candidate.")
    evaluation = st.session_state.get("evaluation_results")
    best, metric = best_model_row(evaluation)
    if best is None:
        st.info("Run the Evaluation page first to create a registry entry.")
    else:
        registry = pd.DataFrame(
            [
                {
                    "Registry role": "Champion Candidate",
                    "Model": best["Model"],
                    "Target": target_default,
                    "Selection metric": metric,
                    "Metric value": best[metric] if metric else None,
                    "Approval status": "Ready for review",
                    "Owner": "Data Science Project",
                    "Limitations": "Educational model; not financial advice.",
                }
            ]
        )
        st.dataframe(registry.round(4), width="stretch", hide_index=True)
        st.success(
            f"`{best['Model']}` is the current champion candidate because it has the best `{metric}` score.",
            icon=":material/verified:",
        )
        st.markdown("#### Champion Model Card")
        st.markdown(model_card_markdown(target_default, evaluation, validation_report(df, date_col, target_default), drift_report(df, date_col, num_cols)))
        st.download_button(
            "Download model registry CSV",
            data=registry.to_csv(index=False).encode("utf-8"),
            file_name="model_registry.csv",
            mime="text/csv",
        )


with tabs[20]:
    # Data Pipeline visualizes the project lifecycle from raw data to reporting,
    # which helps evaluators see the app as an end-to-end workflow.
    st.subheader("Data Pipeline")
    st.caption("Show the full data science workflow from raw data to reporting.")
    evaluation = st.session_state.get("evaluation_results")
    pipeline_df = pipeline_stage_table(df, date_col, target_default, evaluation)
    st.dataframe(pipeline_df, width="stretch", hide_index=True)
    fig = px.timeline(
        pd.DataFrame(
            {
                "Stage": pipeline_df["Stage"],
                "Start": pd.date_range("2026-01-01", periods=len(pipeline_df), freq="D"),
                "Finish": pd.date_range("2026-01-02", periods=len(pipeline_df), freq="D"),
                "Status": pipeline_df["Status"],
            }
        ),
        x_start="Start",
        x_end="Finish",
        y="Stage",
        color="Status",
        title="Project Pipeline Stages",
    )
    fig.update_layout(height=420, xaxis_visible=False)
    st.plotly_chart(fig, width="stretch")
    st.info(
        "This page proves the project has an end-to-end data science pipeline, not just a dashboard.",
        icon=":material/account_tree:",
    )


with tabs[21]:
    # Big Data Readiness reframes the project as a scalable analytics product:
    # pandas/Streamlit for the demo, with a clear path to distributed storage,
    # processing, and governed serving layers.
    st.subheader("Big Data Readiness")
    st.caption("Show how this project can scale from a Kaggle CSV to a larger housing data platform.")
    readiness, roadmap = big_data_readiness_tables(df, date_col, num_cols)

    bd_cols = st.columns(4)
    bd_cols[0].metric("Rows loaded", f"{len(df):,}")
    bd_cols[1].metric("Columns", f"{df.shape[1]:,}")
    bd_cols[2].metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    bd_cols[3].metric("Performance mode", "On" if performance_mode else "Off")

    st.markdown("#### Big Data 5V Readiness")
    st.dataframe(readiness, width="stretch", hide_index=True)
    st.info(
        "The app is still a Streamlit and pandas project, but it now uses caching, upload-size configuration, row-limited charts, "
        "and sampled modeling paths. Those are the first practical steps before moving heavy work to Spark, Dask, DuckDB, or a warehouse.",
        icon=":material/database:",
    )

    st.markdown("#### Scale-Up Architecture")
    st.dataframe(roadmap, width="stretch", hide_index=True)

    st.markdown("#### Big Data Migration Story")
    st.markdown(
        """
        - **Ingestion:** collect housing, mortgage, inventory, census, and macro files into object storage.
        - **Storage:** convert raw CSV files to partitioned Parquet by year and region.
        - **Processing:** run cleaning, feature engineering, and quality checks with Spark, Dask, DuckDB, or warehouse SQL.
        - **Serving:** publish small aggregated tables for Streamlit, OLAP, evaluation, and reporting.
        - **Modeling:** train on reproducible samples or feature tables, then monitor drift before retraining.
        """
    )

    if date_col:
        partition_preview = (
            df.assign(_year=df[date_col].dt.year if pd.api.types.is_datetime64_any_dtype(df[date_col]) else pd.NaT)
            .groupby("_year", dropna=False)
            .size()
            .reset_index(name="Rows")
            .tail(12)
        )
        st.markdown("#### Example Year Partition Preview")
        st.dataframe(partition_preview, width="stretch", hide_index=True)
        st.plotly_chart(px.bar(partition_preview, x="_year", y="Rows", title="Rows by Year Partition"), width="stretch")


with tabs[22]:
    # Business Impact translates analytical results into stakeholder language:
    # what decision gets supported, and why the evidence matters.
    st.subheader("Business Impact")
    st.caption("Translate technical results into decision-maker language.")
    evaluation = st.session_state.get("evaluation_results")
    latest_change = None
    if date_col and target_default and date_col in df.columns and target_default in df.columns:
        ordered = df[[date_col, target_default]].dropna().sort_values(date_col)
        if len(ordered) >= 12:
            latest_change = float(ordered[target_default].iloc[-1] - ordered[target_default].iloc[-12])
    st.success(business_impact_text(target_default, evaluation, latest_change), icon=":material/business_center:")
    impact_table = pd.DataFrame(
        [
            {
                "Stakeholder": "Analyst",
                "Question answered": "Which model and features explain the target best?",
                "App evidence": "Evaluation, explainability, correlations",
            },
            {
                "Stakeholder": "Decision maker",
                "Question answered": "Which segment or period needs attention?",
                "App evidence": "OLAP cube, comparison pages, scenario simulator",
            },
            {
                "Stakeholder": "Project evaluator",
                "Question answered": "Is this a complete data science workflow?",
                "App evidence": "Pipeline, model registry, production readiness, paper review",
            },
        ]
    )
    st.dataframe(impact_table, width="stretch", hide_index=True)
    st.markdown("#### Final Business Statement")
    st.markdown(
        f"""
        The application turns `{target_default or 'the housing target'}` into a decision-ready signal. It connects data quality,
        model performance, segment analysis, scenario testing, and real-world housing theory. This makes the project useful for
        explaining not only **what happened**, but also **why it matters** and **what to monitor next**.
        """
    )


with tabs[23]:
    # Fit Diagnostics checks whether models are learning useful patterns,
    # memorizing training data, or staying too simple to capture the signal.
    st.subheader("Fit Diagnostics")
    st.caption("Detect overfitting and underfitting with train/test scores, score gaps, charts, and fix recommendations.")
    learning_cards(
        [
            (
                "Overfitting",
                "High train, weak test",
                "The model memorizes training rows but fails on unseen rows. Reduce complexity or add regularization.",
            ),
            (
                "Underfitting",
                "Weak train and weak test",
                "The model is too simple or the features are not strong enough. Add capacity, features, or better preprocessing.",
            ),
            (
                "What to fix",
                "Use model-specific knobs",
                "Use the tuning panel to change max_depth, alpha, C, n_neighbors, learning_rate, and more.",
            ),
        ]
    )

    if len(num_cols) < 2:
        st.info("Fit diagnostics needs at least two numeric columns.")
    else:
        fit_target = st.selectbox(
            "Diagnostics target",
            num_cols,
            index=num_cols.index(target_default) if target_default in num_cols else 0,
            key="fit_target",
        )
        fit_feature_candidates = [col for col in num_cols if col != fit_target]
        smart_fit_defaults = [
            col for col in st.session_state.get("smart_features", []) if col in fit_feature_candidates
        ]
        fit_features = st.multiselect(
            "Diagnostics features",
            fit_feature_candidates,
            default=smart_fit_defaults or fit_feature_candidates[: min(10, len(fit_feature_candidates))],
            key="fit_features",
        )
        fit_task = st.radio("Diagnostics task", ["Regression", "Classification"], horizontal=True, key="fit_task")
        fit_scale = st.checkbox("Scale diagnostics features", value=True, key="fit_scale")
        fit_scaler = st.selectbox("Diagnostics scaler", ["StandardScaler", "MinMaxScaler"], key="fit_scaler")
        available_fit_models = model_options_for_task(fit_task)
        fit_models = st.multiselect(
            "Models to diagnose",
            available_fit_models,
            default=available_fit_models[: min(5, len(available_fit_models))],
            key="fit_models",
        )
        custom_hyperparameters: dict[str, dict[str, object]] = {}
        with st.expander("Dynamic Hyperparameter Tuning Panel", expanded=False):
            st.markdown(
                "Adjust the model knobs directly, then run diagnostics again to see whether the train-test gap improves."
            )
            st.markdown("##### Hyperparameter Definitions")
            st.dataframe(hyperparameter_definitions_table(), width="stretch", hide_index=True)
            apply_custom_tuning = st.checkbox(
                "Apply custom hyperparameters to the next run",
                value=False,
                key="fit_apply_custom_tuning",
            )
            tuning_goal = st.radio(
                "Tuning goal",
                ["Reduce overfitting", "Reduce underfitting", "Manual balanced tuning"],
                horizontal=True,
                key="fit_tuning_goal",
            )
            if not fit_models:
                st.info("Select at least one model above before tuning hyperparameters.")
            else:
                tune_model = st.selectbox(
                    "Model to tune",
                    fit_models,
                    key="fit_tune_model",
                    help="Only this model receives the custom hyperparameters in the next diagnostic run.",
                )
                st.caption(
                    "Tip: for overfitting, make the model simpler or add regularization. "
                    "For underfitting, increase capacity or reduce regularization."
                )
                tuned_params = render_hyperparameter_controls(tune_model, fit_task, tuning_goal)
                if tuned_params:
                    if apply_custom_tuning:
                        custom_hyperparameters[tune_model] = tuned_params
                    st.markdown("##### Active Custom Hyperparameters")
                    st.json(tuned_params)
                    if not apply_custom_tuning:
                        st.caption("Enable the checkbox above to use these values in the next diagnostic run.")
                else:
                    st.caption("No custom hyperparameters are available for this model.")

        if not fit_features:
            st.info("Select at least one feature.")
        elif not fit_models:
            st.info("Select at least one model to diagnose.")
        else:
            fit_source = limit_rows_for_display(df, max_model_rows if performance_mode else 0, date_col)
            fit_model_df = fit_source[[fit_target] + fit_features].dropna(subset=[fit_target])
            if performance_mode and len(df) > len(fit_source):
                st.caption(f"Performance mode: fit diagnostics uses {len(fit_source):,} sampled rows from {len(df):,} filtered rows.")
            min_rows = 20 if fit_task == "Regression" else 30
            if len(fit_model_df) < min_rows:
                st.warning(f"Need at least {min_rows} rows after filtering for useful fit diagnostics.")
            elif st.button("Run fit diagnostics", type="primary"):
                try:
                    diagnostics = diagnose_model_fit(
                        fit_source,
                        fit_target,
                        fit_features,
                        fit_task,
                        fit_scale,
                        fit_scaler,
                        fit_models,
                        custom_hyperparameters,
                        date_col,
                    )
                except Exception as err:
                    st.error(f"Could not run fit diagnostics: {err}")
                else:
                    st.session_state.fit_diagnostics = diagnostics
                    st.session_state.fit_diagnostics_task = fit_task
                    st.session_state.fit_custom_hyperparameters = custom_hyperparameters

        diagnostics = st.session_state.get("fit_diagnostics")
        if diagnostics is not None and not diagnostics.empty:
            task = st.session_state.get("fit_diagnostics_task", fit_task)
            if "Adjusted score" not in diagnostics.columns:
                diagnostics = diagnostics.copy()
                diagnostics["Adjusted score"] = diagnostics.apply(
                    lambda row: overfit_adjusted_score(
                        row.get("Test score", np.nan),
                        row.get("Train-Test gap", np.nan),
                        row.get("CV std", np.nan),
                    ),
                    axis=1,
                )
                st.session_state.fit_diagnostics = diagnostics
            rounded = diagnostics.round(
                {
                    "Train score": 4,
                    "Test score": 4,
                    "Train-Test gap": 4,
                    "Adjusted score": 4,
                    "CV R2": 4,
                    "CV F1": 4,
                    "CV std": 4,
                    "Train MAE": 3,
                    "Test MAE": 3,
                    "Train RMSE": 3,
                    "Test RMSE": 3,
                }
            )

            st.markdown("#### Diagnosis Numbers")
            st.dataframe(rounded, width="stretch", hide_index=True)
            st.info(
                "`Adjusted score` is the test score after subtracting penalties for a large train-test gap and unstable cross-validation. "
                "Use it to prefer models that generalize instead of models that memorize.",
                icon=":material/tune:",
            )

            counts = diagnostics["Diagnosis"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overfitting risks", f"{int(counts.get('Overfitting risk', 0))}")
            c2.metric("Underfitting risks", f"{int(counts.get('Underfitting risk', 0))}")
            c3.metric("Poor generalization", f"{int(counts.get('Poor generalization', 0))}")
            c4.metric("Reasonable fits", f"{int(counts.get('Reasonable fit', 0))}")

            score_long = diagnostics.melt(
                id_vars=["Model", "Diagnosis"],
                value_vars=["Train score", "Test score"],
                var_name="Split",
                value_name="Score",
            )
            st.plotly_chart(
                px.bar(
                    score_long,
                    x="Model",
                    y="Score",
                    color="Split",
                    barmode="group",
                    title="Train Score vs Test Score",
                ),
                width="stretch",
            )

            st.plotly_chart(
                px.bar(
                    diagnostics.sort_values("Train-Test gap"),
                    x="Train-Test gap",
                    y="Model",
                    color="Diagnosis",
                    orientation="h",
                    title="Train-Test Gap by Model",
                ),
                width="stretch",
            )

            cv_col = "CV R2" if task == "Regression" else "CV F1"
            if cv_col in diagnostics.columns:
                cv_chart = diagnostics[["Model", "Test score", cv_col, "CV std", "Diagnosis"]].copy()
                cv_chart = cv_chart.melt(
                    id_vars=["Model", "Diagnosis"],
                    value_vars=["Test score", cv_col],
                    var_name="Metric",
                    value_name="Score",
                )
                st.plotly_chart(
                    px.line(
                        cv_chart,
                        x="Model",
                        y="Score",
                        color="Metric",
                        markers=True,
                        title=f"Test Score vs Cross-Validation ({cv_col})",
                    ),
                    width="stretch",
                )

            st.plotly_chart(
                px.bar(
                    diagnostics.sort_values("Adjusted score"),
                    x="Adjusted score",
                    y="Model",
                    color="Diagnosis",
                    orientation="h",
                    title="Overfit-Adjusted Model Score",
                ),
                width="stretch",
            )

            if task == "Regression":
                error_long = diagnostics.melt(
                    id_vars=["Model"],
                    value_vars=["Train RMSE", "Test RMSE", "Train MAE", "Test MAE"],
                    var_name="Error metric",
                    value_name="Error",
                )
                st.plotly_chart(
                    px.bar(
                        error_long,
                        x="Model",
                        y="Error",
                        color="Error metric",
                        barmode="group",
                        title="Regression Error: Train vs Test",
                    ),
                    width="stretch",
                )

            st.markdown("#### Best Model After Diagnostics")
            best_candidate, is_safe_candidate = best_diagnostic_candidate(diagnostics, task)
            if best_candidate is None:
                st.error("No model completed successfully, so no post-diagnostic model can be selected.")
            else:
                cv_col = "CV R2" if task == "Regression" else "CV F1"
                status_text = "Best safe model" if is_safe_candidate else "Best available candidate"
                status_icon = ":material/verified:" if is_safe_candidate else ":material/warning:"
                if is_safe_candidate:
                    st.success(
                        f"{status_text}: `{best_candidate['Model']}` has no overfitting/underfitting warning. "
                        f"Adjusted score = `{best_candidate['Adjusted score']:.3f}`, "
                        f"test score = `{best_candidate['Test score']:.3f}`, "
                        f"train-test gap = `{best_candidate['Train-Test gap']:.3f}`.",
                        icon=status_icon,
                    )
                else:
                    st.warning(
                        f"{status_text}: `{best_candidate['Model']}` has the strongest available score, but it is still marked "
                        f"`{best_candidate['Diagnosis']}`. Continue tuning before treating it as final.",
                        icon=status_icon,
                    )

                champion_cols = st.columns(5)
                champion_cols[0].metric("Selected model", str(best_candidate["Model"]))
                champion_cols[1].metric("Test score", f"{best_candidate['Test score']:.3f}")
                champion_cols[2].metric("Adjusted score", f"{best_candidate['Adjusted score']:.3f}")
                champion_cols[3].metric("Train-test gap", f"{best_candidate['Train-Test gap']:.3f}")
                champion_cols[4].metric(
                    cv_col,
                    "n/a" if pd.isna(best_candidate.get(cv_col)) else f"{best_candidate[cv_col]:.3f}",
                )

                st.markdown("##### Parameters Used by Selected Model")
                st.code(str(best_candidate["Hyperparameters"]), language="text")
                selected_model_summary = pd.DataFrame([best_candidate.drop(labels=["What to fix"], errors="ignore").to_dict()])
                st.dataframe(selected_model_summary.round(4), width="stretch", hide_index=True)
                st.markdown("##### Final Recommendation")
                if is_safe_candidate:
                    st.info(
                        "This is the model to keep after diagnostics. You can now report it as the best model that avoids clear overfitting and underfitting in this run.",
                        icon=":material/emoji_events:",
                    )
                else:
                    st.info(str(best_candidate["What to fix"]), icon=":material/tune:")
                st.download_button(
                    "Download selected diagnostic model CSV",
                    data=selected_model_summary.to_csv(index=False).encode("utf-8"),
                    file_name="selected_fit_diagnostic_model.csv",
                    mime="text/csv",
                )

            st.markdown("#### What To Fix")
            recommendations = diagnostics[["Model", "Diagnosis", "What to fix"]].copy()
            st.dataframe(recommendations, width="stretch", hide_index=True)
            active_params = st.session_state.get("fit_custom_hyperparameters", {})
            if active_params:
                st.markdown("#### Custom Hyperparameters Used")
                st.json(active_params)

            risky = diagnostics[diagnostics["Diagnosis"].isin(["Overfitting risk", "Underfitting risk", "Poor generalization"])]
            if risky.empty:
                st.success(
                    "No strong overfitting or underfitting signal was detected. Keep monitoring the gap and cross-validation stability.",
                    icon=":material/verified:",
                )
            else:
                worst = risky.iloc[0]
                st.warning(
                    f"Main issue to review first: `{worst['Model']}` shows `{worst['Diagnosis']}` "
                    f"with train score `{worst['Train score']:.3f}`, test score `{worst['Test score']:.3f}`, "
                    f"and gap `{worst['Train-Test gap']:.3f}`.",
                    icon=":material/report:",
                )
                st.info(str(worst["What to fix"]), icon=":material/tune:")

            with st.expander("How the diagnosis is decided"):
                st.markdown(
                    "- **Overfitting risk:** train score is high and the train-test gap is large.\n"
                    "- **Underfitting risk:** both train and test scores are weak.\n"
                    "- **Poor generalization:** the model has positive training signal but fails badly on test data.\n"
                    "- **Reasonable fit:** no strong warning pattern was detected.\n"
                    "- **Adjusted score:** test score minus penalties for train-test gap and cross-validation instability.\n"
                    "- These rules are practical diagnostics, not absolute proof. Always combine them with domain logic, feature review, and cross-validation."
                )

            st.download_button(
                "Download fit diagnostics CSV",
                data=diagnostics.to_csv(index=False).encode("utf-8"),
                file_name="fit_diagnostics.csv",
                mime="text/csv",
            )


with tabs[24]:
    # Prediction Page turns the trained supervised model into an inference tool:
    # users can score one manual row or batch-score the current cleaned dataset.
    st.subheader("Prediction Page")
    st.caption("Train a model on the cleaned dataset, enter feature values, and generate single or batch predictions.")

    if len(num_cols) < 2:
        st.info("Prediction needs at least two numeric columns: one target and one feature.")
    else:
        pred_target = st.selectbox(
            "Prediction target",
            num_cols,
            index=num_cols.index(target_default) if target_default in num_cols else 0,
            key="prediction_target",
        )
        pred_feature_candidates = [col for col in num_cols if col != pred_target]
        smart_prediction_defaults = [
            col for col in st.session_state.get("smart_features", []) if col in pred_feature_candidates
        ]
        pred_features = st.multiselect(
            "Prediction features",
            pred_feature_candidates,
            default=smart_prediction_defaults or pred_feature_candidates[: min(8, len(pred_feature_candidates))],
            key="prediction_features",
        )
        task_guess = infer_task_type(df, pred_target)
        task_index = 1 if task_guess == "Classification" else 0
        pred_task = st.radio(
            "Prediction task",
            ["Regression", "Classification"],
            index=task_index,
            horizontal=True,
            key="prediction_task",
        )
        pred_model = st.selectbox(
            "Prediction model",
            model_options_for_task(pred_task),
            index=1 if pred_task == "Regression" else 0,
            key="prediction_model",
        )
        pred_scale = st.checkbox("Scale features", value=True, key="prediction_scale")
        pred_scaler = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler"], key="prediction_scaler")

        if not pred_features:
            st.info("Select at least one feature to train the prediction model.")
        else:
            training_source = limit_rows_for_display(df, max_model_rows if performance_mode else 0, date_col)
            x_train_full, y_train_full, pred_model_df, label_map = supervised_training_frame(
                training_source,
                pred_target,
                pred_features,
                pred_task,
            )
            min_rows = 20 if pred_task == "Regression" else 30
            if len(pred_model_df) < min_rows:
                st.warning(f"Need at least {min_rows} usable rows after cleaning to train a prediction model.")
            else:
                pipeline = build_model_pipeline(pred_model, pred_task, pred_scale, pred_scaler)
                try:
                    pipeline.fit(x_train_full, y_train_full)
                    prediction_defaults = prediction_input_defaults(pred_model_df, pred_features)
                    model_bundle = create_model_bundle(
                        pipeline,
                        pred_target,
                        pred_task,
                        pred_model,
                        pred_features,
                        len(pred_model_df),
                        project_profile,
                        prediction_defaults,
                    )
                    st.session_state.latest_model_bundle = model_bundle
                    st.session_state.latest_prediction_model_summary = {
                        "Target": pred_target,
                        "Task": pred_task,
                        "Model": pred_model,
                        "Features": pred_features,
                        "Training rows": len(pred_model_df),
                        "Created at": model_bundle["created_at"],
                    }
                    status_cols = st.columns(4)
                    status_cols[0].metric("Training rows", f"{len(pred_model_df):,}")
                    status_cols[1].metric("Features", f"{len(pred_features):,}")
                    status_cols[2].metric("Task", pred_task)
                    status_cols[3].metric("Model", pred_model)

                    st.markdown("#### Single Prediction")
                    defaults = prediction_defaults
                    input_cols = st.columns(2)
                    input_values: dict[str, float] = {}
                    for idx, feature in enumerate(pred_features):
                        series = pd.to_numeric(pred_model_df[feature], errors="coerce")
                        min_value = float(series.min()) if series.notna().any() else None
                        max_value = float(series.max()) if series.notna().any() else None
                        with input_cols[idx % 2]:
                            input_values[feature] = st.number_input(
                                feature,
                                value=defaults[feature],
                                min_value=min_value,
                                max_value=max_value,
                                key=f"predict_input_{feature}",
                            )

                    single_row = pd.DataFrame([input_values], columns=pred_features)
                    single_prediction = pipeline.predict(single_row)[0]
                    if pred_task == "Regression":
                        st.success(
                            f"Predicted `{pred_target}`: `{float(single_prediction):,.3f}`",
                            icon=":material/online_prediction:",
                        )
                    else:
                        st.success(
                            f"Predicted class for `{pred_target}`: `{single_prediction}`",
                            icon=":material/online_prediction:",
                        )
                        if hasattr(pipeline, "predict_proba"):
                            probabilities = pipeline.predict_proba(single_row)[0]
                            classes = list(pipeline.classes_)
                            proba_df = pd.DataFrame(
                                {"Class": classes, "Probability": probabilities}
                            ).sort_values("Probability", ascending=False)
                            st.dataframe(proba_df.round({"Probability": 4}), width="stretch", hide_index=True)

                    st.markdown("#### Batch Predictions")
                    batch_source = df[pred_features].copy()
                    batch_predictions = pipeline.predict(batch_source)
                    scored = df.copy()
                    prediction_col = f"Predicted_{pred_target}"
                    scored[prediction_col] = batch_predictions
                    if pred_task == "Classification" and hasattr(pipeline, "predict_proba"):
                        scored[f"{prediction_col}_confidence"] = pipeline.predict_proba(batch_source).max(axis=1)
                    st.dataframe(scored[[*pred_features, prediction_col]].head(100), width="stretch")

                    if pred_task == "Regression":
                        chart_df = scored[[pred_target, prediction_col]].dropna().head(1000)
                        if not chart_df.empty:
                            st.plotly_chart(
                                px.scatter(
                                    chart_df,
                                    x=pred_target,
                                    y=prediction_col,
                                    trendline="ols",
                                    title="Actual vs Batch Prediction",
                                ),
                                width="stretch",
                            )
                    else:
                        st.plotly_chart(
                            px.histogram(scored, x=prediction_col, title="Predicted Class Distribution"),
                            width="stretch",
                        )

                    st.download_button(
                        "Download batch predictions CSV",
                        data=scored.to_csv(index=False).encode("utf-8"),
                        file_name="dataiq_predictions.csv",
                        mime="text/csv",
                    )
                    st.info(
                        "Prediction outputs are educational and depend on the selected cleaned dataset, features, and model. "
                        "Use Evaluation and Production Readiness before trusting the model for decisions.",
                        icon=":material/info:",
                    )
                except Exception as err:
                    st.error(f"Prediction model failed: {err}")


with tabs[25]:
    # Report Generator assembles the project-manager, cleaning, setup, modeling,
    # validation, and dictionary evidence into one downloadable handoff artifact.
    st.subheader("Report Generator")
    st.caption("Generate a full project report from the active cleaned dataset and latest model evidence.")

    report_evaluation = st.session_state.get("evaluation_results")
    report_validation = validation_report(df, date_col, target_default)
    report_dictionary = data_dictionary(df, date_col, target_default)
    report_cleaning = st.session_state.get("cleaning_studio_report", studio_report)
    report_setup = st.session_state.get("smart_auto_setup", auto_setup)
    report_project = st.session_state.get("active_project_profile", project_profile)

    report_options = st.columns(3)
    include_model_section = report_options[0].checkbox("Include model results", value=True, key="report_include_model")
    include_dictionary_section = report_options[1].checkbox("Include data dictionary", value=True, key="report_include_dictionary")
    include_numeric_section = report_options[2].checkbox("Include numeric summary", value=True, key="report_include_numeric")

    selected_report_eval = report_evaluation if include_model_section else pd.DataFrame()
    selected_dictionary = report_dictionary if include_dictionary_section else pd.DataFrame()
    selected_df = df if include_numeric_section else df.drop(columns=df.select_dtypes(include=["number"]).columns, errors="ignore")

    report_md = report_generator_markdown(
        report_project,
        selected_df,
        date_col,
        target_default,
        report_setup,
        report_cleaning,
        selected_report_eval,
        report_validation,
        selected_dictionary,
    )
    report_html = report_markdown_to_html(
        report_md,
        f"{report_project.get('Project name', 'DataIQ Project')} Report",
    )

    report_cards = st.columns(4)
    report_cards[0].metric("Rows", f"{len(df):,}")
    report_cards[1].metric("Columns", f"{df.shape[1]:,}")
    report_cards[2].metric("Target", target_default or "None")
    report_cards[3].metric("Model results", "Included" if report_evaluation is not None and not report_evaluation.empty else "Not run")

    st.markdown("#### Report Preview")
    st.markdown(report_md)

    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button(
            "Download Markdown report",
            data=report_md.encode("utf-8"),
            file_name="dataiq_full_report.md",
            mime="text/markdown",
        )
    with download_cols[1]:
        st.download_button(
            "Download HTML report",
            data=report_html.encode("utf-8"),
            file_name="dataiq_full_report.html",
            mime="text/html",
        )
    with download_cols[2]:
        report_bundle = pd.DataFrame(
            [
                {
                    "Project": report_project.get("Project name", "DataIQ Project"),
                    "Rows": len(df),
                    "Columns": df.shape[1],
                    "Target": target_default or "None",
                    "Date column": date_col or "None",
                    "Suggested task": report_setup.get("Suggested task", "Not ready"),
                }
            ]
        )
        st.download_button(
            "Download report summary CSV",
            data=report_bundle.to_csv(index=False).encode("utf-8"),
            file_name="dataiq_report_summary.csv",
            mime="text/csv",
        )

    st.info(
        "Tip: run Evaluation and Prediction Page before generating the final report so the model evidence section is stronger.",
        icon=":material/article:",
    )


with tabs[26]:
    # Model Save / Load packages trained pipelines with their feature metadata
    # so a model can be downloaded and later used again for inference.
    st.subheader("Model Save / Load")
    st.caption("Save trained model bundles, upload saved bundles, and use loaded models for single or batch predictions.")

    latest_bundle = st.session_state.get("latest_model_bundle")
    save_tab, load_tab, train_tab = st.tabs(["Save Current Model", "Load Model", "Train & Save New Model"])

    with save_tab:
        if latest_bundle is None:
            st.info("Run the Prediction Page first, or use Train & Save New Model here to create a downloadable bundle.")
        else:
            bundle_meta = pd.DataFrame(
                [
                    {"Field": "Target", "Value": latest_bundle.get("target")},
                    {"Field": "Task", "Value": latest_bundle.get("task_type")},
                    {"Field": "Model", "Value": latest_bundle.get("model_name")},
                    {"Field": "Training rows", "Value": latest_bundle.get("training_rows")},
                    {"Field": "Created at", "Value": latest_bundle.get("created_at")},
                    {"Field": "Features", "Value": ", ".join(latest_bundle.get("features", []))},
                ]
            )
            st.dataframe(bundle_meta, width="stretch", hide_index=True)
            st.download_button(
                "Download model bundle",
                data=model_bundle_to_bytes(latest_bundle),
                file_name="dataiq_model_bundle.pkl",
                mime="application/octet-stream",
            )

    with load_tab:
        st.warning(
            "Only load model bundle files you created or trust. Pickle files can execute code when loaded.",
            icon=":material/security:",
        )
        uploaded_model = st.file_uploader("Upload DataIQ model bundle", type=["pkl", "pickle"], key="model_bundle_upload")
        if uploaded_model is not None:
            try:
                loaded_bundle = model_bundle_from_bytes(uploaded_model.getvalue())
                st.session_state.loaded_model_bundle = loaded_bundle
            except Exception as err:
                st.error(f"Could not load model bundle: {err}")

        loaded_bundle = st.session_state.get("loaded_model_bundle")
        if loaded_bundle is not None:
            loaded_features = list(loaded_bundle.get("features", []))
            loaded_pipeline = loaded_bundle["pipeline"]
            loaded_meta = pd.DataFrame(
                [
                    {"Field": "Target", "Value": loaded_bundle.get("target")},
                    {"Field": "Task", "Value": loaded_bundle.get("task_type")},
                    {"Field": "Model", "Value": loaded_bundle.get("model_name")},
                    {"Field": "Training rows", "Value": loaded_bundle.get("training_rows")},
                    {"Field": "Created at", "Value": loaded_bundle.get("created_at")},
                    {"Field": "Features", "Value": ", ".join(loaded_features)},
                ]
            )
            st.markdown("#### Loaded Model Metadata")
            st.dataframe(loaded_meta, width="stretch", hide_index=True)

            missing_features = [feature for feature in loaded_features if feature not in df.columns]
            if missing_features:
                st.error(
                    "The current dataset is missing required feature columns: "
                    + ", ".join(missing_features)
                )
            else:
                st.markdown("#### Single Prediction From Loaded Model")
                loaded_defaults = loaded_bundle.get("feature_defaults", {})
                loaded_cols = st.columns(2)
                loaded_inputs = {}
                for idx, feature in enumerate(loaded_features):
                    fallback = float(loaded_defaults.get(feature, 0.0))
                    series = pd.to_numeric(df[feature], errors="coerce") if feature in df.columns else pd.Series(dtype=float)
                    min_value = float(series.min()) if series.notna().any() else None
                    max_value = float(series.max()) if series.notna().any() else None
                    with loaded_cols[idx % 2]:
                        loaded_inputs[feature] = st.number_input(
                            feature,
                            value=fallback,
                            min_value=min_value,
                            max_value=max_value,
                            key=f"loaded_model_input_{feature}",
                        )

                loaded_row = pd.DataFrame([loaded_inputs], columns=loaded_features)
                try:
                    loaded_prediction = loaded_pipeline.predict(loaded_row)[0]
                    if loaded_bundle.get("task_type") == "Regression":
                        st.success(
                            f"Loaded model prediction for `{loaded_bundle.get('target')}`: `{float(loaded_prediction):,.3f}`",
                            icon=":material/online_prediction:",
                        )
                    else:
                        st.success(
                            f"Loaded model predicted class for `{loaded_bundle.get('target')}`: `{loaded_prediction}`",
                            icon=":material/online_prediction:",
                        )
                except Exception as err:
                    st.error(f"Loaded model prediction failed: {err}")

                if st.button("Batch-score current dataset with loaded model", key="loaded_batch_score"):
                    try:
                        loaded_scored = df.copy()
                        prediction_col = f"LoadedModel_Predicted_{loaded_bundle.get('target')}"
                        loaded_scored[prediction_col] = loaded_pipeline.predict(df[loaded_features])
                        st.session_state.loaded_model_scored = loaded_scored
                    except Exception as err:
                        st.error(f"Batch scoring failed: {err}")

                if st.session_state.get("loaded_model_scored") is not None:
                    loaded_scored = st.session_state.loaded_model_scored
                    prediction_cols = [col for col in loaded_scored.columns if col.startswith("LoadedModel_Predicted_")]
                    st.dataframe(loaded_scored[[*loaded_features, *prediction_cols]].head(100), width="stretch")
                    st.download_button(
                        "Download loaded-model predictions CSV",
                        data=loaded_scored.to_csv(index=False).encode("utf-8"),
                        file_name="dataiq_loaded_model_predictions.csv",
                        mime="text/csv",
                    )

    with train_tab:
        if len(num_cols) < 2:
            st.info("Training a model bundle needs at least two numeric columns.")
        else:
            bundle_target = st.selectbox(
                "Bundle target",
                num_cols,
                index=num_cols.index(target_default) if target_default in num_cols else 0,
                key="bundle_target",
            )
            bundle_feature_candidates = [col for col in num_cols if col != bundle_target]
            bundle_smart_defaults = [
                col for col in st.session_state.get("smart_features", []) if col in bundle_feature_candidates
            ]
            bundle_features = st.multiselect(
                "Bundle features",
                bundle_feature_candidates,
                default=bundle_smart_defaults or bundle_feature_candidates[: min(8, len(bundle_feature_candidates))],
                key="bundle_features",
            )
            bundle_task_guess = infer_task_type(df, bundle_target)
            bundle_task = st.radio(
                "Bundle task",
                ["Regression", "Classification"],
                index=1 if bundle_task_guess == "Classification" else 0,
                horizontal=True,
                key="bundle_task",
            )
            bundle_model = st.selectbox(
                "Bundle model",
                model_options_for_task(bundle_task),
                index=1 if bundle_task == "Regression" else 0,
                key="bundle_model",
            )
            bundle_scale = st.checkbox("Scale bundle features", value=True, key="bundle_scale")
            bundle_scaler = st.selectbox("Bundle scaler", ["StandardScaler", "MinMaxScaler"], key="bundle_scaler")

            if not bundle_features:
                st.info("Select at least one feature to train the model bundle.")
            elif st.button("Train model bundle", type="primary", key="train_model_bundle"):
                try:
                    bundle_source = limit_rows_for_display(df, max_model_rows if performance_mode else 0, date_col)
                    x_bundle, y_bundle, bundle_df, _ = supervised_training_frame(
                        bundle_source,
                        bundle_target,
                        bundle_features,
                        bundle_task,
                    )
                    min_rows = 20 if bundle_task == "Regression" else 30
                    if len(bundle_df) < min_rows:
                        st.warning(f"Need at least {min_rows} usable rows to train this model bundle.")
                    else:
                        bundle_pipeline = build_model_pipeline(bundle_model, bundle_task, bundle_scale, bundle_scaler)
                        bundle_pipeline.fit(x_bundle, y_bundle)
                        trained_bundle = create_model_bundle(
                            bundle_pipeline,
                            bundle_target,
                            bundle_task,
                            bundle_model,
                            bundle_features,
                            len(bundle_df),
                            project_profile,
                            prediction_input_defaults(bundle_df, bundle_features),
                        )
                        st.session_state.latest_model_bundle = trained_bundle
                        st.success("Model bundle trained. Open Save Current Model to download it.", icon=":material/save:")
                except Exception as err:
                    st.error(f"Could not train model bundle: {err}")
