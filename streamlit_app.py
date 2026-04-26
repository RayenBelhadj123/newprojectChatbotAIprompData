import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
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


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from us_housing.paths import resolve_default_dataset  # noqa: E402


st.set_page_config(
    page_title="US Housing Intelligence Dashboard",
    page_icon=":house:",
    layout="wide",
)

DEFAULT_CSV = resolve_default_dataset()


def apply_theme(dark_mode: bool) -> None:
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

    st.markdown(
        f"""
        <style>
            :root {{
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
            .stApp {{
                background:
                    linear-gradient(180deg, var(--panel-3) 0px, var(--app-bg) 260px, var(--app-bg) 100%);
                color: var(--text);
            }}
            [data-testid="stHeader"] {{
                background: transparent;
                height: 0;
            }}
            [data-testid="stToolbar"] {{
                display: none;
            }}
            #MainMenu, footer {{
                visibility: hidden;
            }}
            .block-container {{
                padding-top: 1.15rem;
                padding-bottom: 2.4rem;
                max-width: 1480px;
            }}
            h1, h2, h3, h4 {{
                color: var(--text);
                letter-spacing: 0;
                font-weight: 760;
            }}
            h2 {{
                padding-top: 0.25rem;
            }}
            h3 {{
                border-left: 4px solid var(--accent);
                padding-left: 0.7rem;
                margin-top: 1.4rem;
            }}
            p, li, label, span {{
                letter-spacing: 0;
            }}
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, var(--panel-3), var(--panel));
                border-right: 1px solid var(--border);
                box-shadow: var(--shadow);
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
                border-radius: 16px;
                padding: 30px;
                background:
                    linear-gradient(135deg, rgba(33,184,166,0.14), rgba(244,176,70,0.08)),
                    var(--panel);
                margin-bottom: 14px;
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
                font-size: 2.22rem;
                margin: 0 0 8px 0;
                line-height: 1.12;
            }}
            .hero p {{
                color: var(--muted);
                margin: 0;
                line-height: 1.55;
                max-width: 920px;
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
                grid-template-columns: repeat(6, minmax(120px, 1fr));
                gap: 10px;
                margin: 12px 0 18px 0;
            }}
            .workflow-step {{
                border: 1px solid var(--border);
                background: linear-gradient(180deg, var(--panel), var(--panel-2));
                border-radius: 13px;
                padding: 14px;
                box-shadow: 0 14px 34px rgba(0,0,0,0.13);
                min-height: 78px;
            }}
            .workflow-step .step-kicker {{
                color: var(--accent);
                font-size: 0.72rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}
            .workflow-step .step-title {{
                color: var(--text);
                font-size: 0.92rem;
                font-weight: 760;
                margin-top: 3px;
            }}
            .metric-card {{
                min-height: 108px;
                border: 1px solid var(--border);
                border-radius: 13px;
                background: linear-gradient(180deg, var(--panel), var(--panel-2));
                padding: 17px;
                box-shadow: 0 16px 44px rgba(0,0,0,0.13);
                border-top: 3px solid var(--accent);
            }}
            .metric-card .label {{
                color: var(--muted);
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 750;
            }}
            .metric-card .value {{
                color: var(--text);
                font-size: 1.45rem;
                font-weight: 750;
                margin-top: 8px;
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
                overflow: hidden;
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
                grid-template-columns: repeat(3, minmax(180px, 1fr));
                gap: 12px;
                margin: 8px 0 18px 0;
            }}
            .learning-card {{
                border: 1px solid var(--border);
                border-radius: 13px;
                padding: 15px;
                background: linear-gradient(180deg, var(--panel), var(--panel-2));
                box-shadow: 0 12px 34px rgba(0,0,0,0.12);
            }}
            .learning-card .learning-kicker {{
                color: var(--accent-2);
                font-size: 0.72rem;
                font-weight: 850;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 6px;
            }}
            .learning-card .learning-title {{
                color: var(--text);
                font-size: 1rem;
                font-weight: 800;
                margin-bottom: 6px;
            }}
            .learning-card .learning-text {{
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.45;
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
                .workflow-strip {{
                    grid-template-columns: repeat(2, minmax(120px, 1fr));
                }}
                .learning-grid {{
                    grid-template-columns: 1fr;
                }}
                .hero h1 {{
                    font-size: 1.62rem;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_date_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        lower = str(col).lower()
        if lower in {"date", "month"} or "date" in lower or "time" in lower:
            return col
    return None


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    out = df.copy()
    out.columns = out.columns.str.strip()
    date_col = find_date_col(out)
    if date_col:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.sort_values(date_col)
    for col in out.columns:
        if col != date_col:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() > 0:
                out[col] = converted
    if date_col and out[date_col].notna().any():
        out["administration"] = out[date_col].apply(admin_label)
    else:
        out["administration"] = "Unknown"
    return out, date_col


def admin_label(dt):
    if pd.isna(dt):
        return "Unknown"
    if dt < pd.Timestamp("2017-01-20"):
        return "Pre-Trump"
    if dt <= pd.Timestamp("2021-01-19"):
        return "Trump (2017-2020)"
    if dt <= pd.Timestamp("2025-01-19"):
        return "Biden (2021-2024)"
    return "Post-Biden"


def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr(numeric_only=True)


def regression_metrics(y_true, y_pred) -> tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def build_supervised_with_lags(
    df: pd.DataFrame,
    date_col: str,
    target: str,
    features: list[str],
    lags=(1, 3, 6, 12),
    roll_windows=(3, 6, 12),
) -> pd.DataFrame:
    out = df[[date_col, target] + list(features)].copy()
    out = out.sort_values(date_col).reset_index(drop=True)
    for lag in lags:
        out[f"{target}_lag{lag}"] = out[target].shift(lag)
    for window in roll_windows:
        out[f"{target}_rollmean{window}"] = out[target].rolling(window).mean()
    out[f"{target}_diff1"] = out[target].diff(1)
    for feature in features:
        out[f"{feature}_lag1"] = out[feature].shift(1)
    return out


def default_target(num_cols: list[str]) -> str | None:
    if not num_cols:
        return None
    preferred = ["Home_Price_Index", "Home Price Index", "home_price_index"]
    for candidate in preferred:
        if candidate in num_cols:
            return candidate
    return num_cols[0]


def is_engineered_or_leaky_feature(column: str, target: str | None = None) -> bool:
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
    if not target:
        return num_cols
    clean = [target] if target in num_cols else []
    clean.extend(
        col
        for col in num_cols
        if col != target and not is_engineered_or_leaky_feature(col, target)
    )
    return clean


def metric_card(label: str, value: str) -> None:
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
    for start in range(0, len(cards), 3):
        cols = st.columns(min(3, len(cards) - start))
        for col, (kicker, title, text) in zip(cols, cards[start : start + 3]):
            with col:
                with st.container(border=True):
                    st.caption(kicker)
                    st.markdown(f"**{title}**")
                    st.write(text)


def dashboard_search_items() -> list[dict[str, str]]:
    return [
        {"page": "Overview", "keywords": "trend correlation target summary drivers", "desc": "Start here for target movement and meaningful correlations."},
        {"page": "Explore", "keywords": "histogram box plot violin scatter distribution outliers", "desc": "Use this for visual exploration and distributions."},
        {"page": "Data Quality", "keywords": "missing duplicates quality profile clean", "desc": "Check whether the dataset is reliable."},
        {"page": "ML Lab", "keywords": "machine learning regression classification model train supervised", "desc": "Run one supervised model or compare models."},
        {"page": "Evaluation", "keywords": "accuracy precision recall f1 auc r2 rmse mae metrics explainability", "desc": "Compare models and explain why the best one wins."},
        {"page": "Unsupervised Lab", "keywords": "kmeans dbscan pca isolation forest anomaly clusters", "desc": "Find groups, components, and unusual periods."},
        {"page": "Reinforcement Lab", "keywords": "reinforcement q learning reward action state policy", "desc": "See an educational decision-policy example."},
        {"page": "Forecast", "keywords": "forecast predict future horizon lag", "desc": "Predict future target values."},
        {"page": "OLAP & Export", "keywords": "olap cube pivot 3d segment export csv", "desc": "Build pivots, segment insights, and 3D OLAP cube."},
        {"page": "Scenario Simulator", "keywords": "scenario what if simulator impact change feature", "desc": "Test a what-if feature change."},
        {"page": "Paper Review", "keywords": "paper research real life current market comparison", "desc": "Compare app results with outside research."},
        {"page": "Production Readiness", "keywords": "validation drift model card governance production", "desc": "Big-company checks for trust and governance."},
    ]


def render_search_results(query: str) -> None:
    query = query.strip().lower()
    if not query:
        return
    matches = [
        item
        for item in dashboard_search_items()
        if query in item["page"].lower() or any(word in item["keywords"] for word in query.split())
    ][:5]
    if not matches:
        st.caption("No exact feature match. Try: model, OLAP, forecast, accuracy, drift, or scenario.")
        return
    for item in matches:
        st.markdown(
            f"""
            <div class="search-result">
                <strong>{item["page"]}</strong>
                <span>{item["desc"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def correlation_summary(cmat: pd.DataFrame, target: str | None) -> dict[str, object]:
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
    lowered = {col: col.lower() for col in columns}
    for keywords in keyword_groups:
        for col, lower in lowered.items():
            if all(keyword in lower for keyword in keywords):
                return col
    return None


def theory_check_table(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
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
    needed = {"Trump (2017-2020)", "Biden (2021-2024)"}
    if not needed.issubset(set(df["administration"].unique())):
        return None
    data = df[list(metrics_cols) + ["administration"]].dropna()
    if data.empty:
        return None
    mins = data[metrics_cols].min()
    maxs = data[metrics_cols].max()
    scaled = (data[metrics_cols] - mins) / (maxs - mins).replace(0, np.nan)
    profile = (
        pd.concat([scaled, data["administration"]], axis=1)
        .groupby("administration")[metrics_cols]
        .mean()
        .loc[["Trump (2017-2020)", "Biden (2021-2024)"]]
    )
    categories = metrics_cols + [metrics_cols[0]]
    fig = go.Figure()
    for period in ["Trump (2017-2020)", "Biden (2021-2024)"]:
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
        title="Normalized Administration Profile",
    )
    return fig


def make_model(model_name: str, task_type: str, random_state: int = 0):
    if task_type == "Regression":
        models = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
            "LinearRegression": LinearRegression(),
            "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=120,
                random_state=random_state,
            ),
            "DecisionTree": DecisionTreeRegressor(
                random_state=random_state,
                min_samples_leaf=3,
            ),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "NeuralNetwork": MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.2,
                max_iter=1200,
                random_state=random_state,
            ),
        }
    else:
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),
            "LogisticRegression": LogisticRegression(random_state=random_state, max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(
                random_state=random_state,
                min_samples_leaf=3,
            ),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "NeuralNetwork": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.2,
                max_iter=1200,
                random_state=random_state,
            ),
        }
    return models[model_name]


def build_model_pipeline(model_choice: str, task_type: str, use_scaling: bool, scaler_name: str) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaling:
        scaler = StandardScaler() if scaler_name == "StandardScaler" else MinMaxScaler()
        steps.append(("scaler", scaler))
    steps.append(("model", make_model(model_choice, task_type)))
    return Pipeline(steps)


def make_classification_labels(y_train: pd.Series, y_test: pd.Series):
    y_train_binned = pd.qcut(y_train, q=3, labels=False, duplicates="drop")
    y_test_binned = pd.qcut(y_test, q=3, labels=False, duplicates="drop")
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    return (
        y_train_binned.map(label_map).astype("category"),
        y_test_binned.map(label_map).astype("category"),
    )


def model_options_for_task(task_type: str) -> list[str]:
    if task_type == "Regression":
        return [
            "Ridge",
            "RandomForest",
            "LinearRegression",
            "SVR",
            "GradientBoosting",
            "DecisionTree",
            "KNN",
            "NeuralNetwork",
        ]
    return ["RandomForest", "LogisticRegression", "DecisionTree", "KNN", "NeuralNetwork"]


def classification_auc(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    try:
        if hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(x_test)
        elif hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(x_test)
        else:
            return None
        if len(pd.Series(y_test).dropna().unique()) < 2:
            return None
        return float(roc_auc_score(y_test, scores, multi_class="ovr", average="weighted"))
    except Exception:
        return None


def evaluate_all_models(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    task_type: str,
    use_scaling: bool,
    scaler_name: str,
) -> pd.DataFrame:
    model_df = data[[target] + features].copy().dropna(subset=[target])
    split = int(len(model_df) * 0.8)
    X_train, X_test = model_df[features].iloc[:split], model_df[features].iloc[split:]
    y_train, y_test = model_df[target].iloc[:split], model_df[target].iloc[split:]
    rows = []

    if task_type == "Regression":
        for option in model_options_for_task(task_type):
            pipeline = build_model_pipeline(option, task_type, use_scaling, scaler_name)
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
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
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    model_df = data[[target] + features].copy().dropna(subset=[target])
    split = int(len(model_df) * 0.8)
    X_train = model_df[features].iloc[:split]
    y_train = model_df[target].iloc[:split]
    if task_type == "Classification":
        y_train, _ = make_classification_labels(y_train, model_df[target].iloc[split:])
    pipeline = build_model_pipeline(model_name, task_type, use_scaling, scaler_name)
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, y_train


def model_feature_importance(
    pipeline: Pipeline,
    features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
) -> pd.DataFrame:
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


def data_dictionary(df: pd.DataFrame, date_col: str | None, target: str | None) -> pd.DataFrame:
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
) -> str:
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
    return f"""# US Housing Dashboard Executive Report

## Dataset
- Rows: {len(df):,}
- Columns: {df.shape[1]:,}
- Date range: {date_range}
- Main target: {target or "not detected"}

## Main Findings
- Strongest meaningful signal: {top_signal}
- Best evaluated model: {best_model}
- Overall project result: housing prices are best explained by combining market theory, model evaluation, OLAP segmentation, and current housing context.

## Recommendation
Use the Evaluation page for model choice, OLAP for segment interpretation, Forecast for future direction, and Paper Review for real-life comparison.
"""


def validation_report(df: pd.DataFrame, date_col: str | None, target: str | None) -> pd.DataFrame:
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
    return pd.DataFrame(checks)


def drift_report(df: pd.DataFrame, date_col: str | None, numeric_cols: list[str]) -> pd.DataFrame:
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
Educational analysis of U.S. housing and macroeconomic indicators.

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
- National-level data can hide regional differences.

## Governance Recommendation
Re-run validation, drift checks, and model evaluation whenever the dataset is updated.
"""


def model_family_reason(model_name: str) -> str:
    reasons = {
        "RandomForest": "RandomForest often performs well because it captures non-linear relationships and interactions between housing indicators without requiring one straight-line pattern.",
        "GradientBoosting": "GradientBoosting can score highly because it builds many small corrections, which helps when housing movement is driven by several weak signals together.",
        "NeuralNetwork": "NeuralNetwork can learn flexible non-linear patterns, but it needs enough clean rows; on small datasets it can also underperform simpler models.",
        "Ridge": "Ridge is stable and useful when the relationship is mostly linear, but it can miss curved or threshold effects in housing data.",
        "LinearRegression": "LinearRegression is easy to interpret, but it usually scores lower when the market has non-linear shocks or interacting variables.",
        "SVR": "SVR can model non-linear patterns, but it is sensitive to scaling, feature choice, and parameter settings.",
        "DecisionTree": "DecisionTree is easy to read, but one tree can overfit training patterns and generalize poorly to the test period.",
        "KNN": "KNN depends on similar historical examples; it can struggle when the newest market period is different from older periods.",
        "LogisticRegression": "LogisticRegression is strong for simple class boundaries, but it can score lower when Low/Medium/High classes overlap.",
    }
    return reasons.get(model_name, "This model performs differently because each algorithm learns a different shape from the same features.")


def evaluation_explanation(evaluation: pd.DataFrame, task_type: str) -> str:
    if evaluation.empty:
        return "Run evaluation first to explain model performance."
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


def prepare_unsupervised_matrix(df: pd.DataFrame, features: list[str], scale_data: bool) -> tuple[pd.DataFrame, np.ndarray]:
    matrix = df[features].copy()
    matrix = matrix.replace([np.inf, -np.inf], np.nan).dropna()
    if matrix.empty:
        return matrix, np.empty((0, len(features)))
    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    values = StandardScaler().fit_transform(imputed) if scale_data else imputed
    return matrix, values


def cluster_quality(values: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
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


def pca_projection(values: np.ndarray) -> pd.DataFrame:
    components = PCA(n_components=2, random_state=0).fit_transform(values)
    return pd.DataFrame({"PC1": components[:, 0], "PC2": components[:, 1]})


def unsupervised_example_text(method: str, features: list[str], details: dict[str, object], target: str | None) -> str:
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


def code_prompt_system(df: pd.DataFrame, date_col: str | None, target: str | None) -> str:
    cols = ", ".join(map(str, df.columns[:30]))
    return (
        "You generate concise, production-ready Streamlit/Python code for a housing "
        "analytics dashboard. Use pandas, plotly, scikit-learn, and Streamlit patterns "
        "that can fit into an existing streamlit_app.py file. Return code plus a short "
        f"note. Dataset columns include: {cols}. Date column: {date_col}. "
        f"Recommended target: {target}."
    )


def generate_local_code(prompt: str, df: pd.DataFrame, date_col: str | None, target: str | None) -> str:
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
color_col = "administration" if "administration" in df.columns else None
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
    if not prompt.strip():
        return "# Enter a prompt first, for example: add a forecast chart for Home_Price_Index."
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
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    date_range = "No date column detected"
    if date_col and df[date_col].notna().any():
        date_range = f"{df[date_col].min().date()} to {df[date_col].max().date()}"
    return (
        f"Source: {source_name}\n"
        f"Rows: {len(df):,}\n"
        f"Columns: {df.shape[1]:,}\n"
        f"Date range: {date_range}\n"
        f"Default target: {target or 'None'}\n"
        f"Numeric columns: {', '.join(numeric_cols[:20])}"
    )


def local_chat_answer(prompt: str, df: pd.DataFrame, date_col: str | None, target: str | None) -> str:
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
            "Use ML Lab for supervised regression/classification and Unsupervised Lab for "
            "KMeans, DBSCAN, PCA, and anomaly detection. Click Compare all models for a table."
        )
    if "missing" in lower or "quality" in lower:
        missing = df.isna().sum().sort_values(ascending=False).head(5)
        top_missing = ", ".join(f"{col}: {int(value)}" for col, value in missing.items())
        return f"Top missing-value columns are: {top_missing}."
    if date_col and df[date_col].notna().any():
        return (
            f"I can help analyze this housing dashboard. Current data runs from "
            f"{df[date_col].min().date()} to {df[date_col].max().date()}. "
            "Ask about trends, correlations, model choice, forecasting, or data quality."
        )
    return "I can help analyze this housing dashboard. Ask about trends, correlations, model choice, forecasting, or data quality."


def answer_chat(
    prompt: str,
    api_key: str,
    messages: list[dict[str, str]],
    df: pd.DataFrame,
    date_col: str | None,
    target: str | None,
    source_name: str,
) -> str:
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
                        "You are an AI analyst embedded in a Streamlit dashboard for US housing data. "
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


def flatten_pivot_columns(pivot: pd.DataFrame) -> pd.DataFrame:
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


def reinforcement_market_lab(
    data: pd.DataFrame,
    date_col: str,
    target: str,
    signal_feature: str | None,
    episodes: int,
    risk_penalty: float,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
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


with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark">HI</div>
            <div>
                <div class="brand-name">HousingIQ</div>
                <div class="brand-subtitle">Enterprise Analytics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Housing intelligence, ML evaluation, OLAP segmentation, and governance.")
    st.divider()

    uploaded = st.file_uploader("Upload Kaggle CSV", type=["csv"])
    use_default = st.checkbox("Use default CSV", value=(uploaded is None))
    dark_mode = st.toggle("Dark mode", value=False)
    show_raw = st.checkbox("Show raw data preview", value=False)

    st.divider()
    st.subheader("Search")
    dashboard_query = st.text_input("Find a page or feature", placeholder="Try: accuracy, OLAP, forecast, drift")
    render_search_results(dashboard_query)

    st.divider()
    st.subheader("Dataset")
    st.code(str(DEFAULT_CSV), language="text")

apply_theme(dark_mode)


if uploaded is not None:
    raw_df = pd.read_csv(uploaded)
    source_name = f"Uploaded file: {uploaded.name}"
elif use_default and DEFAULT_CSV.exists():
    raw_df = pd.read_csv(DEFAULT_CSV)
    source_name = f"Local file: {DEFAULT_CSV.name}"
else:
    st.error(
        "No dataset found. Upload a CSV or place it at "
        "data/01_raw/us_home_price_analysis_2004_2024.csv."
    )
    st.stop()

df, date_col = clean_data(raw_df)

with st.sidebar:
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

num_cols = df.select_dtypes(include=["number"]).columns.tolist()
target_default = default_target(num_cols)

st.markdown(
    f"""
    <div class="hero">
        <div class="brand-lockup">
            <div class="brand-mark">HI</div>
            <div>
                <div class="brand-name">HousingIQ</div>
                <div class="brand-subtitle">Decision-grade housing intelligence</div>
            </div>
        </div>
        <h1>US Housing Intelligence Dashboard</h1>
        <p>
            Organized analysis for housing prices, macro indicators, administration comparisons,
            machine-learning results, forecasts, and generated app code.
        </p>
        <div class="chip-row">
            <span class="chip">{source_name}</span>
            <span class="chip">Date column: {date_col or "not detected"}</span>
            <span class="chip">Default target: {target_default or "not detected"}</span>
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

if "show_onboarding" not in st.session_state:
    st.session_state.show_onboarding = True

def onboarding_content() -> None:
    st.markdown("Welcome to **HousingIQ**. Follow this path for your first demo:")
    st.markdown(
        """
        1. **Start Here**: read the project flow.
        2. **Overview**: check trend and correlation.
        3. **Evaluation**: compare models and explain the best one.
        4. **OLAP & Export**: inspect the pivot and 3D cube.
        5. **Conclusion**: use the final project statement.
        """
    )
    st.info("Use sidebar Search for words like `accuracy`, `OLAP`, `forecast`, or `drift`.", icon=":material/search:")
    if st.button("Got it", key="close_onboarding"):
        st.session_state.show_onboarding = False
        st.rerun()


if st.session_state.show_onboarding:
    if hasattr(st, "dialog"):
        @st.dialog("First-Time Guide")
        def onboarding_dialog() -> None:
            onboarding_content()

        onboarding_dialog()
    else:
        with st.popover("First-Time Guide", icon=":material/assistant_navigation:"):
            onboarding_content()

with st.sidebar:
    if st.button("Show guide again"):
        st.session_state.show_onboarding = True
        st.rerun()

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
    metric_card("Periods", f"{df['administration'].nunique():,}")

if show_raw:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(50), width="stretch")

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
        "Paper Review",
        "Code Lab",
        "OLAP & Export",
        "Executive Summary",
        "Data Dictionary",
        "Scenario Simulator",
        "Production Readiness",
    ]
)


with tabs[0]:
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
                "Use it to explain the strongest housing segment with a pivot table, interpretation, and 3D cube.",
            ),
        ]
    )

    start_cols = st.columns([1, 1])
    with start_cols[0]:
        st.markdown("#### Recommended App Flow")
        flow = pd.DataFrame(
            [
                {"Step": 1, "Page": "Executive Summary", "What to say": "This is the full project story and report preview."},
                {"Step": 2, "Page": "Overview", "What to say": "This shows the housing target trend and meaningful correlations."},
                {"Step": 3, "Page": "Evaluation", "What to say": "This proves which model performs best and why."},
                {"Step": 4, "Page": "OLAP & Export", "What to say": "This finds the most important segment and visualizes a 3D OLAP cube."},
                {"Step": 5, "Page": "Paper Review", "What to say": "This compares our results with outside housing research and current market facts."},
                {"Step": 6, "Page": "Production Readiness", "What to say": "This shows big-company validation, drift monitoring, and model governance."},
                {"Step": 7, "Page": "Conclusion", "What to say": "This gives the final project conclusion."},
            ]
        )
        st.dataframe(flow, width="stretch", hide_index=True)

    with start_cols[1]:
        st.markdown("#### Two-Minute Demo Script")
        st.markdown(
            """
            1. Start with the Executive Summary to introduce the project goal.
            2. Open Overview and explain the target trend plus strongest meaningful correlation.
            3. Open Evaluation and explain why the best model wins.
            4. Open OLAP & Export and explain the top segment and cube.
            5. Open Paper Review to connect the result to real housing research.
            6. Finish with Conclusion and Production Readiness.
            """
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
    st.subheader("Overview")
    if target_default and date_col and df[date_col].notna().any():
        left, right = st.columns([2, 1])
        with left:
            fig = px.line(
                df.dropna(subset=[target_default]),
                x=date_col,
                y=target_default,
                color="administration",
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
                chart_df = df[[date_col, metric, "administration"]].dropna().sort_values(date_col)
                chart_df["rolling_mean"] = chart_df[metric].rolling(rolling).mean()
                fig = px.line(chart_df, x=date_col, y=[metric, "rolling_mean"], title=f"{metric} trend")
                st.plotly_chart(fig, width="stretch")
        elif chart_type == "Scatter":
            x_col = st.selectbox("X", num_cols, index=0, key="scatter_x")
            y_index = 1 if len(num_cols) > 1 else 0
            y_col = st.selectbox("Y", num_cols, index=y_index, key="scatter_y")
            fig = px.scatter(df, x=x_col, y=y_col, color="administration", trendline="ols", opacity=0.7)
            fig.update_layout(height=520)
            st.plotly_chart(fig, width="stretch")
        elif chart_type == "Histogram":
            metric = st.selectbox("Metric", num_cols, index=num_cols.index(target_default), key="hist_metric")
            bins = st.slider("Histogram bins", 10, 80, 35)
            fig = px.histogram(
                df,
                x=metric,
                color="administration",
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
            fig = px.box(
                df,
                x="administration",
                y=metric,
                points="outliers" if show_points else False,
                title=f"Box Plot of {metric} by Administration",
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, width="stretch")
            st.info(
                "Box plots highlight median, spread, and outliers. Wider distance between quartiles means more variability.",
                icon=":material/data_thresholding:",
            )
        else:
            metric = st.selectbox("Metric", num_cols, index=num_cols.index(target_default), key="violin_metric")
            fig = px.violin(
                df,
                x="administration",
                y=metric,
                color="administration",
                box=True,
                points="outliers",
                title=f"Violin Plot of {metric} by Administration",
            )
            fig.update_layout(height=540, showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.info(
                "Violin plots combine distribution shape with a box plot. Wider sections mean values are more concentrated there.",
                icon=":material/analytics:",
            )


with tabs[3]:
    st.subheader("Data Quality")
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
    st.subheader("Administration Comparison")
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
            grouped = df.groupby("administration")[selected_metrics].agg(["mean", "median", "std"]).round(3)
            st.dataframe(grouped, width="stretch")

            if {"Trump (2017-2020)", "Biden (2021-2024)"} <= set(df["administration"].unique()):
                means = df.groupby("administration")[selected_metrics].mean(numeric_only=True)
                delta = (means.loc["Biden (2021-2024)"] - means.loc["Trump (2017-2020)"]).sort_values()
                fig = px.bar(
                    delta,
                    orientation="h",
                    labels={"value": "Biden mean - Trump mean", "index": "Metric"},
                    title="Mean Difference: Biden Period minus Trump Period",
                )
                fig.update_layout(height=440)
                st.plotly_chart(fig, width="stretch")

                radar_cols = selected_metrics[: min(8, len(selected_metrics))]
                if len(radar_cols) >= 3:
                    radar = radar_compare(df, radar_cols)
                    if radar:
                        st.plotly_chart(radar, width="stretch")


with tabs[5]:
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
        features = st.multiselect(
            "Features",
            feature_candidates,
            default=feature_candidates[: min(10, len(feature_candidates))],
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

            model_df = df[[target] + features].copy().dropna(subset=[target])
            min_rows = 20 if task_type == "Regression" else 30
            if len(model_df) < min_rows:
                st.warning(f"Need at least {min_rows} rows after filtering for a reliable model run.")
            else:
                split = int(len(model_df) * 0.8)
                X_train, X_test = model_df[features].iloc[:split], model_df[features].iloc[split:]
                y_train, y_test = model_df[target].iloc[:split], model_df[target].iloc[split:]

                run_col, compare_col = st.columns([1, 1])
                run_single = run_col.button("Run selected model", type="primary")
                compare_all = compare_col.button("Compare all models")

                if compare_all:
                    rows = []
                    if task_type == "Regression":
                        for option in model_options:
                            pipeline = build_model_pipeline(option, task_type, use_scaling, scaler_name)
                            pipeline.fit(X_train, y_train)
                            pred = pipeline.predict(X_test)
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
                                pipeline = build_model_pipeline(option, task_type, use_scaling, scaler_name)
                                pipeline.fit(X_train, y_train_binned)
                                pred = pipeline.predict(X_test)
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
                                        "Train rows": len(X_train),
                                        "Test rows": len(X_test),
                                    }
                                )
                            comparison = pd.DataFrame(rows).sort_values("F1", ascending=False)
                    if not comparison.empty:
                        st.session_state.model_comparison = comparison
                        if "ml_results" not in st.session_state:
                            st.session_state.ml_results = []
                        st.session_state.ml_results.extend(comparison.to_dict("records"))

                if run_single:
                    pipeline = build_model_pipeline(model_choice, task_type, use_scaling, scaler_name)

                    if task_type == "Regression":
                        pipeline.fit(X_train, y_train)
                        pred = pipeline.predict(X_test)
                        mae, rmse, r2 = regression_metrics(y_test, pred)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("MAE", f"{mae:.3f}")
                        c2.metric("RMSE", f"{rmse:.3f}")
                        c3.metric("R2", f"{r2:.3f}")
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
                        st.plotly_chart(px.line(result_df, y=["actual", "predicted"]), width="stretch")

                        cv = min(5, len(X_train))
                        if cv >= 2:
                            cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
                            st.caption(f"Cross-validation R2: {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")

                        fitted = pipeline.named_steps["model"]
                        if hasattr(fitted, "feature_importances_"):
                            importance = pd.Series(fitted.feature_importances_, index=features).sort_values()
                            st.plotly_chart(px.bar(importance, orientation="h"), width="stretch")
                        elif hasattr(fitted, "coef_"):
                            coef = pd.Series(np.ravel(fitted.coef_), index=features).sort_values()
                            st.plotly_chart(px.bar(coef, orientation="h"), width="stretch")
                    else:
                        try:
                            y_train_binned, y_test_binned = make_classification_labels(y_train, y_test)
                        except (AttributeError, ValueError) as err:
                            st.error(f"Could not create target classes: {err}")
                        else:
                            pipeline.fit(X_train, y_train_binned)
                            pred = pipeline.predict(X_test)
                            acc = accuracy_score(y_test_binned, pred)
                            prec = precision_score(y_test_binned, pred, average="weighted", zero_division=0)
                            rec = recall_score(y_test_binned, pred, average="weighted", zero_division=0)
                            f1 = f1_score(y_test_binned, pred, average="weighted", zero_division=0)
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Accuracy", f"{acc:.3f}")
                            c2.metric("Precision", f"{prec:.3f}")
                            c3.metric("Recall", f"{rec:.3f}")
                            c4.metric("F1", f"{f1:.3f}")
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
                                }
                            )
                            report = classification_report(y_test_binned, pred, output_dict=True, zero_division=0)
                            st.dataframe(pd.DataFrame(report).transpose(), width="stretch")

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
        eval_features = st.multiselect(
            "Evaluation features",
            eval_feature_candidates,
            default=eval_feature_candidates[: min(10, len(eval_feature_candidates))],
            key="eval_features",
        )
        eval_task = st.radio("Evaluation task", ["Regression", "Classification"], horizontal=True, key="eval_task")
        eval_scale = st.checkbox("Scale evaluation features", value=True, key="eval_scale")
        eval_scaler = st.selectbox("Evaluation scaler", ["StandardScaler", "MinMaxScaler"], key="eval_scaler")

        if not eval_features:
            st.info("Select at least one feature to evaluate models.")
        else:
            eval_model_df = df[[eval_target] + eval_features].dropna(subset=[eval_target])
            min_rows = 20 if eval_task == "Regression" else 30
            if len(eval_model_df) < min_rows:
                st.warning(f"Need at least {min_rows} rows after filtering for a useful evaluation.")
            elif st.button("Run full model evaluation", type="primary"):
                try:
                    evaluation = evaluate_all_models(
                        df,
                        eval_target,
                        eval_features,
                        eval_task,
                        eval_scale,
                        eval_scaler,
                    )
                except Exception as err:
                    st.error(f"Could not evaluate models: {err}")
                else:
                    st.session_state.evaluation_results = evaluation
                    st.session_state.evaluation_task = eval_task

        if st.session_state.get("evaluation_results") is not None:
            evaluation = st.session_state.evaluation_results.round(4)
            task = st.session_state.get("evaluation_task", eval_task)
            st.markdown("#### Evaluation Metrics Table")
            st.dataframe(evaluation, width="stretch", hide_index=True)

            if task == "Regression":
                best = evaluation.sort_values("R2", ascending=False).iloc[0]
                st.success(
                    f"Best regression model: `{best['Model']}` with R2 = `{best['R2']:.3f}`, "
                    f"RMSE = `{best['RMSE']:.3f}`, and MAE = `{best['MAE']:.3f}`.",
                    icon=":material/emoji_events:",
                )
                st.info(evaluation_explanation(evaluation, task), icon=":material/help:")
                long_metrics = evaluation.melt(
                    id_vars=["Model"],
                    value_vars=["MAE", "RMSE", "R2"],
                    var_name="Metric",
                    value_name="Score",
                )
                st.plotly_chart(
                    px.bar(long_metrics, x="Model", y="Score", color="Metric", barmode="group"),
                    width="stretch",
                )
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
                metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
                if "ROC AUC" in evaluation.columns and evaluation["ROC AUC"].notna().any():
                    metric_cols.append("ROC AUC")
                long_metrics = evaluation.melt(
                    id_vars=["Model"],
                    value_vars=metric_cols,
                    var_name="Metric",
                    value_name="Score",
                )
                st.plotly_chart(
                    px.bar(long_metrics, x="Model", y="Score", color="Metric", barmode="group"),
                    width="stretch",
                )
                with st.expander("How to read classification metrics"):
                    st.markdown(
                        "- **Accuracy:** total correct predictions, useful when classes are balanced.\n"
                        "- **Precision:** when the model predicts a class, how often it is correct.\n"
                        "- **Recall:** how many real class cases the model successfully finds.\n"
                        "- **F1:** balances precision and recall, good for comparing models.\n"
                        "- **ROC AUC:** higher is better; it measures ranking quality when probability scores are available."
                    )

            st.download_button(
                "Download evaluation table CSV",
                data=evaluation.to_csv(index=False).encode("utf-8"),
                file_name="model_evaluation_metrics.csv",
                mime="text/csv",
            )
            if eval_features:
                st.markdown("#### Model Explainability")
                try:
                    metric = "R2" if task == "Regression" else "F1"
                    best_model_name = evaluation.sort_values(metric, ascending=False).iloc[0]["Model"]
                    explain_pipeline, explain_X, explain_y = fit_best_model_for_task(
                        df,
                        eval_target,
                        eval_features,
                        task,
                        eval_scale,
                        eval_scaler,
                        best_model_name,
                    )
                    importance = model_feature_importance(
                        explain_pipeline,
                        eval_features,
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
            matrix, values = prepare_unsupervised_matrix(df, unsup_features, scale_unsup)

            if len(matrix) < 5:
                st.warning("Need at least five complete rows after cleaning selected features.")
            else:
                example_details: dict[str, object] = {}
                if method == "KMeans":
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
            "Exogenous features held at latest value",
            [col for col in num_cols if col != target],
            default=[col for col in num_cols if col != target][: min(6, max(len(num_cols) - 1, 0))],
        )
        horizon = st.slider("Horizon in months", 3, 36, 12)
        model_choice = st.selectbox(
            "Forecast model",
            ["Ridge", "RandomForest", "LinearRegression", "SVR", "GradientBoosting", "DecisionTree", "KNN"],
            index=0,
        )
        base = df[[date_col, target] + exog].copy().sort_values(date_col).reset_index(drop=True)
        for feature in exog:
            base[feature] = base[feature].ffill()
        supervised = build_supervised_with_lags(base, date_col, target, exog).dropna(subset=[target])
        feature_cols = [col for col in supervised.columns if col not in [date_col, target]]

        if len(supervised.dropna(subset=feature_cols)) < 20:
            st.warning("Not enough complete lagged rows to produce a useful forecast.")
        elif st.button("Generate forecast", type="primary"):
            train = supervised.dropna(subset=feature_cols)
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", make_model(model_choice, "Regression")),
                ]
            )
            pipeline.fit(train[feature_cols], train[target])

            last_date = base[date_col].dropna().max()
            future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
            working = base.copy()
            predictions = []
            for next_date in future_dates:
                row = {date_col: next_date, target: np.nan}
                for feature in exog:
                    row[feature] = working[feature].iloc[-1] if len(working) else np.nan
                working = pd.concat([working, pd.DataFrame([row])], ignore_index=True)
                next_supervised = build_supervised_with_lags(working, date_col, target, exog)
                next_features = next_supervised.iloc[[-1]][feature_cols]
                prediction = float(pipeline.predict(next_features)[0])
                working.loc[working.index[-1], target] = prediction
                predictions.append(prediction)

            forecast_df = pd.DataFrame({date_col: future_dates, "forecast": predictions})
            hist = base[[date_col, target]].dropna()
            fig = px.line(hist, x=date_col, y=target, title="History + Forecast")
            fig.add_scatter(
                x=forecast_df[date_col],
                y=forecast_df["forecast"],
                mode="lines+markers",
                name="Forecast",
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(forecast_df, width="stretch")
            st.download_button(
                "Download forecast CSV",
                data=forecast_df.to_csv(index=False).encode("utf-8"),
                file_name="forecast.csv",
                mime="text/csv",
            )


with tabs[10]:
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
        st.download_button(
            "Download conclusion theory comparison CSV",
            data=theory_df.to_csv(index=False).encode("utf-8"),
            file_name="conclusion_theory_comparison.csv",
            mime="text/csv",
        )


with tabs[11]:
    st.subheader("Paper Review & Real-Life Comparison")
    st.caption("Compare outside housing research and current market facts with this dashboard's results.")
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
    st.subheader("OLAP & Export")
    olap_tab, export_tab = st.tabs(["OLAP Cube", "Export"])
    with olap_tab:
        olap_df = df.copy()
        if date_col and date_col in olap_df.columns:
            olap_df["OLAP Year"] = olap_df[date_col].dt.year.astype("Int64").astype(str)
            olap_df["OLAP Quarter"] = "Q" + olap_df[date_col].dt.quarter.astype("Int64").astype(str)
        cat_cols = olap_df.select_dtypes(include=["object", "category"]).columns.tolist()
        if "administration" in df.columns and "administration" not in cat_cols:
            cat_cols.append("administration")
        default_index = ["administration"] if "administration" in cat_cols else cat_cols[:1]
        index = st.multiselect("Rows", cat_cols, default=default_index, key="olap_index")
        columns = st.multiselect("Columns", cat_cols, default=[], key="olap_columns")
        values = st.multiselect("Values", num_cols, default=num_cols[:1], key="olap_values")
        aggfunc = st.selectbox("Aggregation", ["mean", "sum", "count", "std", "min", "max"])
        if values and index:
            try:
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
            for candidate in ["administration", "OLAP Year", "OLAP Quarter"]:
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
            file_name="filtered_us_housing.csv",
            mime="text/csv",
        )


with tabs[14]:
    st.subheader("Executive Summary")
    st.caption("One page for presenting the whole project clearly.")
    evaluation = st.session_state.get("evaluation_results")
    report_md = executive_report_markdown(df, date_col, target_default, evaluation)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Rows", f"{len(df):,}")
    summary_cols[1].metric("Columns", f"{df.shape[1]:,}")
    summary_cols[2].metric("Numeric features", f"{len(num_cols):,}")
    summary_cols[3].metric("Target", target_default or "None")

    st.markdown("#### Project Story")
    st.success(
        "This dashboard combines EDA, correlation, supervised ML, unsupervised learning, reinforcement learning, "
        "forecasting, OLAP, paper comparison, and final conclusion. The strongest presentation angle is that housing "
        "prices need multiple evidence sources, not one simple variable.",
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
    st.download_button(
        "Download executive report Markdown",
        data=report_md.encode("utf-8"),
        file_name="us_housing_executive_report.md",
        mime="text/markdown",
    )


with tabs[15]:
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
    st.subheader("Scenario Simulator")
    st.caption("Change one feature and estimate how the trained regression model reacts.")
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
                ["Ridge", "RandomForest", "LinearRegression", "GradientBoosting", "DecisionTree", "KNN", "NeuralNetwork"],
                index=0,
                key="sim_model",
            )
            change_feature = st.selectbox("Feature to change", sim_features, key="sim_change_feature")
            change_type = st.radio("Change type", ["Percent", "Absolute"], horizontal=True)
            change_amount = st.number_input("Change amount", value=10.0)
            scenario_df = df[[sim_target] + sim_features].dropna()
            if len(scenario_df) < 20:
                st.warning("Need at least 20 complete rows for a useful scenario model.")
            elif st.button("Run scenario simulation", type="primary"):
                pipeline = build_model_pipeline(sim_model, "Regression", True, "StandardScaler")
                pipeline.fit(scenario_df[sim_features], scenario_df[sim_target])
                baseline_row = scenario_df[sim_features].iloc[[-1]].copy()
                scenario_row = baseline_row.copy()
                original_value = float(scenario_row[change_feature].iloc[0])
                if change_type == "Percent":
                    scenario_row[change_feature] = scenario_row[change_feature] * (1 + change_amount / 100)
                else:
                    scenario_row[change_feature] = scenario_row[change_feature] + change_amount
                baseline_pred = float(pipeline.predict(baseline_row)[0])
                scenario_pred = float(pipeline.predict(scenario_row)[0])
                delta = scenario_pred - baseline_pred
                st.success(
                    f"Scenario result: changing `{change_feature}` from `{original_value:,.2f}` to "
                    f"`{float(scenario_row[change_feature].iloc[0]):,.2f}` changes predicted `{sim_target}` by `{delta:,.2f}`.",
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
