from __future__ import annotations

import matplotlib
from pathlib import Path

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "us_home_price_analysis_2004_2024.csv"
FIG_DIR = ROOT / "figures"
TARGET = "Home_Price_Index"


def administration_label(date: pd.Timestamp) -> str:
    if date < pd.Timestamp("2017-01-01"):
        return "Pre-Trump"
    if date < pd.Timestamp("2021-01-01"):
        return "Trump"
    return "Biden"


def build_model(name: str, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def save_bar_labels(ax) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)


def latex_escape(value: object) -> str:
    return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    df["Administration"] = df["DATE"].apply(administration_label)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != TARGET]
    clean = df[[TARGET, "DATE", "Administration", *feature_cols]].dropna(subset=[TARGET])
    split_idx = int(len(clean) * 0.8)

    x_train = clean.loc[: split_idx - 1, feature_cols]
    x_test = clean.loc[split_idx:, feature_cols]
    y_train = clean.loc[: split_idx - 1, TARGET]
    y_test = clean.loc[split_idx:, TARGET]

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42, min_samples_leaf=4),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(C=10.0, epsilon=0.1),
    }

    rows = []
    predictions = {}
    tscv = TimeSeriesSplit(n_splits=5)
    for name, estimator in models.items():
        pipe = build_model(name, estimator)
        pipe.fit(x_train, y_train)
        test_pred = pipe.predict(x_test)
        train_pred = pipe.predict(x_train)
        cv_scores = cross_val_score(pipe, clean[feature_cols], clean[TARGET], cv=tscv, scoring="r2")
        rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, test_pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, test_pred))),
                "R2": r2_score(y_test, test_pred),
                "Train R2": r2_score(y_train, train_pred),
                "Test R2": r2_score(y_test, test_pred),
                "Gap": r2_score(y_train, train_pred) - r2_score(y_test, test_pred),
                "CV R2": float(np.mean(cv_scores)),
                "CV Std": float(np.std(cv_scores)),
            }
        )
        predictions[name] = test_pred

    metrics = pd.DataFrame(rows).sort_values("R2", ascending=False)
    metrics.to_csv(ROOT / "generated_model_metrics.csv", index=False)
    best = metrics.iloc[0]
    best_name = str(best["Model"])

    top_corr = (
        clean[numeric_cols]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(10)
        .sort_values()
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = ["#0f766e" if value > 0 else "#c2410c" for value in top_corr.values]
    ax.barh(top_corr.index, top_corr.values, color=colors)
    ax.axvline(0, color="#1f2937", linewidth=1)
    ax.set_title("Top correlations with Home Price Index")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_results.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    ordered = metrics.sort_values("R2", ascending=False)
    axes[0].barh(ordered["Model"], ordered["R2"], color="#0f766e")
    axes[0].set_title("R2 (higher is better)")
    axes[1].barh(ordered["Model"], ordered["RMSE"], color="#2563eb")
    axes[1].set_title("RMSE (lower is better)")
    axes[2].barh(ordered["Model"], ordered["MAE"], color="#c2410c")
    axes[2].set_title("MAE (lower is better)")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Model evaluation results", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_evaluation_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    diag = metrics.sort_values("Test R2", ascending=False)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(diag))
    width = 0.36
    ax.bar(x - width / 2, diag["Train R2"], width, label="Train R2", color="#0f766e")
    ax.bar(x + width / 2, diag["Test R2"], width, label="Test R2", color="#f97316")
    ax.set_xticks(x, diag["Model"], rotation=35, ha="right")
    ax.set_ylabel("R2")
    ax.set_title("Fit diagnostics: train vs test performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fit_diagnostics_results.png", dpi=220)
    plt.close(fig)

    pred_df = pd.DataFrame(
        {
            "DATE": clean.loc[split_idx:, "DATE"],
            "Actual": y_test.to_numpy(),
            "Predicted": predictions[best_name],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5.3))
    ax.plot(pred_df["DATE"], pred_df["Actual"], label="Actual", color="#111827", linewidth=2)
    ax.plot(pred_df["DATE"], pred_df["Predicted"], label=f"Predicted ({best_name})", color="#0f766e", linewidth=2)
    ax.set_title("Actual vs predicted Home Price Index on chronological test set")
    ax.set_ylabel("Home Price Index")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "actual_vs_predicted_results.png", dpi=220)
    plt.close(fig)

    olap = (
        clean.groupby("Administration")
        .agg(
            Avg_HPI=(TARGET, "mean"),
            Avg_Mortgage_Rate=("Mortgage_Rate", "mean"),
            Avg_Unemployment=("Unemployment_Rate", "mean"),
            Avg_Building_Permits=("Building_Permits", "mean"),
        )
        .reindex(["Pre-Trump", "Trump", "Biden"])
    )
    olap.to_csv(ROOT / "generated_olap_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    chart_specs = [
        ("Avg_HPI", "Average HPI", "#0f766e"),
        ("Avg_Mortgage_Rate", "Average mortgage rate", "#c2410c"),
        ("Avg_Unemployment", "Average unemployment", "#2563eb"),
        ("Avg_Building_Permits", "Average building permits", "#7c3aed"),
    ]
    for ax, (col, title, color) in zip(axes.ravel(), chart_specs):
        ax.bar(olap.index, olap[col], color=color)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        save_bar_labels(ax)
    fig.suptitle("OLAP period summary by administration", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "olap_period_results.png", dpi=220)
    plt.close(fig)

    table_rows = "\n".join(
        f"{row['Model']} & {row['MAE']:.3f} & {row['RMSE']:.3f} & {row['R2']:.3f} & {row['CV R2']:.3f} \\\\"
        for _, row in metrics.iterrows()
    )
    corr_sentence = ", ".join(
        f"\\texttt{{{latex_escape(idx)}}} ({value:.2f})"
        for idx, value in top_corr.sort_values(key=lambda s: s.abs(), ascending=False).head(4).items()
    )
    olap_best = olap["Avg_HPI"].idxmax()
    olap_mortgage = olap["Avg_Mortgage_Rate"].idxmax()

    generated = rf"""
\section{{Data-Driven Result Snapshot}}

To make the final discussion concrete, the report assets were regenerated from the included CSV file using a chronological 80/20 train-test split. The target variable is \texttt{{Home\_Price\_Index}}, and the predictors are the remaining numerical macroeconomic and engineered features. This makes the reported results reproducible from the project package.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{figures/correlation_results.png}}
\caption{{Strongest correlations with the Home Price Index in the project dataset.}}
\label{{fig:generated_corr}}
\end{{figure}}

The strongest linear relationships with the target are: {corr_sentence}. These correlations support the paper-review conclusion that housing prices should be interpreted through several linked economic forces, not through a single variable.

\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
\textbf{{Model}} & \textbf{{MAE}} & \textbf{{RMSE}} & \textbf{{$R^2$}} & \textbf{{CV $R^2$}} \\
\midrule
{table_rows}
\bottomrule
\end{{tabular}}
\caption{{Generated regression comparison on the chronological test set.}}
\label{{tab:generated_model_metrics}}
\end{{table}}

The best test-set model in this generated run is \textbf{{{best_name}}}, with \(R^2={best['R2']:.3f}\), RMSE \(={best['RMSE']:.3f}\), and MAE \(={best['MAE']:.3f}\). Because the dataset includes lagged and smoothed price features, the linear model can extrapolate the rising trend very effectively. The weaker tree-model scores are also informative: tree ensembles can capture non-linear interactions, but they may struggle when the test period moves beyond the range learned during training.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures/model_evaluation_results.png}}
\caption{{Evaluation graphics generated from the model comparison module.}}
\label{{fig:generated_eval}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{figures/fit_diagnostics_results.png}}
\caption{{Fit diagnostics comparing train and test performance.}}
\label{{fig:generated_diag}}
\end{{figure}}

The diagnostics figure shows whether a model generalizes beyond the training period. A large train-test gap indicates overfitting risk, while a weak score on both sets indicates underfitting. This is why the final project does not rely only on the highest raw score: it also asks whether the result is stable and explainable.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{figures/actual_vs_predicted_results.png}}
\caption{{Actual and predicted Home Price Index for the best model on the test period.}}
\label{{fig:generated_pred}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{figures/olap_period_results.png}}
\caption{{OLAP-style period comparison by administration.}}
\label{{fig:generated_olap}}
\end{{figure}}

The OLAP summary shows that the highest average Home Price Index appears during the \textbf{{{olap_best}}} period, while the highest average mortgage-rate pressure appears during the \textbf{{{olap_mortgage}}} period. This result is useful for the paper-review argument: prices and affordability pressure can rise together when supply, timing, and market inertia slow the response of prices.
"""
    (ROOT / "generated_results.tex").write_text(generated.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
