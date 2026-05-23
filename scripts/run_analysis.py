"""Reproducible analysis workflow for the BMP explainable-ML manuscript.

This script reproduces the analysis outputs used in the manuscript:

1. Model benchmarking by repeated cross-validation.
2. OLS regression and interaction/slope-difference tests.
3. SHAP interpretation and bootstrap SHAP stability.
4. Stratified validation of the Lignin-DM association.
5. Manuscript and supplementary figures/tables.

Run from the repository root:

    python scripts/run_analysis.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
DM_FILTER = 15.0
N_BOOT = 100
N_BOOT_CORR = 5000

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "table_complete.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

FEATURES = [
    "DM Mean (% FM)",
    "VS Mean (% FM)",
    "C/N",
    "Carbon Mean (% DM)",
    "Hydrogen Mean (% DM)",
    "Nitrogen Mean (% DM)",
    "Sulfur Mean (% DM)",
    "Oxygen Mean (% DM)",
    "Cellulose Mean (g/100g DM)",
    "Hemicelluloses Mean (g/100g DM)",
    "Lignin Mean (g/100g DM)",
]

SHORT = {
    "DM Mean (% FM)": "DM",
    "VS Mean (% FM)": "VS",
    "C/N": "C/N",
    "Carbon Mean (% DM)": "Carbon",
    "Hydrogen Mean (% DM)": "Hydrogen",
    "Nitrogen Mean (% DM)": "Nitrogen",
    "Sulfur Mean (% DM)": "Sulfur",
    "Oxygen Mean (% DM)": "Oxygen",
    "Cellulose Mean (g/100g DM)": "Cellulose",
    "Hemicelluloses Mean (g/100g DM)": "Hemicelluloses",
    "Lignin Mean (g/100g DM)": "Lignin",
}

TARGET = "BMP_DM"
BMP_VS_COL = "BMP exp Mean (Nm3 CH4/t VS)"
VS_DM_RATIO_COL = "VS/DM Mean (% FM)"

RF_PARAMS = {
    "n_estimators": 50,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "random_state": SEED,
    "n_jobs": 1,
}


@dataclass
class OLSResult:
    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    r2: float
    adj_r2: float
    rmse: float
    sse: float
    df_resid: int
    names: list[str]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (FIG_DIR, TABLE_DIR):
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()


def clean_label(name: str) -> str:
    return SHORT.get(name, name)


def rf_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH, na_values="---", thousands=",")
    for col in FEATURES + [BMP_VS_COL, VS_DM_RATIO_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Lallement reports BMP per t VS and VS/DM as a fractional ratio.
    df[TARGET] = df[BMP_VS_COL] * df[VS_DM_RATIO_COL]
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    filtered = df[df["DM Mean (% FM)"] >= DM_FILTER].copy()
    filtered = filtered.reset_index(drop=True)

    X = filtered[FEATURES].copy()
    y = filtered[TARGET].copy()
    return filtered, X, y


def descriptive_tables(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    desc = X.describe().T
    desc["r_BMP"] = [stats.pearsonr(X[col], y)[0] for col in X.columns]
    desc = desc.rename(
        columns={
            "count": "n",
            "mean": "Mean",
            "std": "SD",
            "min": "Min",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Max",
            "r_BMP": "r_BMP",
        }
    )
    desc.insert(0, "Feature", [clean_label(c) for c in X.columns])

    bmp_row = {
        "Feature": "BMP",
        "n": len(y),
        "Mean": y.mean(),
        "SD": y.std(),
        "Min": y.min(),
        "Q1": y.quantile(0.25),
        "Median": y.median(),
        "Q3": y.quantile(0.75),
        "Max": y.max(),
        "r_BMP": 1.0,
    }
    out = pd.concat([desc, pd.DataFrame([bmp_row])], ignore_index=True)
    out.to_csv(TABLE_DIR / "TableS1_descriptive_statistics.csv", index=False)
    return out


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv5 = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rcv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)

    models = {
        "OLS Linear Regression": LinearRegression(),
        "Ridge Regression (alpha=1)": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(**RF_PARAMS),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=SEED,
        ),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=1),
    }

    rows = []
    for name, model in models.items():
        pipe = rf_pipeline(model)
        r2_5 = cross_val_score(pipe, X, y, cv=cv5, scoring="r2")
        rmse_5 = -cross_val_score(pipe, X, y, cv=cv5, scoring="neg_root_mean_squared_error")
        r2_rep = cross_val_score(pipe, X, y, cv=rcv, scoring="r2")
        rmse_rep = -cross_val_score(pipe, X, y, cv=rcv, scoring="neg_root_mean_squared_error")
        rows.append(
            {
                "Model": name,
                "R2_5fold_mean": r2_5.mean(),
                "R2_5fold_sd": r2_5.std(),
                "RMSE_5fold_mean": rmse_5.mean(),
                "RMSE_5fold_sd": rmse_5.std(),
                "R2_repeated_mean": r2_rep.mean(),
                "R2_repeated_sd": r2_rep.std(),
                "RMSE_repeated_mean": rmse_rep.mean(),
                "RMSE_repeated_sd": rmse_rep.std(),
            }
        )

    out = pd.DataFrame(rows).sort_values("R2_repeated_mean", ascending=False)
    plot_model_comparison(out)
    return out


def plot_model_comparison(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1f77b4" if "Linear" not in m and "Ridge" not in m else "#ff7f0e" for m in table["Model"]]
    y_pos = np.arange(len(table))

    axes[0].barh(y_pos, table["R2_repeated_mean"], color=colors, edgecolor="0.4")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(table["Model"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean R2 (repeated 5 x 10 CV)")
    axes[0].set_xlim(0, 0.7)
    axes[0].grid(False)
    for i, val in enumerate(table["R2_repeated_mean"]):
        axes[0].text(val + 0.01, i, f"{val:.3f}", va="center")

    axes[1].barh(y_pos, table["RMSE_repeated_mean"], color=colors, edgecolor="0.4")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(table["Model"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean RMSE (Nm3 CH4/t DM)")
    axes[1].grid(False)
    for i, val in enumerate(table["RMSE_repeated_mean"]):
        axes[1].text(val + 0.5, i, f"{val:.1f}", va="center")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig1_model_comparison.png", dpi=300)
    plt.close(fig)


def fit_ols(matrix: np.ndarray, y: Iterable[float], names: list[str]) -> OLSResult:
    y_arr = np.asarray(y, dtype=float)
    x = np.asarray(matrix, dtype=float)
    n = len(y_arr)
    p = x.shape[1]
    beta = np.linalg.lstsq(x, y_arr, rcond=None)[0]
    pred = x @ beta
    resid = y_arr - pred
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1.0 - sse / sst
    df_resid = n - p
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid
    mse = sse / df_resid
    cov = mse * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    t_stat = beta / se
    p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df_resid))
    return OLSResult(beta, se, t_stat, p_vals, r2, adj_r2, math.sqrt(sse / n), sse, df_resid, names)


def f_test_nested(base: OLSResult, full: OLSResult, df_diff: int) -> tuple[float, float, float]:
    f_val = ((base.sse - full.sse) / df_diff) / (full.sse / full.df_resid)
    p_val = 1.0 - stats.f.cdf(f_val, df_diff, full.df_resid)
    delta_r2 = full.r2 - base.r2
    return float(f_val), float(p_val), float(delta_r2)


def ols_and_interaction_tests(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(y)
    x_base = np.column_stack([np.ones(n), X.values])
    base_names = ["Intercept"] + [clean_label(c) for c in X.columns]
    base = fit_ols(x_base, y, base_names)

    coef = pd.DataFrame(
        {
            "Predictor": base.names,
            "Beta": base.beta,
            "SE": base.se,
            "t": base.t,
            "p": base.p,
        }
    )
    lignin = X["Lignin Mean (g/100g DM)"].to_numpy()
    dm = X["DM Mean (% FM)"].to_numpy()
    lignin_c = lignin - lignin.mean()
    dm_c = dm - dm.mean()
    high_dm = (dm > np.median(dm)).astype(float)

    # Full-feature continuous interaction: conservative classical test.
    x_full_cont = np.column_stack([x_base, lignin_c * dm_c])
    full_cont = fit_ols(x_full_cont, y, base_names + ["Lignin_c_x_DM_c"])
    f_cont, p_cont, dr2_cont = f_test_nested(base, full_cont, 1)

    # Simple slope-difference model: directly tests the stratified pattern.
    x_slope_base = np.column_stack([np.ones(n), lignin_c, high_dm])
    slope_base = fit_ols(x_slope_base, y, ["Intercept", "Lignin_c", "High_DM"])
    x_slope_full = np.column_stack([x_slope_base, lignin_c * high_dm])
    slope_full = fit_ols(x_slope_full, y, ["Intercept", "Lignin_c", "High_DM", "Lignin_c_x_High_DM"])
    f_slope, p_slope, dr2_slope = f_test_nested(slope_base, slope_full, 1)

    # Full-feature binary interaction sensitivity.
    x_full_binary = np.column_stack([x_base, lignin_c * high_dm])
    full_binary = fit_ols(x_full_binary, y, base_names + ["Lignin_c_x_High_DM"])
    f_binary, p_binary, dr2_binary = f_test_nested(base, full_binary, 1)

    rows = [
        {
            "Test": "Overall OLS feature-set test",
            "Model": "BMP ~ all 11 features",
            "R2": base.r2,
            "Adjusted_R2": base.adj_r2,
            "RMSE": base.rmse,
            "F": (base.r2 / (x_base.shape[1] - 1)) / ((1 - base.r2) / base.df_resid),
            "df1": x_base.shape[1] - 1,
            "df2": base.df_resid,
            "p": 1.0 - stats.f.cdf((base.r2 / (x_base.shape[1] - 1)) / ((1 - base.r2) / base.df_resid), x_base.shape[1] - 1, base.df_resid),
            "Delta_R2": np.nan,
            "Interpretation": "Feature set is collectively predictive; individual coefficients are unstable under multicollinearity.",
        },
        {
            "Test": "Continuous interaction sensitivity",
            "Model": "All features + centered Lignin x centered DM",
            "R2": full_cont.r2,
            "Adjusted_R2": full_cont.adj_r2,
            "RMSE": full_cont.rmse,
            "F": f_cont,
            "df1": 1,
            "df2": full_cont.df_resid,
            "p": p_cont,
            "Delta_R2": dr2_cont,
            "Interpretation": "No evidence for a simple linear multiplicative interaction after all covariates.",
        },
        {
            "Test": "Median-DM slope-difference test",
            "Model": "BMP ~ Lignin + High_DM + Lignin x High_DM",
            "R2": slope_full.r2,
            "Adjusted_R2": slope_full.adj_r2,
            "RMSE": slope_full.rmse,
            "F": f_slope,
            "df1": 1,
            "df2": slope_full.df_resid,
            "p": p_slope,
            "Delta_R2": dr2_slope,
            "Interpretation": "The Lignin-BMP slope is steeper in low-DM samples than high-DM samples.",
        },
        {
            "Test": "Full-feature binary interaction sensitivity",
            "Model": "All features + Lignin x High_DM",
            "R2": full_binary.r2,
            "Adjusted_R2": full_binary.adj_r2,
            "RMSE": full_binary.rmse,
            "F": f_binary,
            "df1": 1,
            "df2": full_binary.df_resid,
            "p": p_binary,
            "Delta_R2": dr2_binary,
            "Interpretation": "Directionally consistent but attenuated after adding collinear covariates.",
        },
    ]
    tests = pd.DataFrame(rows)

    slope_details = pd.DataFrame(
        {
            "Parameter": slope_full.names,
            "Beta": slope_full.beta,
            "SE": slope_full.se,
            "t": slope_full.t,
            "p": slope_full.p,
        }
    )

    plot_slope_validation(X, y, slope_full)
    return coef, tests


def plot_slope_validation(X: pd.DataFrame, y: pd.Series, slope_model: OLSResult) -> None:
    lignin = X["Lignin Mean (g/100g DM)"]
    dm = X["DM Mean (% FM)"]
    median_dm = dm.median()
    low = dm <= median_dm
    high = ~low

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(lignin[low], y[low], color="#1f77b4", alpha=0.65, edgecolor="0.4", label=f"Low DM <= {median_dm:.1f}%")
    ax.scatter(lignin[high], y[high], color="#ff7f0e", alpha=0.65, edgecolor="0.4", label=f"High DM > {median_dm:.1f}%")

    x_line = np.linspace(lignin.min(), lignin.max(), 100)
    for mask, color in [(low, "#1f77b4"), (high, "#ff7f0e")]:
        slope, intercept, *_ = stats.linregress(lignin[mask], y[mask])
        ax.plot(x_line, intercept + slope * x_line, color=color, linestyle="--", lw=2)

    r_low = stats.pearsonr(lignin[low], y[low])[0]
    r_high = stats.pearsonr(lignin[high], y[high])[0]
    ax.text(0.03, 0.95, f"Low DM r = {r_low:.3f}", transform=ax.transAxes, color="#1f77b4", va="top")
    ax.text(0.03, 0.89, f"High DM r = {r_high:.3f}", transform=ax.transAxes, color="#ff7f0e", va="top")
    ax.set_xlabel("Lignin (g/100g DM)")
    ax.set_ylabel("BMP (Nm3 CH4/t DM)")
    ax.legend(frameon=False)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig4_stratified_lignin_DM_validation.png", dpi=300)
    plt.close(fig)


def stratified_validation(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    lignin = X["Lignin Mean (g/100g DM)"]
    dm = X["DM Mean (% FM)"]
    median_dm = dm.median()
    rows = []

    masks = {
        f"Low DM <= {median_dm:.1f}%": dm <= median_dm,
        f"High DM > {median_dm:.1f}%": dm > median_dm,
        "All samples": pd.Series(True, index=X.index),
    }
    for label, mask in masks.items():
        r, p = stats.pearsonr(lignin[mask], y[mask])
        rho, sp = stats.spearmanr(lignin[mask], y[mask])
        slope, intercept, slope_r, slope_p, slope_se = stats.linregress(lignin[mask], y[mask])
        rows.append(
            {
                "Stratum": label,
                "n": int(mask.sum()),
                "Pearson_r": r,
                "Pearson_p": p,
                "Spearman_rho": rho,
                "Spearman_p": sp,
                "Slope_BMP_per_lignin": slope,
                "Slope_p": slope_p,
            }
        )

    low = dm <= median_dm
    high = dm > median_dm
    r_low = rows[0]["Pearson_r"]
    r_high = rows[1]["Pearson_r"]
    z_low = np.arctanh(r_low)
    z_high = np.arctanh(r_high)
    se_diff = math.sqrt(1 / (low.sum() - 3) + 1 / (high.sum() - 3))
    z_test = (z_low - z_high) / se_diff
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_test)))

    boot = []
    low_l = lignin[low].to_numpy()
    low_y = y[low].to_numpy()
    high_l = lignin[high].to_numpy()
    high_y = y[high].to_numpy()
    for _ in range(N_BOOT_CORR):
        li = rng.integers(0, len(low_l), len(low_l))
        hi = rng.integers(0, len(high_l), len(high_l))
        boot.append(stats.pearsonr(low_l[li], low_y[li])[0] - stats.pearsonr(high_l[hi], high_y[hi])[0])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    summary = pd.DataFrame(rows)
    summary["Fisher_z_for_low_minus_high"] = np.nan
    summary["Fisher_p_for_low_minus_high"] = np.nan
    summary["Bootstrap_delta_r_CI_low"] = np.nan
    summary["Bootstrap_delta_r_CI_high"] = np.nan
    summary.loc[0, "Fisher_z_for_low_minus_high"] = z_test
    summary.loc[0, "Fisher_p_for_low_minus_high"] = p_diff
    summary.loc[0, "Bootstrap_delta_r_CI_low"] = ci_lo
    summary.loc[0, "Bootstrap_delta_r_CI_high"] = ci_hi

    tertile_labels = pd.qcut(dm, 3, labels=["Low DM tertile", "Mid DM tertile", "High DM tertile"])
    tertile_rows = []
    for label in ["Low DM tertile", "Mid DM tertile", "High DM tertile"]:
        mask = tertile_labels == label
        r, p = stats.pearsonr(lignin[mask], y[mask])
        tertile_rows.append(
            {
                "Stratum": label,
                "n": int(mask.sum()),
                "DM_min": dm[mask].min(),
                "DM_max": dm[mask].max(),
                "Pearson_r": r,
                "Pearson_p": p,
            }
        )
    tertiles = pd.DataFrame(tertile_rows)

    return summary


def shap_analysis(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    pipe = rf_pipeline(RandomForestRegressor(**RF_PARAMS))
    pipe.fit(X, y)
    preprocessor = Pipeline([("imputer", pipe.named_steps["imputer"]), ("scaler", pipe.named_steps["scaler"])])
    x_model = pd.DataFrame(preprocessor.transform(X), columns=X.columns)
    x_display = X.rename(columns=SHORT)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer.shap_values(x_model, check_additivity=False)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"Feature": [clean_label(c) for c in X.columns], "Mean_abs_SHAP": mean_abs})
    importance = importance.sort_values("Mean_abs_SHAP", ascending=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(importance["Feature"][::-1], importance["Mean_abs_SHAP"][::-1], color="#4c78a8", edgecolor="0.4")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig2a_SHAP_importance.png", dpi=300)
    plt.close(fig)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, x_display, show=False, max_display=len(FEATURES))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig2b_SHAP_beeswarm.png", dpi=300)
    plt.close()

    lignin_idx = X.columns.get_loc("Lignin Mean (g/100g DM)")
    dm_idx = X.columns.get_loc("DM Mean (% FM)")
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        ind=lignin_idx,
        shap_values=shap_values,
        features=X.values,
        feature_names=[clean_label(c) for c in X.columns],
        interaction_index=dm_idx,
        show=False,
    )
    plt.xlabel("Lignin (g/100g DM)")
    plt.ylabel("SHAP value for Lignin")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig3_SHAP_lignin_DM_dependence.png", dpi=300)
    plt.close()

    return shap_values, importance


def bootstrap_shap(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    all_importance = np.zeros((N_BOOT, len(FEATURES)))
    rank_matrix = np.zeros((N_BOOT, len(FEATURES)), dtype=int)

    for i in range(N_BOOT):
        idx = rng.integers(0, len(X), len(X))
        x_boot = X.iloc[idx]
        y_boot = y.iloc[idx]
        pipe = rf_pipeline(RandomForestRegressor(**{**RF_PARAMS, "random_state": i}))
        pipe.fit(x_boot, y_boot)
        preprocessor = Pipeline([("imputer", pipe.named_steps["imputer"]), ("scaler", pipe.named_steps["scaler"])])
        x_model = pd.DataFrame(preprocessor.transform(x_boot), columns=X.columns)
        sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(x_model, check_additivity=False)
        imp = np.abs(sv).mean(axis=0)
        all_importance[i] = imp
        order = np.argsort(imp)[::-1]
        ranks = np.empty(len(FEATURES), dtype=int)
        ranks[order] = np.arange(1, len(FEATURES) + 1)
        rank_matrix[i] = ranks

    mean_imp = all_importance.mean(axis=0)
    ci_low = np.percentile(all_importance, 2.5, axis=0)
    ci_high = np.percentile(all_importance, 97.5, axis=0)
    mean_rank = rank_matrix.mean(axis=0)
    top1 = (rank_matrix == 1).sum(axis=0)
    top3 = (rank_matrix <= 3).sum(axis=0)

    table = pd.DataFrame(
        {
            "Feature": [clean_label(c) for c in FEATURES],
            "Mean_abs_SHAP": mean_imp,
            "CI_low": ci_low,
            "CI_high": ci_high,
            "Mean_rank": mean_rank,
            "Top1_count_of_100": top1,
            "Top3_count_of_100": top3,
        }
    ).sort_values("Mean_abs_SHAP", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    rev = table.iloc[::-1]
    xerr = np.vstack([rev["Mean_abs_SHAP"] - rev["CI_low"], rev["CI_high"] - rev["Mean_abs_SHAP"]])
    ax.barh(rev["Feature"], rev["Mean_abs_SHAP"], xerr=xerr, color="#4c78a8", edgecolor="0.4", capsize=3)
    ax.set_xlabel("Bootstrap mean absolute SHAP value (95% CI)")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FigS3_bootstrap_SHAP_stability.png", dpi=300)
    plt.close(fig)
    return table


def correlation_heatmap(X: pd.DataFrame) -> None:
    corr = X.rename(columns=SHORT).corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, annot=True, fmt=".2f", square=True, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FigS2_feature_correlation_heatmap.png", dpi=300)
    plt.close(fig)


def cv_predicted_vs_observed(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe = rf_pipeline(RandomForestRegressor(**RF_PARAMS))
    pred = cross_val_predict(pipe, X, y, cv=cv)
    out = pd.DataFrame({"Observed_BMP": y, "Predicted_BMP": pred})

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y, pred, color="#4c78a8", alpha=0.65, edgecolor="0.4")
    lo = min(y.min(), pred.min()) - 10
    hi = max(y.max(), pred.max()) + 10
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed BMP (Nm3 CH4/t DM)")
    ax.set_ylabel("Predicted BMP (Nm3 CH4/t DM)")
    ax.text(0.04, 0.96, f"R2 = {r2_score(y, pred):.3f}\nRMSE = {rmse(y, pred):.1f}", transform=ax.transAxes, va="top")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "FigS1_predicted_vs_observed.png", dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    ensure_dirs()
    df, X, y = load_data()
    print(f"Loaded {len(df)} samples after DM >= {DM_FILTER:g}% filter.")

    descriptive_tables(df, X, y)
    compare_models(X, y)
    ols_and_interaction_tests(X, y)
    stratified_validation(X, y)
    shap_analysis(X, y)
    bootstrap_shap(X, y)
    correlation_heatmap(X)
    cv_predicted_vs_observed(X, y)

    print("Workflow complete.")
    print(f"Tables: {TABLE_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
