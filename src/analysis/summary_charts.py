"""
src/analysis/summary_charts.py
===============================
Cross-model summary charts built from the result CSVs that the ablation,
hurdle and feature-importance steps write to ``results/``.

The per-model plots (residuals, Q–Q, predicted vs actual, importances) live
in residuals.py and feature_importance.py. These charts compare models,
feature sets and tasks side by side instead.

Charts written to ``results/summary_charts/``
----------------------------------------------
  01_test_mae_by_model.png           test MAE per model and feature set
  02_test_r2_by_model.png            test R² per model and feature set
  03_val_vs_test_mae.png             validation vs test MAE
  04_hurdle_model.png                hurdle stage-1 AUC; hurdle vs best single model
  05_top_features_gbt_full.png       top 15 features, GBT on the full feature set
  06_importance_by_feature_family.png importance summed by feature family

A chart is skipped (with a warning) when its input CSV is missing, so this
can run after any subset of the modelling steps.

Public API
----------
  run_summary_charts(paths) -> list[Path]
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config.settings import PATHS
from src.utils.logger import get_logger

logger = get_logger(__name__)

ALGOS = ["MeanBaseline", "LinearRegression", "RandomForest", "GBTRegressor"]
CONDS = ["bts_only", "bts_weather", "full"]
TASKS = ["departure", "arrival"]
COLORS = {"MeanBaseline": "#9e9e9e", "LinearRegression": "#4c72b0",
          "RandomForest": "#55a868", "GBTRegressor": "#c44e52"}

# Feature name prefixes per family (one-hot features are named "Origin=ATL" etc.)
FEATURE_FAMILIES = {
    "Schedule": ["scheduled_month", "scheduled_day_of_week", "scheduled_departure_hour",
                 "CRSElapsedTime"],
    "Route / airline": ["Origin=", "Dest=", "Reporting_Airline="],
    "Congestion / prior leg": ["origin_congestion_2h", "prior_leg_delay"],
    "Weather": ["w_", "weather_"],
    "Same-day airport delays": ["airport_"],
    "Storms / holidays": ["storm_", "is_holiday"],
}


def _family(feature: str) -> str:
    for fam, keys in FEATURE_FAMILIES.items():
        if any(feature == k or feature.startswith(k) for k in keys):
            return fam
    return "Other"


def _save(fig, out_dir: Path, name: str, written: list) -> None:
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    logger.info("Summary chart saved → %s", path)


def _metric(sub: pd.DataFrame, cond: str, algo: str, metric: str) -> float:
    row = sub[(sub["Condition"] == cond) & (sub["Algorithm"] == algo)]
    return float(row[metric].iloc[0]) if len(row) else float("nan")


def _grouped_bars(abl, metric, ylabel, title, out_dir, name, written):
    conds = [c for c in CONDS if c in set(abl["Condition"])]
    algos = [a for a in ALGOS if a in set(abl["Algorithm"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.8 / max(len(algos), 1)
    for ax, task in zip(axes, TASKS):
        sub = abl[abl["Task"] == task]
        for i, algo in enumerate(algos):
            vals = [_metric(sub, c, algo, metric) for c in conds]
            xs = [j + (i - (len(algos) - 1) / 2) * width for j in range(len(conds))]
            bars = ax.bar(xs, vals, width, label=algo, color=COLORS.get(algo))
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds)
        ax.set_title(f"{task.capitalize()} delay")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if metric.endswith("R2"):
            ax.axhline(0, color="black", linewidth=0.8)
        elif len(sub):
            ax.set_ylim(sub[metric].min() * 0.85, sub[metric].max() * 1.04)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(title, fontweight="bold")
    _save(fig, out_dir, name, written)


def _val_vs_test(abl, out_dir, written):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, task in zip(axes, TASKS):
        sub = abl[(abl["Task"] == task) & (abl["Algorithm"] != "MeanBaseline")]
        labels = [f"{r.Condition}\n{r.Algorithm.replace('Regressor', '').replace('Regression', '')}"
                  for r in sub.itertuples()]
        xs = list(range(len(sub)))
        ax.bar([x - 0.2 for x in xs], sub["Val_MAE_min"], 0.4,
               label="Validation", color="#8da0cb")
        ax.bar([x + 0.2 for x in xs], sub["Test_MAE_min"], 0.4,
               label="Test", color="#fc8d62")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
        ax.set_title(f"{task.capitalize()} delay")
        ax.set_ylabel("MAE (minutes)")
        if len(sub):
            ax.set_ylim(sub[["Val_MAE_min", "Test_MAE_min"]].min().min() * 0.85, None)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)
    fig.suptitle("Validation vs test MAE", fontweight="bold")
    _save(fig, out_dir, "03_val_vs_test_mae.png", written)


def _hurdle(hur, abl, out_dir, written):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [f"{r.Task}\n{r.Condition}" for r in hur.itertuples()]

    ax = axes[0]
    ax.bar(labels, hur["Stage1_ROC_AUC"], color="#4c72b0")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="random guess (0.5)")
    for i, v in enumerate(hur["Stage1_ROC_AUC"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.45, max(0.72, hur["Stage1_ROC_AUC"].max() + 0.03))
    ax.set_title("Stage 1: chance of a delay (ROC-AUC)")
    ax.legend(fontsize=8)

    ax = axes[1]
    xs = list(range(len(hur)))
    ax.bar([x - 0.2 for x in xs], hur["TwoStage_MAE_min"], 0.4,
           label="Two-stage hurdle", color="#c44e52")
    if abl is not None:
        best = []
        for r in hur.itertuples():
            s = abl[(abl["Task"] == r.Task) & (abl["Condition"] == r.Condition)
                    & (abl["Algorithm"] != "MeanBaseline")]["Test_MAE_min"]
            best.append(float(s.min()) if len(s) else float("nan"))
        ax.bar([x + 0.2 for x in xs], best, 0.4, label="Best single model", color="#55a868")
        ax.set_ylim(min(best + list(hur["TwoStage_MAE_min"])) * 0.9, None)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test MAE (minutes)")
    ax.set_title("Combined prediction vs best single model")
    ax.legend(fontsize=8)
    fig.suptitle("Two-stage hurdle model", fontweight="bold")
    _save(fig, out_dir, "04_hurdle_model.png", written)


def _top_features(imp, out_dir, written):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, task in zip(axes, TASKS):
        sub = imp[(imp["task"] == task) & (imp["condition"] == "full")
                  & (imp["algorithm"] == "GBTRegressor")].nlargest(15, "importance")[::-1]
        ax.barh(sub["feature"], sub["importance"], color="#c44e52")
        ax.set_title(f"{task.capitalize()} delay — GBT, full features")
        ax.set_xlabel("Importance (share of total gain)")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.suptitle("Top 15 features", fontweight="bold")
    _save(fig, out_dir, "05_top_features_gbt_full.png", written)


def _importance_by_family(imp, out_dir, written):
    imp = imp.assign(family=imp["feature"].map(_family))
    fam = (imp[imp["condition"] == "full"]
           .groupby(["task", "algorithm", "family"])["importance"].sum().reset_index())
    order = list(FEATURE_FAMILIES.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, task in zip(axes, TASKS):
        for i, algo in enumerate(["RandomForest", "GBTRegressor"]):
            sub = fam[(fam["task"] == task) & (fam["algorithm"] == algo)].set_index("family")
            vals = [float(sub["importance"].get(f, 0.0)) for f in order]
            ax.barh([y + (i - 0.5) * 0.38 for y in range(len(order))], vals, 0.38,
                    label=algo, color=COLORS[algo])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.set_title(f"{task.capitalize()} delay — full features")
        ax.set_xlabel("Summed importance")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)
    fig.suptitle("Which kinds of features the tree models use", fontweight="bold")
    _save(fig, out_dir, "06_importance_by_feature_family.png", written)


def _read(path: Path):
    if path.exists():
        return pd.read_csv(path)
    logger.warning("Summary charts: %s not found — skipping the charts that need it.", path)
    return None


def run_summary_charts(paths: dict = PATHS) -> list:
    """
    Build the cross-model summary charts from the CSVs in ``paths['results']``
    and save them to ``paths['results'] / 'summary_charts'``.

    Returns the list of chart paths written.
    """
    results = Path(paths["results"])
    out_dir = results / "summary_charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    abl = _read(results / "controlled_ablation_summary.csv")
    hur = _read(results / "two_stage_hurdle_results.csv")
    imp = _read(results / "feature_importances.csv")

    written: list = []
    if abl is not None:
        _grouped_bars(abl, "Test_MAE_min", "Test MAE (minutes, lower is better)",
                      "Test MAE by model and feature set", out_dir,
                      "01_test_mae_by_model.png", written)
        _grouped_bars(abl, "Test_R2", "Test R² (higher is better)",
                      "Test R² by model and feature set", out_dir,
                      "02_test_r2_by_model.png", written)
        _val_vs_test(abl, out_dir, written)
    if hur is not None:
        _hurdle(hur, abl, out_dir, written)
    if imp is not None:
        _top_features(imp, out_dir, written)
        _importance_by_family(imp, out_dir, written)

    logger.info("Summary charts complete: %d charts in %s", len(written), out_dir)
    return written
