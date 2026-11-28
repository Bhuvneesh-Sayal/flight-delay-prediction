"""
Build the summary charts for REPORT.md from the pipeline's result CSVs.

Run inside the Spark image (it has pandas + matplotlib), from the repo root:
  docker run --rm -v "$PWD":/app -w /app spark-local:4.0.0-zscaler \
      python3 reports/spark4_cluster_run_2026-11-21/make_charts.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CHARTS = HERE / "charts"
CHARTS.mkdir(exist_ok=True)

ALGOS = ["MeanBaseline", "LinearRegression", "RandomForest", "GBTRegressor"]
CONDS = ["bts_only", "bts_weather", "full"]
COLORS = {"MeanBaseline": "#9e9e9e", "LinearRegression": "#4c72b0",
          "RandomForest": "#55a868", "GBTRegressor": "#c44e52"}

abl = pd.read_csv(DATA / "controlled_ablation_summary.csv")
hur = pd.read_csv(DATA / "two_stage_hurdle_results.csv")
imp = pd.read_csv(DATA / "feature_importances.csv")
abl_351 = pd.read_csv(DATA / "controlled_ablation_summary_spark351.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(CHARTS / name, dpi=140)
    plt.close(fig)
    print("saved", name)


def grouped_bars(metric, ylabel, name, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, task in zip(axes, ["departure", "arrival"]):
        sub = abl[abl["Task"] == task]
        width = 0.2
        for i, algo in enumerate(ALGOS):
            vals = [sub[(sub["Condition"] == c) & (sub["Algorithm"] == algo)][metric].iloc[0]
                    for c in CONDS]
            xs = [j + (i - 1.5) * width for j in range(len(CONDS))]
            bars = ax.bar(xs, vals, width, label=algo, color=COLORS[algo])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels(CONDS)
        ax.set_title(f"{task.capitalize()} delay")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if metric == "Test_R2":
            ax.axhline(0, color="black", linewidth=0.8)
        else:
            lo = sub[metric].min()
            ax.set_ylim(lo * 0.85, sub[metric].max() * 1.04)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(title, fontweight="bold")
    save(fig, name)


# 1-2. Test MAE and R² per model / feature set
grouped_bars("Test_MAE_min", "Test MAE (minutes, lower is better)",
             "01_test_mae_by_model.png", "Test MAE by model and feature set (Nov–Dec 2025)")
grouped_bars("Test_R2", "Test R² (higher is better)",
             "02_test_r2_by_model.png", "Test R² by model and feature set")

# 3. Validation vs test MAE (Sep–Oct vs Nov–Dec)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, task in zip(axes, ["departure", "arrival"]):
    sub = abl[(abl["Task"] == task) & (abl["Algorithm"] != "MeanBaseline")]
    labels = [f"{r.Condition}\n{r.Algorithm.replace('Regressor', '').replace('Regression', '')}"
              for r in sub.itertuples()]
    xs = range(len(sub))
    ax.bar([x - 0.2 for x in xs], sub["Val_MAE_min"], 0.4, label="Validation (Sep–Oct)", color="#8da0cb")
    ax.bar([x + 0.2 for x in xs], sub["Test_MAE_min"], 0.4, label="Test (Nov–Dec)", color="#fc8d62")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
    ax.set_title(f"{task.capitalize()} delay")
    ax.set_ylabel("MAE (minutes)")
    ax.set_ylim(sub[["Val_MAE_min", "Test_MAE_min"]].min().min() * 0.85, None)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
axes[0].legend(fontsize=8)
fig.suptitle("Validation vs test MAE: models that win on validation do not always win on test",
             fontweight="bold")
save(fig, "03_val_vs_test_mae.png")

# 4. Hurdle model vs best single-stage model
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
lbl = [f"{r.Task}\n{r.Condition}" for r in hur.itertuples()]
ax.bar(lbl, hur["Stage1_ROC_AUC"], color="#4c72b0")
ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="random guess (0.5)")
for i, v in enumerate(hur["Stage1_ROC_AUC"]):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0.45, 0.72)
ax.set_title("Stage 1: chance of a delay (ROC-AUC)")
ax.legend(fontsize=8)
ax = axes[1]
best = []
for r in hur.itertuples():
    s = abl[(abl["Task"] == r.Task) & (abl["Condition"] == r.Condition)
            & (abl["Algorithm"] != "MeanBaseline")]["Test_MAE_min"].min()
    best.append(s)
xs = range(len(hur))
ax.bar([x - 0.2 for x in xs], hur["TwoStage_MAE_min"], 0.4, label="Two-stage hurdle", color="#c44e52")
ax.bar([x + 0.2 for x in xs], best, 0.4, label="Best single model", color="#55a868")
ax.set_xticks(list(xs))
ax.set_xticklabels(lbl)
ax.set_ylim(min(best + list(hur["TwoStage_MAE_min"])) * 0.9, None)
ax.set_ylabel("Test MAE (minutes)")
ax.set_title("Combined prediction vs best single model")
ax.legend(fontsize=8)
fig.suptitle("Two-stage hurdle model", fontweight="bold")
save(fig, "04_hurdle_model.png")

# 5. Top features (GBT, full feature set)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, task in zip(axes, ["departure", "arrival"]):
    sub = imp[(imp["task"] == task) & (imp["condition"] == "full")
              & (imp["algorithm"] == "GBTRegressor")].nlargest(15, "importance")[::-1]
    ax.barh(sub["feature"], sub["importance"], color="#c44e52")
    ax.set_title(f"{task.capitalize()} delay — GBT, full features")
    ax.set_xlabel("Importance (share of total gain)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
fig.suptitle("Top 15 features", fontweight="bold")
save(fig, "05_top_features_gbt_full.png")

# 6. Feature importance grouped by feature family
FAMILY = {
    "Schedule": ["scheduled_month", "scheduled_day_of_week", "scheduled_departure_hour",
                 "CRSElapsedTime"],
    "Route / airline": ["Origin=", "Dest=", "Reporting_Airline="],
    "Congestion / prior leg": ["origin_congestion_2h", "prior_leg_delay"],
    "Weather": ["w_", "weather_"],
    "Same-day airport delays": ["airport_"],
    "Storms / holidays": ["storm_", "is_holiday"],
}


def family(f):
    for fam, keys in FAMILY.items():
        if any(f == k or f.startswith(k) for k in keys):
            return fam
    return "Other"


imp["family"] = imp["feature"].map(family)
fam = (imp[(imp["condition"] == "full")]
       .groupby(["task", "algorithm", "family"])["importance"].sum().reset_index())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
order = list(FAMILY.keys())
for ax, task in zip(axes, ["departure", "arrival"]):
    for i, algo in enumerate(["RandomForest", "GBTRegressor"]):
        sub = fam[(fam["task"] == task) & (fam["algorithm"] == algo)].set_index("family")
        vals = [sub["importance"].get(f, 0.0) for f in order]
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
save(fig, "06_importance_by_feature_family.png")

# 7. Spark 3.5.1 (single container) vs Spark 4.0.0 (cluster) test MAE
m = abl.merge(abl_351, on=["Task", "Condition", "Algorithm"], suffixes=("_400", "_351"))
m = m[m["Algorithm"] != "MeanBaseline"]
fig, ax = plt.subplots(figsize=(6, 6))
for algo in ["LinearRegression", "RandomForest", "GBTRegressor"]:
    s = m[m["Algorithm"] == algo]
    ax.scatter(s["Test_MAE_min_351"], s["Test_MAE_min_400"], label=algo, color=COLORS[algo], s=45)
lo, hi = m[["Test_MAE_min_351", "Test_MAE_min_400"]].min().min() - 0.5, \
    m[["Test_MAE_min_351", "Test_MAE_min_400"]].max().max() + 0.5
ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="same result")
ax.set_xlabel("Test MAE, Spark 3.5.1 run (minutes)")
ax.set_ylabel("Test MAE, Spark 4.0.0 cluster run (minutes)")
ax.set_title("Spark 3.5.1 vs 4.0.0: same models, same data", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(linestyle="--", alpha=0.4)
save(fig, "07_spark351_vs_spark400.png")

# 8. Step run times on the cluster
steps = pd.DataFrame({
    "step": ["ingest + prep", "features", "ablation (18 models)", "hurdle", "analysis"],
    "minutes": [28 / 60, 4.5, 19.35, 2.8, 1.12],
})
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh(steps["step"][::-1], steps["minutes"][::-1], color="#4c72b0")
for y, v in enumerate(steps["minutes"][::-1]):
    ax.text(v, y, f" {v:.1f} min", va="center", fontsize=8)
ax.set_xlabel("Minutes")
ax.set_title("Run time per step — Spark 4.0.0, 3 workers × 2 cores, 1 GB executors",
             fontweight="bold", fontsize=10)
ax.grid(axis="x", linestyle="--", alpha=0.4)
save(fig, "08_step_runtimes.png")
