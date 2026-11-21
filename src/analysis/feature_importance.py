"""
src/analysis/feature_importance.py
====================================
Extracts and visualises feature importances from trained RandomForest and GBT
models produced by the ablation step.

How it works
------------
Spark ML's RandomForestRegressionModel and GBTRegressionModel both expose a
``featureImportances`` SparseVector.  The vector index corresponds to the
position of a feature in the VectorAssembler's input list.  We reconstruct
that ordered list by inspecting the saved pipeline metadata, then zip indices
to names.

Public API
----------
  load_feature_importances(model_path, feature_cols) -> pd.DataFrame
  plot_importances(df, title, out_path) -> None
  run_feature_importance_analysis(paths, tasks, conditions, algorithms) -> None
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler as VectorAssemblerModel
from pyspark.ml.regression import GBTRegressionModel, RandomForestRegressionModel

from config.settings import (
    BTS_FEATURE_COLS,
    FULL_FEATURE_COLS,
    PATHS,
    WEATHER_FEATURE_COLS,
)
from src.models.pipeline import CATEGORICAL_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Which algorithms expose featureImportances
_SUPPORTED_ALGOS = {"RandomForest", "GBTRegressor"}

# Map condition name → feature column list
_CONDITION_COLS = {
    "bts_only":    BTS_FEATURE_COLS,
    "bts_weather": WEATHER_FEATURE_COLS,
    "full":        FULL_FEATURE_COLS,
}


# ---------------------------------------------------------------------------
# Feature-name reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_feature_names(feature_cols: list) -> list:
    """
    Reproduce the exact ordered list of feature names that VectorAssembler
    sees after StringIndexer + OneHotEncoder + Imputer.

    For categorical columns the OHE produces k-1 dummy columns named
    ``{col}_oh_0``, ``{col}_oh_1``, …  The number of dummies cannot be known
    without running the pipeline, so we use a placeholder ``{col}_oh[*]``
    for the group and later annotate the plot with the group name.

    For numeric columns the name is ``{col}_imp`` (if imputed) or ``{col}``.
    We conservatively assume every numeric column is imputed here, which
    matches the pipeline's default behaviour.
    """
    names: list = []
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS]
    num_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    # OHE columns: we add a single placeholder per categorical group
    for c in cat_cols:
        names.append(f"{c} (OHE)")

    # Numeric columns
    for c in num_cols:
        names.append(f"{c}_imp")

    return names


def _extract_importances_from_pipeline(
    pipeline_model: PipelineModel,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Pull featureImportances out of the last stage of *pipeline_model* and
    return a tidy DataFrame with columns [feature, importance].

    Feature names are recovered from the VectorAssembler's ``inputCols``
    metadata, which is authoritative — it lists exactly the columns that
    were assembled into the feature vector, in the same order as the
    importance indices.
    """
    last_stage = pipeline_model.stages[-1]

    if not isinstance(last_stage, (RandomForestRegressionModel, GBTRegressionModel)):
        raise TypeError(
            f"Last stage is {type(last_stage).__name__}, not RF or GBT — "
            "featureImportances not available."
        )

    importances = last_stage.featureImportances.toArray()

    # Find the VectorAssembler in the pipeline (search backwards — it's
    # usually the second-to-last stage, but may shift if stages change).
    assembler_inputs = None
    for stage in reversed(pipeline_model.stages):
        if isinstance(stage, VectorAssemblerModel) or hasattr(stage, "getInputCols"):
            try:
                assembler_inputs = stage.getInputCols()
                break
            except Exception:
                continue

    if assembler_inputs is None:
        assembler_inputs = _reconstruct_feature_names(feature_cols)
        logger.warning(
            "Could not extract inputCols from VectorAssembler; "
            "falling back to reconstructed names."
        )

    # Pad / truncate names to match the vector length
    n = len(importances)
    names = list(assembler_inputs)[:n]
    if len(names) < n:
        names += [f"feature_{i}" for i in range(len(names), n)]

    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_importances(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    top_n: int = 20,
) -> None:
    """
    Horizontal bar chart of the top *top_n* features by importance.
    Saves to *out_path* at 150 dpi.
    """
    df_top = df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(df_top["feature"][::-1], df_top["importance"][::-1], color="#4c72b0")
    ax.set_xlabel("Feature Importance (Gini / gain-based)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance plot saved → %s", out_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_feature_importance_analysis(
    paths: dict = PATHS,
    tasks: list = None,
    conditions: list = None,
    algorithms: list = None,
) -> None:
    """
    Load every saved RF / GBT model, extract feature importances, save a CSV
    and a bar-chart PNG per model into ``results/``.

    Parameters
    ----------
    paths      : project paths dict (from config.settings.PATHS)
    tasks      : list of task names, e.g. ['departure', 'arrival']
    conditions : list of condition names, e.g. ['bts_only', 'bts_weather']
    algorithms : list of algorithm names to analyse (must be RF or GBT)
    """
    tasks      = tasks      or ["departure", "arrival"]
    conditions = conditions or ["bts_only", "bts_weather", "full"]
    algorithms = algorithms or ["RandomForest", "GBTRegressor"]

    all_rows: list = []

    for task in tasks:
        for condition in conditions:
            feature_cols = _CONDITION_COLS[condition]

            for algo in algorithms:
                if algo not in _SUPPORTED_ALGOS:
                    logger.warning("Skipping %s — featureImportances not supported.", algo)
                    continue

                model_dir = paths["models"] / f"{task}_{condition}_{algo}"
                if not model_dir.exists():
                    logger.warning("Model not found at %s — skipping.", model_dir)
                    continue

                logger.info("Extracting importances: %s | %s | %s", task, condition, algo)

                try:
                    pipeline_model = PipelineModel.load(str(model_dir))
                    imp_df = _extract_importances_from_pipeline(pipeline_model, feature_cols)
                except Exception as exc:
                    logger.error("Failed for %s/%s/%s: %s", task, condition, algo, exc)
                    continue

                # Tag and accumulate
                imp_df.insert(0, "algorithm", algo)
                imp_df.insert(0, "condition", condition)
                imp_df.insert(0, "task", task)
                all_rows.append(imp_df)

                # Per-model plot
                title = f"Feature Importances — {task} | {condition} | {algo}"
                out_png = paths["results"] / f"feat_importance_{task}_{condition}_{algo}.png"
                plot_importances(imp_df, title, out_png)

    if not all_rows:
        logger.warning("No models found — run the ablation step first.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    out_csv = paths["results"] / "feature_importances.csv"
    combined.to_csv(out_csv, index=False)
    logger.info("All feature importances saved → %s", out_csv)
