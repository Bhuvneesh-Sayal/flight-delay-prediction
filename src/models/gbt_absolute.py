"""
GBT trained with absolute loss (lossType="absolute").

The ablation trains every regressor with squared error, which makes it predict
the conditional mean; the reported metric is MAE, which is minimised by the
conditional median. Training with absolute loss aligns the two.

Protocol is identical to ablation.py: each grid setting is fit on train
(Jan-Aug) and scored by MAE on validation (Sep-Oct); the best setting is refit
on Jan-Oct and scored once on test (Nov-Dec). Test MAE gets a 95% day-block
bootstrap interval for the difference to the train-median baseline.

  python -m src.models.gbt_absolute            (repo root, local Spark)
  spark-submit --master spark://... src/models/gbt_absolute.py   (cluster)
Writes results/gbt_absolute_loss_results.csv and models/<task>_full_GBTAbsolute.
Data and model folders come from config.settings.PATHS; FDP_DATA / FDP_MODELS
override them (e.g. to read the cluster volumes from a local run).
"""
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyspark.ml.regression import GBTRegressor  # noqa: E402
from pyspark.sql import SparkSession, functions as F  # noqa: E402

from config.settings import FULL_FEATURE_COLS, PATHS  # noqa: E402
from src.models.pipeline import build_pipeline  # noqa: E402

FEATURES = Path(os.environ["FDP_DATA"]) / "features" if "FDP_DATA" in os.environ else PATHS["features"]
MODELS = Path(os.environ.get("FDP_MODELS", PATHS["models"]))
OUT = PATHS["results"] / "gbt_absolute_loss_results.csv"
GRID = {"maxIter": [50, 100], "maxDepth": [4, 6], "stepSize": [0.1, 0.3]}
N_BOOT, SEED = 1000, 42


def mae(pdf, col):
    return float((pdf["y"] - pdf[col]).abs().mean())


def day_block_ci(pdf, a, b):
    """95% CI of MAE(a) - MAE(b), resampling whole test days."""
    days = pdf["FlightDate"].astype(str).values
    uniq = np.unique(days)
    err_a = (pdf["y"] - pdf[a]).abs().groupby(days).agg(["sum", "count"])
    err_b = (pdf["y"] - pdf[b]).abs().groupby(days).agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sa, sb = err_a.loc[pick], err_b.loc[pick]
        diffs.append(sa["sum"].sum() / sa["count"].sum() - sb["sum"].sum() / sb["count"].sum())
    return np.percentile(diffs, [2.5, 97.5])


def main():
    builder = SparkSession.builder.appName("gbt-absolute").config("spark.sql.shuffle.partitions", "32")
    if "PYSPARK_GATEWAY_PORT" not in os.environ:
        # plain `python -m ...` -> local mode; spark-submit sets the gateway and --master
        builder = builder.master("local[*]").config("spark.driver.memory", "8g")
    spark = builder.getOrCreate()
    print("master:", spark.sparkContext.master, "| features:", FEATURES, flush=True)
    spark.sparkContext.setLogLevel("ERROR")
    rows = []
    for task, tgt in [("departure", "DepDelay"), ("arrival", "ArrDelay")]:
        ds = spark.read.parquet(str(FEATURES / f"{task}_full"))
        train = ds.filter("data_split = 'train'").cache()
        val = ds.filter("data_split = 'validation'").cache()
        test = ds.filter("data_split = 'test'").cache()
        median = train.approxQuantile(tgt, [0.5], 0.0)[0]

        est = (GBTRegressor(lossType="absolute", maxBins=64, seed=SEED)
               .setLabelCol(tgt).setFeaturesCol("features"))
        pipe = build_pipeline(FULL_FEATURE_COLS, est, df=train)
        best = None
        for it, dp, ss in itertools.product(GRID["maxIter"], GRID["maxDepth"], GRID["stepSize"]):
            params = {est.maxIter: it, est.maxDepth: dp, est.stepSize: ss}
            p = pipe.fit(train, params).transform(val)
            v = p.select(F.avg(F.abs(F.col(tgt) - F.col("prediction")))).first()[0]
            print(f"{task} maxIter={it} maxDepth={dp} stepSize={ss} val MAE={v:.3f}", flush=True)
            if best is None or v < best[0]:
                best = (v, it, dp, ss, params)
        v, it, dp, ss, params = best
        model = pipe.fit(train.union(val), params)
        model.write().overwrite().save(str(MODELS / f"{task}_full_GBTAbsolute"))

        pdf = (model.transform(test)
               .select("FlightDate", F.col(tgt).alias("y"), "prediction").toPandas())
        pdf["median"] = median
        m, mb = mae(pdf, "prediction"), mae(pdf, "median")
        lo, hi = day_block_ci(pdf, "prediction", "median")
        r2 = 1 - ((pdf.y - pdf.prediction) ** 2).sum() / ((pdf.y - pdf.y.mean()) ** 2).sum()
        rmse = float(np.sqrt(((pdf.y - pdf.prediction) ** 2).mean()))
        rows.append(dict(Task=task, Condition="full", Algorithm="GBT (absolute loss)",
                         maxIter=it, maxDepth=dp, stepSize=ss, Val_MAE_min=round(v, 2),
                         Test_MAE_min=round(m, 2), Test_RMSE_min=round(rmse, 2), Test_R2=round(r2, 4),
                         Median_baseline_MAE=round(mb, 2), Diff_vs_median=round(m - mb, 2),
                         Diff_CI_low=round(lo, 2), Diff_CI_high=round(hi, 2)))
        print(rows[-1], flush=True)
        for d in (train, val, test):
            d.unpersist()
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("saved", OUT)
    spark.stop()


if __name__ == "__main__":
    main()
