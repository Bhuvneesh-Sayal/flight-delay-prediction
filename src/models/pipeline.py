"""
src/models/pipeline.py
=======================
Reusable Spark ML pipeline builder.

The pipeline encodes categorical columns (StringIndexer → OneHotEncoder),
imputes missing numerics (median strategy) and assembles a feature vector —
all before the supplied estimator stage.

When a reference DataFrame is supplied, only numeric columns that actually
contain nulls/NaNs are imputed.  Columns without missing values bypass the
Imputer and go straight to the VectorAssembler (no ``_imp`` suffix).

Public API
----------
  build_pipeline(predictors, estimator, df=None) -> Pipeline
  get_null_counts(df, cols) -> dict[str, int]
  CATEGORICAL_COLS  : list of column names treated as categorical
"""

from typing import Dict, List, Optional

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns that should always be treated as categorical regardless of their dtype
CATEGORICAL_COLS: list = ["Origin", "Dest", "Reporting_Airline"]

_FLOAT_TYPES = (T.FloatType, T.DoubleType)


def get_null_counts(df: DataFrame, cols: List[str]) -> Dict[str, int]:
    """
    Count missing values (null, plus NaN for float/double columns) per column.

    Parameters
    ----------
    df   : DataFrame to inspect.
    cols : Column names to check.

    Returns
    -------
    dict mapping each column name to its missing-value count.
    The count is computed in a single Spark pass.
    """
    if not cols:
        return {}

    dtypes = {f.name: f.dataType for f in df.schema.fields}
    exprs = []
    for c in cols:
        missing = F.col(c).isNull()
        if isinstance(dtypes[c], _FLOAT_TYPES):
            missing = missing | F.isnan(F.col(c))
        exprs.append(F.sum(F.when(missing, 1).otherwise(0)).alias(c))

    row = df.agg(*exprs).first()
    return {c: int(row[c] or 0) for c in cols}


def build_pipeline(
    predictors: list,
    estimator,
    df: Optional[DataFrame] = None,
) -> Pipeline:
    """
    Build a Spark ML Pipeline for the given *predictors* and *estimator*.

    Parameters
    ----------
    predictors : list[str]
        Column names used as model inputs.  Categorical columns are
        identified by membership in CATEGORICAL_COLS.
    estimator :
        Any Spark ML estimator that accepts 'features' as its input column
        (e.g. LinearRegression, RandomForestRegressor, LogisticRegression).
    df : DataFrame, optional
        Reference data (normally the training set) used to find which numeric
        columns contain nulls.  Only those are imputed; the rest are passed
        directly to the VectorAssembler.  If omitted, every numeric column
        is imputed (previous behaviour).

    Returns
    -------
    pyspark.ml.Pipeline
    """
    cat_cols = [c for c in predictors if c in CATEGORICAL_COLS]
    num_cols = [c for c in predictors if c not in CATEGORICAL_COLS]

    if df is None:
        impute_cols = list(num_cols)
    else:
        null_counts = get_null_counts(df, num_cols)
        impute_cols = [c for c in num_cols if null_counts[c] > 0]
        logger.info(
            "Imputing %d/%d numeric columns with nulls: %s",
            len(impute_cols), len(num_cols), impute_cols,
        )
    passthrough_cols = [c for c in num_cols if c not in impute_cols]

    stages: list = []

    # 1. Index each categorical column → integer index
    stages += [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat_cols
    ]

    # 2. One-hot encode all indexed columns in a single pass
    stages.append(
        OneHotEncoder(
            inputCols=[f"{c}_idx" for c in cat_cols],
            outputCols=[f"{c}_oh" for c in cat_cols],
            handleInvalid="keep",
        )
    )

    # 3. Impute only the numeric columns that have missing values.
    #    Without a reference DataFrame the Imputer is always emitted.
    if df is None or impute_cols:
        stages.append(
            Imputer(
                strategy="median",
                inputCols=impute_cols,
                outputCols=[f"{c}_imp" for c in impute_cols],
            )
        )

    # 4. Assemble encoded / imputed / passthrough columns into 'features'
    assembler_inputs = (
        [f"{c}_oh" for c in cat_cols]
        + [f"{c}_imp" for c in impute_cols]
        + passthrough_cols
    )
    stages.append(
        VectorAssembler(
            inputCols=assembler_inputs,
            outputCol="features",
            handleInvalid="keep",
        )
    )

    stages.append(estimator)
    return Pipeline(stages=stages)
