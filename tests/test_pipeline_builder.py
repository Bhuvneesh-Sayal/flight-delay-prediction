"""Unit tests for src/models/pipeline.py.

Constructing pyspark.ml stages needs a live SparkContext (they wrap JVM
objects), so a minimal local session is created once per test module.
"""

from unittest.mock import MagicMock

import pytest
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession

from src.models.pipeline import CATEGORICAL_COLS, build_pipeline


@pytest.fixture(scope="module", autouse=True)
def spark():
    session = (
        SparkSession.builder.appName("test_pipeline_builder")
        .master("local[1]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()


def _stages_of(pipeline, cls):
    return [s for s in pipeline.getStages() if isinstance(s, cls)]


def test_build_pipeline_has_correct_stages():
    estimator = MagicMock()
    pipeline = build_pipeline(
        ["Origin", "Dest", "CRSElapsedTime", "DepDelay"], estimator
    )
    stages = pipeline.getStages()

    assert len(_stages_of(pipeline, StringIndexer)) == 2
    assert len(_stages_of(pipeline, OneHotEncoder)) == 1
    assert len(_stages_of(pipeline, Imputer)) == 1
    assert len(_stages_of(pipeline, VectorAssembler)) == 1
    assert stages[-1] is estimator
    assert len(stages) == 6

    imputer = _stages_of(pipeline, Imputer)[0]
    assert imputer.getInputCols() == ["CRSElapsedTime", "DepDelay"]
    assert imputer.getOutputCols() == ["CRSElapsedTime_imp", "DepDelay_imp"]

    assembler = _stages_of(pipeline, VectorAssembler)[0]
    assert assembler.getInputCols() == [
        "Origin_oh",
        "Dest_oh",
        "CRSElapsedTime_imp",
        "DepDelay_imp",
    ]


def test_build_pipeline_without_numeric_columns():
    """With only categorical predictors the Imputer stage is still emitted,
    but it has no input columns to process."""
    estimator = MagicMock()
    pipeline = build_pipeline(["Origin", "Dest"], estimator)

    assert len(_stages_of(pipeline, StringIndexer)) == 2
    imputers = _stages_of(pipeline, Imputer)
    assert len(imputers) == 1
    assert imputers[0].getInputCols() == []
    assert pipeline.getStages()[-1] is estimator


def test_build_pipeline_without_categorical_columns():
    """Numeric-only predictors produce no StringIndexers."""
    estimator = MagicMock()
    pipeline = build_pipeline(["CRSElapsedTime", "DepDelay"], estimator)

    assert _stages_of(pipeline, StringIndexer) == []
    assert len(_stages_of(pipeline, Imputer)) == 1
    assert len(_stages_of(pipeline, VectorAssembler)) == 1
    assert pipeline.getStages()[-1] is estimator


def test_categorical_cols_are_identified():
    for col in ("Origin", "Dest", "Reporting_Airline"):
        assert col in CATEGORICAL_COLS


def test_non_categorical_cols_go_to_imputer():
    assert "CRSElapsedTime" not in CATEGORICAL_COLS

    pipeline = build_pipeline(
        ["Origin", "CRSElapsedTime"], MagicMock()
    )
    imputer = _stages_of(pipeline, Imputer)[0]
    assert "CRSElapsedTime" in imputer.getInputCols()
    assert "Origin" not in imputer.getInputCols()
