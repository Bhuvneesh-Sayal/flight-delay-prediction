"""Unit tests for the NOAA weather logic in src/data/preprocessing.py.

These call the real `standardise_weather` function on small PSV files written
to tmp_path. It uses only pandas/pyarrow, so no Spark session is needed.
"""

import re

import pandas as pd
import pytest

from config.settings import NOAA_KEEP_COLS
from src.data.preprocessing import standardise_weather

TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def _write_psv(path, dates):
    """Write a PSV file with one row per entry in *dates*."""
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "DATE": d,
                "temperature": 10.0 + i,
                "dew_point_temperature": 5.0,
                "wind_direction": 180,
                "wind_speed": 3.5,
                "visibility": 16.0,
                "sea_level_pressure": 1013.2,
            }
        )
    pd.DataFrame(rows, columns=NOAA_KEEP_COLS).to_csv(path, sep="|", index=False)
    return path


def _run(tmp_path, dates_by_airport, year=2025):
    noaa_paths = {
        airport: _write_psv(tmp_path / f"{airport}.psv", dates)
        for airport, dates in dates_by_airport.items()
    }
    paths = {"processed": tmp_path}
    out = standardise_weather(noaa_paths, year=year, paths=paths)
    return pd.read_parquet(out)


def test_weather_standardise_filters_year(tmp_path):
    df = _run(
        tmp_path,
        {
            "ATL": [
                "2024-12-31T23:00:00",
                "2025-01-01T00:00:00",
                "2025-06-15T12:00:00",
                "2026-01-01T00:00:00",
            ]
        },
    )

    assert len(df) == 2
    years = pd.to_datetime(df["weather_time_str"]).dt.year
    assert set(years) == {2025}


def test_weather_standardise_deduplicates(tmp_path):
    df = _run(
        tmp_path,
        {
            "ATL": [
                "2025-03-01T10:00:00",
                "2025-03-01T10:00:00",
            ]
        },
    )

    assert len(df) == 1
    assert df.iloc[0]["airport"] == "ATL"


def test_weather_dedup_is_per_airport(tmp_path):
    """The same timestamp at two different airports must both be kept."""
    df = _run(
        tmp_path,
        {
            "ATL": ["2025-03-01T10:00:00"],
            "DEN": ["2025-03-01T10:00:00"],
        },
    )

    assert len(df) == 2
    assert set(df["airport"]) == {"ATL", "DEN"}


def test_weather_time_str_format(tmp_path):
    df = _run(
        tmp_path,
        {"ATL": ["2025-01-01T00:00:00", "2025-07-04T13:45:09"]},
    )

    assert "weather_time_str" in df.columns
    assert len(df) > 0
    for value in df["weather_time_str"]:
        assert TIME_RE.match(value), f"Unexpected format: {value!r}"
