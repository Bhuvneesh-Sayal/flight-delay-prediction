"""Unit tests for config/settings.py."""

from config.settings import (
    AIRPORT_RUNWAYS,
    AIRPORT_STATIONS,
    CV_FOLDS,
    DELAY_BINARY_THRESHOLD,
    DELAY_POSITIVE_THRESHOLD,
    PATHS,
    PROJECT_DIR,
    SPLIT_TEST_MONTHS,
    SPLIT_TRAIN_MONTHS,
    SPLIT_VALIDATION_MONTHS,
    TOP_10_AIRPORTS,
)


def test_project_dir_exists():
    assert PROJECT_DIR.exists()
    assert PROJECT_DIR.is_dir()


def test_all_paths_created():
    for name, path in PATHS.items():
        assert path.is_dir(), f"PATHS[{name!r}] is not a directory: {path}"


def test_top_10_airports_count():
    assert len(TOP_10_AIRPORTS) == 10
    assert len(set(TOP_10_AIRPORTS)) == 10


def test_airport_stations_keys_match():
    assert set(AIRPORT_STATIONS.keys()) == set(TOP_10_AIRPORTS)


def test_airport_runways_keys_match():
    assert set(AIRPORT_RUNWAYS.keys()) == set(TOP_10_AIRPORTS)


def test_split_months_are_contiguous():
    assert SPLIT_TRAIN_MONTHS[1] + 1 == SPLIT_VALIDATION_MONTHS[0]
    assert SPLIT_VALIDATION_MONTHS[1] + 1 == SPLIT_TEST_MONTHS[0]


def test_cv_folds_positive():
    assert CV_FOLDS > 0


def test_delay_thresholds_non_negative():
    assert DELAY_POSITIVE_THRESHOLD >= 0
    assert DELAY_BINARY_THRESHOLD >= 0
