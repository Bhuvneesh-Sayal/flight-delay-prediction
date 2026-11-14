"""Unit tests for src/data/ingestion.py (no network access)."""

import hashlib
import importlib
from unittest.mock import MagicMock, patch

import pytest
import requests

import src.data.ingestion as ingestion_module


@pytest.fixture
def ingestion():
    """Reload the module so download_manifest and the session start fresh."""
    return importlib.reload(ingestion_module)


def _mock_response(payload: bytes = b"x" * 2000) -> MagicMock:
    """A mock streaming response usable with or without a `with` block."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = [payload]
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_sha256_file(ingestion, tmp_path):
    content = b"flight delay prediction"
    f = tmp_path / "sample.bin"
    f.write_bytes(content)

    assert ingestion.sha256_file(f) == hashlib.sha256(content).hexdigest()


def test_get_or_download_cached(ingestion, tmp_path):
    dest = tmp_path / "cached.zip"
    dest.write_bytes(b"y" * 5000)

    with patch.object(requests.Session, "get") as mock_get:
        result = ingestion.get_or_download(
            "http://example.com/cached.zip", dest, min_bytes=1000
        )

    mock_get.assert_not_called()
    assert result == dest
    assert len(ingestion.download_manifest) == 1
    assert ingestion.download_manifest[0]["status"] == "cached"
    assert ingestion.download_manifest[0]["bytes"] == 5000


def test_get_or_download_downloads(ingestion, tmp_path):
    dest = tmp_path / "sub" / "new.zip"

    with patch.object(
        requests.Session, "get", return_value=_mock_response()
    ) as mock_get:
        result = ingestion.get_or_download(
            "http://example.com/new.zip", dest, min_bytes=1000
        )

    mock_get.assert_called_once()
    assert result == dest
    assert dest.exists()
    assert dest.stat().st_size == 2000
    assert not dest.with_suffix(dest.suffix + ".part").exists()
    entry = ingestion.download_manifest[-1]
    assert entry["status"] == "downloaded"
    assert entry["sha256"] == hashlib.sha256(b"x" * 2000).hexdigest()


def test_get_or_download_retries_on_failure(ingestion, tmp_path):
    """Two ConnectionErrors followed by success. Requires retry logic in
    get_or_download."""
    dest = tmp_path / "retry.zip"
    side_effects = [
        requests.exceptions.ConnectionError("boom 1"),
        requests.exceptions.ConnectionError("boom 2"),
        _mock_response(),
    ]

    # Patch sleep so any backoff in the retry logic does not slow the test.
    with patch.object(
        requests.Session, "get", side_effect=side_effects
    ) as mock_get, patch("time.sleep"):
        ingestion.get_or_download(
            "http://example.com/retry.zip", dest, min_bytes=1000
        )

    assert mock_get.call_count == 3
    assert dest.exists()
    assert ingestion.download_manifest[-1]["status"] == "downloaded"
