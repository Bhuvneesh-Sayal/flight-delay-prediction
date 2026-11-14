"""
src/data/ingestion.py
=====================
Handles all remote data acquisition:
  - SHA-256 checksum auditing
  - Cache-first / resume logic (avoids re-downloading)
  - Retry with exponential backoff on failed downloads
  - BTS monthly ZIP → CSV extraction
  - NOAA hourly PSV download

Public API
----------
  download_manifest : list[dict]  — populated by every download/cache call
  get_or_download(url, destination, min_bytes, max_retries) -> Path
  ingest_bts(year, paths, bts_base_url) -> list[str]
  ingest_noaa(year, paths, noaa_base_url, airport_stations) -> dict[str, Path]
"""

import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import requests

from config.settings import (
    AIRPORT_STATIONS,
    BTS_BASE_URL,
    NOAA_BASE_URL,
    PATHS,
    YEAR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level download manifest — appended to by every get_or_download call
# ---------------------------------------------------------------------------
download_manifest: list = []

_session = requests.Session()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1_048_576), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_once(url: str, dest: Path, temp: Path, min_bytes: int) -> None:
    """Single download attempt. Raises on HTTP error, network error or truncation."""
    try:
        with _session.get(url, stream=True, timeout=(60, 600)) as response:
            response.raise_for_status()
            with open(temp, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1_048_576):
                    if chunk:
                        fh.write(chunk)

        if temp.stat().st_size < min_bytes:
            raise RuntimeError(f"Download failed or truncated: {url}")
    except Exception:
        temp.unlink(missing_ok=True)
        raise

    temp.replace(dest)


def get_or_download(
    url: str,
    destination: Path,
    min_bytes: int = 1_000,
    max_retries: int = 5,
) -> Path:
    """
    Download *url* to *destination*, skipping if the file already exists
    and is at least *min_bytes* large.

    Uses an atomic .part temporary file so an interrupted download never
    leaves a corrupt file behind.

    On failure the download is retried up to *max_retries* times with
    exponential backoff (1s, 2s, 4s, ...). A warning is logged for each
    retry. When all retries are exhausted a RuntimeError is raised.

    Returns the destination Path.
    """
    dest = Path(destination)

    if dest.exists() and dest.stat().st_size >= min_bytes:
        download_manifest.append(
            {"url": url, "file": str(dest), "bytes": dest.stat().st_size, "status": "cached"}
        )
        logger.info("Cache hit: %s (%s bytes)", dest.name, f"{dest.stat().st_size:,}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".part")

    last_exc: Exception = None
    for attempt in range(max_retries + 1):
        try:
            logger.info("Downloading %s", url)
            _download_once(url, dest, temp, min_bytes)
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001 — retry on any download failure
            last_exc = exc
            if attempt < max_retries:
                delay = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s, 80s
                try:
                    logger.warning(
                        "Download failed for %s (%s). Retry %d/%d in %ds",
                        url, exc, attempt + 1, max_retries, delay,
                    )
                except Exception:
                    pass  # never let a logger flush crash stop the retry
                time.sleep(delay)
            else:
                try:
                    logger.error("Download failed for %s after %d retries: %s", url, max_retries, exc)
                except Exception:
                    pass

    if last_exc is not None:
        raise RuntimeError(
            f"Failed to download {url} after {max_retries} retries: {last_exc}"
        ) from last_exc

    download_manifest.append(
        {
            "url": url,
            "file": str(dest),
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "status": "downloaded",
        }
    )
    logger.info("Downloaded %s (%s bytes)", dest.name, f"{dest.stat().st_size:,}")
    return dest


# ---------------------------------------------------------------------------
# BTS ingestion
# ---------------------------------------------------------------------------

def ingest_bts(
    year: int = YEAR,
    paths: dict = PATHS,
    bts_base_url: str = BTS_BASE_URL,
) -> list:
    """
    Download and extract all 12 monthly BTS on-time performance ZIPs for
    *year*.  Returns a list of local CSV file paths (strings).

    Each ZIP contains a single CSV; the largest member is selected to avoid
    the BTS readme / metadata files that sometimes appear inside the archive.
    """
    bts_csv_files: list = []

    for month in range(1, 13):
        zip_name = (
            f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
        )
        zip_dest = paths["bts_raw"] / f"bts_{year}_{month:02d}.zip"
        csv_dest = paths["bts_raw"] / f"bts_{year}_{month:02d}.csv"

        get_or_download(f"{bts_base_url}/{zip_name}", zip_dest, min_bytes=50_000)

        if not csv_dest.exists() or csv_dest.stat().st_size < 1_000:
            with zipfile.ZipFile(zip_dest) as zf:
                csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                member = max(csv_members, key=lambda n: zf.getinfo(n).file_size)
                with zf.open(member) as src, open(csv_dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        bts_csv_files.append(str(csv_dest))
        logger.info("Validated BTS %d-%02d", year, month)

    return bts_csv_files


# ---------------------------------------------------------------------------
# NOAA ingestion
# ---------------------------------------------------------------------------

def ingest_noaa(
    year: int = YEAR,
    paths: dict = PATHS,
    noaa_base_url: str = NOAA_BASE_URL,
    airport_stations: dict = AIRPORT_STATIONS,
) -> dict:
    """
    Download GHCNh hourly PSV files for all airports in *airport_stations*.
    Returns a dict mapping airport code -> local Path.
    """
    noaa_paths: dict = {}

    for airport, station_id in airport_stations.items():
        filename = f"GHCNh_{station_id}_{year}.psv"
        dest = paths["noaa_raw"] / filename
        get_or_download(f"{noaa_base_url}/{filename}", dest, min_bytes=1_000)
        noaa_paths[airport] = dest
        logger.info("Validated NOAA station for %s", airport)

    return noaa_paths


# ---------------------------------------------------------------------------
# Manifest persistence
# ---------------------------------------------------------------------------

def save_manifest(paths: dict = PATHS) -> None:
    """Write the current download_manifest to logs/download_manifest.json."""
    out = paths["logs"] / "download_manifest.json"
    with open(out, "w") as fh:
        json.dump(download_manifest, fh, indent=2)
    logger.info("Download manifest saved → %s", out)


# ---------------------------------------------------------------------------
# Convenience: run everything
# ---------------------------------------------------------------------------

def _discover_cached(
    year: int = YEAR,
    paths: dict = PATHS,
    airport_stations: dict = AIRPORT_STATIONS,
) -> tuple:
    """
    Discover already-downloaded BTS CSVs and NOAA PSVs on disk.
    Used when `--steps prep` is run without `--steps ingest` in the same session.
    Raises FileNotFoundError if any expected file is missing.
    """
    bts_csv_files = []
    for month in range(1, 13):
        csv_path = paths["bts_raw"] / f"bts_{year}_{month:02d}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"BTS CSV not found: {csv_path}. Run --steps ingest first."
            )
        bts_csv_files.append(str(csv_path))

    noaa_paths_map = {}
    for airport, station_id in airport_stations.items():
        psv_path = paths["noaa_raw"] / f"GHCNh_{station_id}_{year}.psv"
        if not psv_path.exists():
            raise FileNotFoundError(
                f"NOAA PSV not found: {psv_path}. Run --steps ingest first."
            )
        noaa_paths_map[airport] = psv_path

    logger.info("Discovered %d BTS CSVs and %d NOAA PSVs on disk.",
                len(bts_csv_files), len(noaa_paths_map))
    return bts_csv_files, noaa_paths_map


def run_ingestion(
    year: int = YEAR,
    paths: dict = PATHS,
) -> tuple:
    """
    Run full ingestion (BTS + NOAA) and persist the manifest.
    Returns (bts_csv_files, noaa_paths).
    """
    logger.info("=== BTS Ingestion ===")
    bts_csv_files = ingest_bts(year=year, paths=paths)

    logger.info("=== NOAA Ingestion ===")
    noaa_paths = ingest_noaa(year=year, paths=paths)

    save_manifest(paths=paths)
    return bts_csv_files, noaa_paths
