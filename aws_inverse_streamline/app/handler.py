"""
AWS-ready inverse gas dispersion runner (streamline back-trajectory).

What it does (end-to-end):
1) Takes an ALERT payload (timestamp, sensor lat/lon, PM2.5).
2) Finds the nearest metmast to the sensor.
3) Fetches wind speed + direction (FROM, degrees) closest to alert timestamp.
4) Matches the closest pre-simulated WindNinja windfield from an index.
5) Downloads vel.asc + ang.asc (+ DSM GeoTIFF) from S3 (or uses local paths).
6) Runs streamline back-trajectory inverse to estimate a source band (min/mid/max).
7) Emits a JSON result (and optionally writes GeoJSON trace).

You can run this locally OR as an AWS Lambda container:
- Local: python -m app.handler --alert alert.json
- Lambda: set HANDLER entrypoint to app.handler.lambda_handler

Dependencies:
  pip install numpy rasterio pyproj boto3 python-dateutil

Notes:
- This implementation assumes WindNinja ang.asc is "direction FROM" in degrees.
- The back-trajectory steps upstream (opposite to flow direction).

Author: generated from your notebook prototype (cell 3), refactored for automation.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

from dateutil import parser as dtparser

from .metmast import MetMastClient
from .windfield_index import WindfieldIndex, match_windfield
from .inverse_streamline import run_inverse_streamline, InverseParams
from .storage import S3Store, LocalStore, Store


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _parse_alert(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects event payload (from MQTT / IoT Core / HTTP / local):
      {
        "timestamp_utc": "2026-02-02T10:12:31Z",
        "sensor_id": "SENSOR_04",
        "lat": 49.179924,
        "lon": 10.336225,
        "pm25": 800.0
      }
    """
    # Allow nested payloads (common with IoT rules)
    payload = event.get("alert", event)

    ts = payload.get("timestamp_utc") or payload.get("timestamp") or payload.get("time")
    if not ts:
        raise ValueError("Alert missing timestamp_utc/timestamp/time")
    timestamp = dtparser.isoparse(ts)

    lat = float(payload["lat"])
    lon = float(payload["lon"])
    pm25 = float(payload["pm25"])

    return {
        "timestamp": timestamp,
        "sensor_id": payload.get("sensor_id", "unknown"),
        "lat": lat,
        "lon": lon,
        "pm25": pm25,
        "raw": payload,
    }


def _build_store() -> Store:
    """
    Choose where to load artifacts (index, windfields, DSM) from.
    - STORE_MODE=s3 (recommended for AWS)
    - STORE_MODE=local (for dev/testing)
    """
    mode = os.getenv("STORE_MODE", "s3").lower().strip()
    if mode == "s3":
        bucket = _env("S3_BUCKET")
        prefix = os.getenv("S3_PREFIX", "windninja_library").strip("/")
        return S3Store(bucket=bucket, prefix=prefix)
    if mode == "local":
        root = _env("LOCAL_LIBRARY_ROOT")
        return LocalStore(root=root)
    raise ValueError(f"Unsupported STORE_MODE: {mode}")


def _build_metmast_client() -> MetMastClient:
    """
    Choose how to fetch metmast wind data.
    - METMAST_MODE=local_csv  (dev)
    - METMAST_MODE=dynamodb   (AWS)
    - METMAST_MODE=timestream (AWS, optional)
    """
    mode = os.getenv("METMAST_MODE", "local_csv").lower().strip()
    return MetMastClient(mode=mode)


def _load_index(store: Store) -> WindfieldIndex:
    """
    Loads the windfield index:
      - from S3: s3://bucket/<prefix>/index/windfield_index.jsonl
      - from local: <LOCAL_LIBRARY_ROOT>/index/windfield_index.jsonl
    Cached in memory per warm Lambda.
    """
    index_key = os.getenv("WINDFIELD_INDEX_KEY", "index/windfield_index.jsonl")
    # store.get_text handles s3/local
    txt = store.get_text(index_key)
    return WindfieldIndex.from_jsonl(txt)


def _ensure_local_files(store: Store, record: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Ensure vel/ang/dsm are present on local filesystem for rasterio.
    - For S3 store: download to /tmp (Lambda) or temp dir (local).
    - For Local store: returns local paths directly.
    """
    vel_ref = record["vel_ref"]
    ang_ref = record["ang_ref"]
    dsm_ref = record["dsm_ref"]

    # Local store returns OS paths; S3 store returns keys.
    # We always "materialize" to real paths.
    tmpdir = os.getenv("WORK_DIR") or tempfile.gettempdir()
    vel_path = store.materialize(vel_ref, tmpdir=tmpdir)
    ang_path = store.materialize(ang_ref, tmpdir=tmpdir)
    dsm_path = store.materialize(dsm_ref, tmpdir=tmpdir)
    return vel_path, ang_path, dsm_path


# --------------------
# Lambda entrypoint
# --------------------

_INDEX_CACHE: Optional[WindfieldIndex] = None


def lambda_handler(event, context=None):
    """
    AWS Lambda handler.
    """
    global _INDEX_CACHE

    alert = _parse_alert(event)

    store = _build_store()
    met = _build_metmast_client()

    if _INDEX_CACHE is None:
        _INDEX_CACHE = _load_index(store)

    # 1) Find nearest metmast + fetch wind at time
    mast = met.find_nearest_mast(alert["lat"], alert["lon"])
    wspd, wdir_from, wind_ts = met.get_wind_at_time(mast["mast_id"], alert["timestamp"])

    # 2) Match closest windfield
    record = match_windfield(_INDEX_CACHE, wspd, wdir_from, domain_hint=mast.get("domain_id"))

    # 3) Materialize needed files
    vel_path, ang_path, dsm_path = _ensure_local_files(store, record)

    # 4) Run inverse streamline model
    params = InverseParams.from_env()
    result = run_inverse_streamline(
        sensor_lat=alert["lat"],
        sensor_lon=alert["lon"],
        pm25_obs=alert["pm25"],
        vel_path=vel_path,
        ang_path=ang_path,
        dsm_path=dsm_path,
        params=params,
    )

    # 5) Attach metadata
    result["alert"] = {
        "sensor_id": alert["sensor_id"],
        "timestamp_utc": alert["timestamp"].isoformat(),
        "lat": alert["lat"],
        "lon": alert["lon"],
        "pm25": alert["pm25"],
    }
    result["metmast"] = {
        "mast_id": mast["mast_id"],
        "mast_lat": mast["lat"],
        "mast_lon": mast["lon"],
        "wind_timestamp_utc": wind_ts.isoformat(),
        "wspd_ms": wspd,
        "wdir_from_deg": wdir_from,
    }
    result["windfield"] = {
        "domain_id": record.get("domain_id"),
        "wspd_ms": record.get("wspd_ms"),
        "dir_deg": record.get("dir_deg"),
        "vel_ref": record.get("vel_ref"),
        "ang_ref": record.get("ang_ref"),
        "dsm_ref": record.get("dsm_ref"),
        "match_type": record.get("match_type"),
    }

    # Optional: write result + trace GeoJSON to S3
    if os.getenv("WRITE_OUTPUTS", "0") == "1":
        out_prefix = os.getenv("OUTPUT_PREFIX", "outputs").strip("/")
        key_base = f"{out_prefix}/{alert['sensor_id']}/{alert['timestamp'].strftime('%Y%m%dT%H%M%SZ')}"
        store.put_text(f"{key_base}.json", json.dumps(result, indent=2))
        if result.get("trace_geojson"):
            store.put_text(f"{key_base}_trace.geojson", json.dumps(result["trace_geojson"]))

    return {
        "statusCode": 200,
        "body": json.dumps(result),
        "headers": {"Content-Type": "application/json"},
    }


# --------------------
# Local CLI runner
# --------------------

def _cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--alert", required=True, help="Path to alert JSON file")
    args = p.parse_args()

    event = json.loads(open(args.alert, "r", encoding="utf-8").read())
    resp = lambda_handler(event, None)
    body = json.loads(resp["body"])
    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    _cli()
