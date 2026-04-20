from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List, Optional

import math

import boto3
from dateutil import parser as dtparser
import requests
from datetime import datetime, timezone

def get_wind_at_latlon_time(lat: float, lon: float, when: datetime):
    """
    Fetch wind from Open-Meteo (hourly) for a given lat/lon and time.
    Returns (wspd_ms, wdir_from_deg, timestamp_used_utc)
    """
    when = when.astimezone(timezone.utc)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        "&hourly=wind_speed_10m,wind_direction_10m"
        "&timezone=UTC"
    )

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    hourly = r.json()["hourly"]

    times = hourly["time"]
    speeds = hourly["wind_speed_10m"]
    dirs = hourly["wind_direction_10m"]

    best_idx = min(
        range(len(times)),
        key=lambda i: abs(
            datetime.fromisoformat(times[i]).replace(tzinfo=timezone.utc) - when
        ).total_seconds()
    )

    ts_used = datetime.fromisoformat(times[best_idx]).replace(tzinfo=timezone.utc)
    return float(speeds[best_idx]), float(dirs[best_idx]), ts_used

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


@dataclass
class MetMastClient:
    """
    Fetches wind speed + direction from the nearest metmast.

    Supported modes:
    - local_csv: uses two local files:
        METMAST_TABLE_PATH: CSV with columns mast_id,lat,lon,domain_id(optional)
        METMAST_WIND_PATH:  CSV with columns mast_id,timestamp_utc,wspd_ms,wdir_from_deg
    - dynamodb:
        METMAST_TABLE_DDB: DynamoDB table for mast metadata (mast_id PK)
        METMAST_WIND_DDB:  DynamoDB table for wind (mast_id PK, timestamp_utc SK)
    - timestream: placeholder (add your query when ready)
    """
    mode: str = "local_csv"

    def __post_init__(self):
        self.mode = (self.mode or "local_csv").lower().strip()
        if self.mode == "dynamodb":
            self.ddb = boto3.resource("dynamodb")
        else:
            self.ddb = None

    def _load_masts_local(self) -> List[Dict[str, Any]]:
        path = os.getenv("METMAST_TABLE_PATH", "metmasts.csv")
        masts = []
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(",")
                row = dict(zip(header, parts))
                masts.append({
                    "mast_id": row["mast_id"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "domain_id": row.get("domain_id") or None
                })
        if not masts:
            raise RuntimeError("No metmasts found in METMAST_TABLE_PATH")
        return masts

    def find_nearest_mast(self, lat: float, lon: float) -> Dict[str, Any]:
        if self.mode == "local_csv":
            masts = self._load_masts_local()
        elif self.mode == "dynamodb":
            table_name = os.getenv("METMAST_TABLE_DDB")
            if not table_name:
                raise RuntimeError("Set METMAST_TABLE_DDB for dynamodb mode")
            table = self.ddb.Table(table_name)
            # Scan is OK if you have few masts; for many masts, keep a precomputed list or use geo index.
            resp = table.scan()
            masts = [{
                "mast_id": it["mast_id"],
                "lat": float(it["lat"]),
                "lon": float(it["lon"]),
                "domain_id": it.get("domain_id")
            } for it in resp.get("Items", [])]
        else:
            raise ValueError(f"Unsupported METMAST_MODE: {self.mode}")

        best = None
        best_d = float("inf")
        for m in masts:
            d = _haversine_m(lat, lon, m["lat"], m["lon"])
            if d < best_d:
                best_d = d
                best = m
        if best is None:
            raise RuntimeError("Could not select nearest metmast.")
        best["distance_m"] = best_d
        return best

    def get_wind_at_time(self, mast_id: str, when: datetime) -> Tuple[float, float, datetime]:
        """
        Returns (wspd_ms, wdir_from_deg, timestamp_used_utc) closest to 'when'.
        """
        when = when.astimezone(timezone.utc)

        if self.mode == "local_csv":
            path = os.getenv("METMAST_WIND_PATH", "metmast_wind.csv")
            best = None
            best_dt = None
            best_delta = float("inf")

            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split(",")
                    row = dict(zip(header, parts))
                    if row["mast_id"] != mast_id:
                        continue
                    ts = dtparser.isoparse(row["timestamp_utc"]).astimezone(timezone.utc)
                    delta = abs((ts - when).total_seconds())
                    if delta < best_delta:
                        best_delta = delta
                        best = row
                        best_dt = ts

            if best is None:
                raise RuntimeError(f"No wind records for mast_id={mast_id} in {path}")

            return float(best["wspd_ms"]), float(best["wdir_from_deg"]), best_dt

        if self.mode == "dynamodb":
        # Instead of reading DynamoDB, fetch directly from Open-Meteo
            import requests

            # We need mast location to call API
            mast_table = os.getenv("METMAST_TABLE_DDB")
            if not mast_table:
                raise RuntimeError("Set METMAST_TABLE_DDB for dynamodb mode")
            
            mast_tbl = self.ddb.Table(mast_table)
            mast_resp = mast_tbl.get_item(Key={"mast_id": mast_id})
            mast = mast_resp.get("Item")

            if not mast:
                raise RuntimeError(f"Mast {mast_id} not found in metadata table")

            lat = float(mast["lat"])
            lon = float(mast["lon"])

            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}"
                f"&longitude={lon}"
                "&hourly=wind_speed_10m,wind_direction_10m"
                "&timezone=UTC"
            )

            r = requests.get(url, timeout=10)
            data = r.json()["hourly"]

            times = data["time"]
            speeds = data["wind_speed_10m"]
            dirs = data["wind_direction_10m"]

            best = None
            best_dt = None
            best_delta = float("inf")

            for i in range(len(times)):
                ts = datetime.fromisoformat(times[i]).replace(tzinfo=timezone.utc)
                delta = abs((ts - when).total_seconds())
                if delta < best_delta:
                    best_delta = delta
                    best = (speeds[i], dirs[i])
                    best_dt = ts

            if best is None:
                raise RuntimeError("No wind data from API")

            return float(best[0]), float(best[1]), best_dt