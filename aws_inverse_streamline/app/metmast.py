from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List, Optional

import math

import boto3
from dateutil import parser as dtparser


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
            wind_table = os.getenv("METMAST_WIND_DDB")
            if not wind_table:
                raise RuntimeError("Set METMAST_WIND_DDB for dynamodb mode")
            table = self.ddb.Table(wind_table)

            # Strategy: query a small time window around 'when' and pick nearest.
            # Store timestamps as ISO strings like 2026-02-02T10:12:31Z.
            from datetime import timedelta
            start = (when - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
            end = (when + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")

            resp = table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key("mast_id").eq(mast_id) &
                                       boto3.dynamodb.conditions.Key("timestamp_utc").between(start, end)
            )
            items = resp.get("Items", [])
            if not items:
                raise RuntimeError(f"No wind records for mast_id={mast_id} within +/-10 min in DynamoDB")

            best = None
            best_dt = None
            best_delta = float("inf")
            for it in items:
                ts = dtparser.isoparse(it["timestamp_utc"]).astimezone(timezone.utc)
                delta = abs((ts - when).total_seconds())
                if delta < best_delta:
                    best_delta = delta
                    best = it
                    best_dt = ts

            return float(best["wspd_ms"]), float(best["wdir_from_deg"]), best_dt

        raise ValueError(f"Unsupported METMAST_MODE: {self.mode}")
