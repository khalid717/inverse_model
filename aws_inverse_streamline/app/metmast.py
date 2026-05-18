from __future__ import annotations

import os
from datetime import datetime, timezone

import requests
from dateutil import parser as dtparser


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
