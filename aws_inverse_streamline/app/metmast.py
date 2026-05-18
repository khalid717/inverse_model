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

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Open-Meteo returned HTTP {r.status_code} for lat={lat}, lon={lon}: {r.text[:200]}"
        ) from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Open-Meteo request failed for lat={lat}, lon={lon}: {e}"
        ) from e

    try:
        hourly = r.json()["hourly"]
        times = hourly["time"]
        speeds = hourly["wind_speed_10m"]
        dirs = hourly["wind_direction_10m"]
    except (KeyError, ValueError) as e:
        raise RuntimeError(
            f"Unexpected Open-Meteo response structure for lat={lat}, lon={lon}: {e}. "
            f"Response: {r.text[:200]}"
        ) from e

    if not times:
        raise RuntimeError(
            f"Open-Meteo returned empty time series for lat={lat}, lon={lon}, when={when.isoformat()}. "
            "Requested time may be outside the forecast window."
        )

    best_idx = min(
        range(len(times)),
        key=lambda i: abs(
            datetime.fromisoformat(times[i]).replace(tzinfo=timezone.utc) - when
        ).total_seconds()
    )

    ts_used = datetime.fromisoformat(times[best_idx]).replace(tzinfo=timezone.utc)
    return float(speeds[best_idx]), float(dirs[best_idx]), ts_used
