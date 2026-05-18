import pytest
import requests
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from app.metmast import get_wind_at_latlon_time


def _make_response(times, speeds, dirs, status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {
        "hourly": {
            "time": times,
            "wind_speed_10m": speeds,
            "wind_direction_10m": dirs,
        }
    }
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = ""
    return mock_resp


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_returns_closest_timestamp():
    times = ["2026-02-02T10:00", "2026-02-02T11:00"]
    resp = _make_response(times, [5.0, 8.0], [270.0, 260.0])

    with patch("app.metmast.requests.get", return_value=resp):
        wspd, wdir, ts = get_wind_at_latlon_time(
            49.18, 10.33, datetime(2026, 2, 2, 10, 12, tzinfo=timezone.utc)
        )

    assert wspd == 5.0
    assert wdir == 270.0
    assert ts == datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc)


def test_picks_later_hour_when_closer():
    times = ["2026-02-02T10:00", "2026-02-02T11:00"]
    resp = _make_response(times, [5.0, 8.0], [270.0, 260.0])

    with patch("app.metmast.requests.get", return_value=resp):
        wspd, wdir, ts = get_wind_at_latlon_time(
            49.18, 10.33, datetime(2026, 2, 2, 10, 50, tzinfo=timezone.utc)
        )

    assert wspd == 8.0
    assert wdir == 260.0


# ---------------------------------------------------------------------------
# HTTP errors
# ---------------------------------------------------------------------------

def test_http_error_raises_runtime_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable"
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("503")

    with patch("app.metmast.requests.get", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="503"):
            get_wind_at_latlon_time(49.18, 10.33, datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc))


def test_network_failure_raises_runtime_error():
    with patch("app.metmast.requests.get", side_effect=requests.exceptions.ConnectionError("timeout")):
        with pytest.raises(RuntimeError, match="Open-Meteo request failed"):
            get_wind_at_latlon_time(49.18, 10.33, datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Malformed response
# ---------------------------------------------------------------------------

def test_missing_hourly_key_raises_runtime_error():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"latitude": 49.18}
    mock_resp.text = '{"latitude": 49.18}'

    with patch("app.metmast.requests.get", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="Unexpected Open-Meteo response"):
            get_wind_at_latlon_time(49.18, 10.33, datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Empty time series
# ---------------------------------------------------------------------------

def test_empty_time_series_raises_runtime_error():
    resp = _make_response([], [], [])

    with patch("app.metmast.requests.get", return_value=resp):
        with pytest.raises(RuntimeError, match="empty time series"):
            get_wind_at_latlon_time(49.18, 10.33, datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc))
