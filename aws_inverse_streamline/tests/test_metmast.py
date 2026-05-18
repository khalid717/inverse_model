import csv
import os
import pytest
from datetime import datetime, timezone

import app.metmast as m_mod
from app.metmast import _haversine_m, MetMastClient


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(autouse=True)
def clear_caches():
    """Reset module-level caches before each test."""
    m_mod._MAST_LIST_CACHE.clear()
    m_mod._WIND_CSV_CACHE.clear()
    m_mod._DDB_MAST_CACHE.clear()
    yield


# ---------------------------------------------------------------------------
# _haversine_m
# ---------------------------------------------------------------------------

def test_haversine_zero_distance():
    assert _haversine_m(48.0, 11.0, 48.0, 11.0) == 0.0


def test_haversine_munich_berlin():
    # ~504 km
    d = _haversine_m(48.137, 11.576, 52.520, 13.405)
    assert 500_000 < d < 510_000


def test_haversine_symmetric():
    d1 = _haversine_m(48.0, 11.0, 52.0, 13.0)
    d2 = _haversine_m(52.0, 13.0, 48.0, 11.0)
    assert abs(d1 - d2) < 1e-6


# ---------------------------------------------------------------------------
# find_nearest_mast (local_csv)
# ---------------------------------------------------------------------------

def test_find_nearest_mast_picks_closest(tmp_path, monkeypatch):
    mast_path = str(tmp_path / "masts.csv")
    _write_csv(mast_path, [
        {"mast_id": "M1", "lat": "49.0", "lon": "10.0", "domain_id": "alps"},
        {"mast_id": "M2", "lat": "51.0", "lon": "12.0", "domain_id": "valley"},
    ], ["mast_id", "lat", "lon", "domain_id"])
    monkeypatch.setenv("METMAST_TABLE_PATH", mast_path)

    client = MetMastClient(mode="local_csv")
    mast = client.find_nearest_mast(49.1, 10.1)
    assert mast["mast_id"] == "M1"
    assert "distance_m" in mast


def test_find_nearest_mast_empty_file_raises(tmp_path, monkeypatch):
    mast_path = str(tmp_path / "empty.csv")
    _write_csv(mast_path, [], ["mast_id", "lat", "lon", "domain_id"])
    monkeypatch.setenv("METMAST_TABLE_PATH", mast_path)

    client = MetMastClient(mode="local_csv")
    with pytest.raises(RuntimeError, match="No metmasts"):
        client.find_nearest_mast(49.0, 10.0)


def test_find_nearest_mast_caches_on_second_call(tmp_path, monkeypatch):
    import builtins

    mast_path = str(tmp_path / "masts.csv")
    _write_csv(mast_path, [
        {"mast_id": "M1", "lat": "49.0", "lon": "10.0", "domain_id": ""},
    ], ["mast_id", "lat", "lon", "domain_id"])
    monkeypatch.setenv("METMAST_TABLE_PATH", mast_path)

    client = MetMastClient(mode="local_csv")
    client.find_nearest_mast(49.0, 10.0)
    assert mast_path in m_mod._MAST_LIST_CACHE

    # Second call with same mtime must not re-open the file
    real_open = builtins.open
    open_calls = {"n": 0}

    def tracking_open(path, *args, **kwargs):
        if path == mast_path:
            open_calls["n"] += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    mast = client.find_nearest_mast(49.0, 10.0)
    assert open_calls["n"] == 0, "Second call should use cache, not re-read the file"
    assert mast["mast_id"] == "M1"


# ---------------------------------------------------------------------------
# get_wind_at_time (local_csv)
# ---------------------------------------------------------------------------

def test_get_wind_picks_closest_timestamp(tmp_path, monkeypatch):
    wind_path = str(tmp_path / "wind.csv")
    _write_csv(wind_path, [
        {"mast_id": "M1", "timestamp_utc": "2026-02-02T10:00:00Z",
         "wspd_ms": "5.0", "wdir_from_deg": "270"},
        {"mast_id": "M1", "timestamp_utc": "2026-02-02T10:10:00Z",
         "wspd_ms": "6.0", "wdir_from_deg": "265"},
    ], ["mast_id", "timestamp_utc", "wspd_ms", "wdir_from_deg"])
    monkeypatch.setenv("METMAST_WIND_PATH", wind_path)

    client = MetMastClient(mode="local_csv")
    when = datetime(2026, 2, 2, 10, 3, 0, tzinfo=timezone.utc)
    wspd, wdir, ts = client.get_wind_at_time("M1", when)
    assert wspd == 5.0   # closer to 10:00 than 10:10
    assert wdir == 270.0


def test_get_wind_no_records_raises(tmp_path, monkeypatch):
    wind_path = str(tmp_path / "wind.csv")
    _write_csv(wind_path, [], ["mast_id", "timestamp_utc", "wspd_ms", "wdir_from_deg"])
    monkeypatch.setenv("METMAST_WIND_PATH", wind_path)

    client = MetMastClient(mode="local_csv")
    with pytest.raises(RuntimeError, match="No wind records"):
        client.get_wind_at_time("M1", datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_get_wind_caches_csv(tmp_path, monkeypatch):
    wind_path = str(tmp_path / "wind.csv")
    _write_csv(wind_path, [
        {"mast_id": "M1", "timestamp_utc": "2026-02-02T10:00:00Z",
         "wspd_ms": "5.0", "wdir_from_deg": "270"},
    ], ["mast_id", "timestamp_utc", "wspd_ms", "wdir_from_deg"])
    monkeypatch.setenv("METMAST_WIND_PATH", wind_path)

    client = MetMastClient(mode="local_csv")
    when = datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc)
    client.get_wind_at_time("M1", when)
    assert wind_path in m_mod._WIND_CSV_CACHE


def test_unsupported_mode_raises():
    client = MetMastClient(mode="csv")  # invalid
    with pytest.raises(ValueError, match="Unsupported"):
        client.find_nearest_mast(49.0, 10.0)
