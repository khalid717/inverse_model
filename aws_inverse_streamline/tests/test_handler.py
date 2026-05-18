import json
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pyproj import Transformer
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import app.handler as handler_mod
from app.handler import _parse_alert


# ---------------------------------------------------------------------------
# _parse_alert
# ---------------------------------------------------------------------------

def test_parse_alert_standard_keys():
    event = {
        "timestamp_utc": "2026-02-02T10:12:31Z",
        "sensor_id": "SENSOR_04",
        "lat": 49.179924,
        "lon": 10.336225,
        "pm25": 800.0,
    }
    a = _parse_alert(event)
    assert a["sensor_id"] == "SENSOR_04"
    assert a["lat"] == pytest.approx(49.179924)
    assert a["lon"] == pytest.approx(10.336225)
    assert a["pm25"] == 800.0
    assert isinstance(a["timestamp"], datetime)


def test_parse_alert_nested_payload():
    event = {"alert": {
        "timestamp_utc": "2026-02-02T10:12:31Z",
        "sensor_id": "S1", "lat": 48.0, "lon": 11.0, "pm25": 100.0,
    }}
    a = _parse_alert(event)
    assert a["sensor_id"] == "S1"


def test_parse_alert_accepts_time_key():
    event = {"time": "2026-02-02T10:00:00Z", "lat": 48.0, "lon": 11.0, "pm25": 50.0}
    a = _parse_alert(event)
    assert isinstance(a["timestamp"], datetime)


def test_parse_alert_accepts_timestamp_key():
    event = {"timestamp": "2026-02-02T10:00:00Z", "lat": 48.0, "lon": 11.0, "pm25": 50.0}
    a = _parse_alert(event)
    assert isinstance(a["timestamp"], datetime)


def test_parse_alert_missing_timestamp_raises():
    with pytest.raises((ValueError, KeyError)):
        _parse_alert({"lat": 48.0, "lon": 11.0, "pm25": 100.0})


def test_parse_alert_unknown_sensor_id():
    event = {"timestamp_utc": "2026-02-02T10:00:00Z", "lat": 48.0, "lon": 11.0, "pm25": 50.0}
    a = _parse_alert(event)
    assert a["sensor_id"] == "unknown"


# ---------------------------------------------------------------------------
# lambda_handler  (end-to-end with mocked I/O, real model)
# ---------------------------------------------------------------------------

def _write_uniform(path, value, crs_epsg, west, south, east, north, w=30, h=30):
    transform = from_bounds(west, south, east, north, w, h)
    data = np.full((1, h, w), value, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs=CRS.from_epsg(crs_epsg), transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture()
def uniform_rasters(tmp_path):
    crs_epsg = 32632
    west, south, east, north = 500000.0, 5440000.0, 501000.0, 5441000.0

    vel = str(tmp_path / "vel.tif")
    ang = str(tmp_path / "ang.tif")
    dsm = str(tmp_path / "dsm.tif")
    _write_uniform(vel, 5.0,   crs_epsg, west, south, east, north)
    _write_uniform(ang, 270.0, crs_epsg, west, south, east, north)
    _write_uniform(dsm, 500.0, crs_epsg, west, south, east, north)

    back = Transformer.from_crs(crs_epsg, 4326, always_xy=True)
    cx, cy = (west + east) / 2, (south + north) / 2
    sensor_lon, sensor_lat = back.transform(cx, cy)

    return {
        "vel": vel, "ang": ang, "dsm": dsm,
        "sensor_lat": sensor_lat, "sensor_lon": sensor_lon,
        "crs_epsg": crs_epsg,
    }


def test_lambda_handler_returns_200(tmp_path, monkeypatch, uniform_rasters):
    r = uniform_rasters
    index_line = json.dumps({
        "wspd_ms": 5.0, "dir_deg": 270,
        "vel_ref": r["vel"], "ang_ref": r["ang"], "dsm_ref": r["dsm"],
        "domain_id": "test",
    })

    monkeypatch.setenv("STORE_MODE", "local")
    monkeypatch.setenv("LOCAL_LIBRARY_ROOT", str(tmp_path))
    monkeypatch.setenv("METMAST_MODE", "local_csv")
    monkeypatch.setenv("FORCE_MODEL_CRS", f"EPSG:{r['crs_epsg']}")
    monkeypatch.setenv("D_MAX_GLOBAL_M", "400.0")

    handler_mod._INDEX_CACHE = None

    with (patch("app.handler.get_wind_at_latlon_time",
                return_value=(5.0, 270.0, datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc))),
          patch("app.storage.LocalStore.get_text", return_value=index_line),
          patch("app.storage.LocalStore.materialize", side_effect=lambda ref, tmpdir: ref),
          patch("app.handler._iot_publish")):

        event = {
            "timestamp_utc": "2026-02-02T10:12:31Z",
            "sensor_id": "SENSOR_04",
            "lat": r["sensor_lat"],
            "lon": r["sensor_lon"],
            "pm25": 800.0,
        }
        resp = handler_mod.lambda_handler(event)

    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert "source_band" in body
    assert set(body["source_band"]) == {"near", "mid", "far"}
    assert body["alert"]["sensor_id"] == "SENSOR_04"
    assert body["metmast"]["wind_source"] == "open-meteo"
    assert body["windfield"]["domain_id"] == "test"


def test_lambda_handler_index_cached_across_calls(tmp_path, monkeypatch, uniform_rasters):
    """_INDEX_CACHE must not be reloaded on a second invocation."""
    r = uniform_rasters
    index_line = json.dumps({
        "wspd_ms": 5.0, "dir_deg": 270,
        "vel_ref": r["vel"], "ang_ref": r["ang"], "dsm_ref": r["dsm"],
        "domain_id": "test",
    })

    monkeypatch.setenv("STORE_MODE", "local")
    monkeypatch.setenv("LOCAL_LIBRARY_ROOT", str(tmp_path))
    monkeypatch.setenv("METMAST_MODE", "local_csv")
    monkeypatch.setenv("FORCE_MODEL_CRS", f"EPSG:{r['crs_epsg']}")
    monkeypatch.setenv("D_MAX_GLOBAL_M", "400.0")

    handler_mod._INDEX_CACHE = None
    get_text_mock = MagicMock(return_value=index_line)

    event = {
        "timestamp_utc": "2026-02-02T10:12:31Z",
        "sensor_id": "S1",
        "lat": r["sensor_lat"],
        "lon": r["sensor_lon"],
        "pm25": 800.0,
    }

    with (patch("app.handler.get_wind_at_latlon_time",
                return_value=(5.0, 270.0, datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc))),
          patch("app.storage.LocalStore.get_text", get_text_mock),
          patch("app.storage.LocalStore.materialize", side_effect=lambda ref, tmpdir: ref),
          patch("app.handler._iot_publish")):

        handler_mod.lambda_handler(event)
        handler_mod.lambda_handler(event)

    assert get_text_mock.call_count == 1, "Index should be loaded only once"


def test_iot_not_published_when_disabled(tmp_path, monkeypatch, uniform_rasters):
    r = uniform_rasters
    index_line = json.dumps({
        "wspd_ms": 5.0, "dir_deg": 270,
        "vel_ref": r["vel"], "ang_ref": r["ang"], "dsm_ref": r["dsm"],
        "domain_id": "test",
    })

    monkeypatch.setenv("STORE_MODE", "local")
    monkeypatch.setenv("LOCAL_LIBRARY_ROOT", str(tmp_path))
    monkeypatch.setenv("FORCE_MODEL_CRS", f"EPSG:{r['crs_epsg']}")
    monkeypatch.setenv("D_MAX_GLOBAL_M", "400.0")
    monkeypatch.setenv("PUBLISH_IGNITION", "0")

    handler_mod._INDEX_CACHE = None

    event = {
        "timestamp_utc": "2026-02-02T10:12:31Z",
        "sensor_id": "S1",
        "lat": r["sensor_lat"],
        "lon": r["sensor_lon"],
        "pm25": 800.0,
    }

    with (patch("app.handler.get_wind_at_latlon_time",
                return_value=(5.0, 270.0, datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc))),
          patch("app.storage.LocalStore.get_text", return_value=index_line),
          patch("app.storage.LocalStore.materialize", side_effect=lambda ref, tmpdir: ref),
          patch("app.handler._iot_publish") as mock_publish):

        handler_mod.lambda_handler(event)

    mock_publish.assert_not_called()
