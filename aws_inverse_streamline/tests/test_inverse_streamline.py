import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pyproj import Transformer

from app.inverse_streamline import (
    decay_k,
    distance_band_from_obs,
    interpolate_point_linear,
    inside,
    _validate_rasters,
    InverseParams,
    run_inverse_streamline,
)


# ---------------------------------------------------------------------------
# decay_k
# ---------------------------------------------------------------------------

def test_decay_k_roundtrip():
    k = decay_k(1000.0, 45.0, 15.0)
    assert k > 0
    assert abs(1000.0 * np.exp(-k * 15.0) - 45.0) < 1e-6


def test_decay_k_invalid_c_ref_above_c0():
    with pytest.raises(ValueError):
        decay_k(100.0, 200.0, 15.0)


def test_decay_k_invalid_zero_c_ref():
    with pytest.raises(ValueError):
        decay_k(100.0, 0.0, 15.0)


# ---------------------------------------------------------------------------
# distance_band_from_obs
# ---------------------------------------------------------------------------

def test_distance_band_typical():
    k = decay_k(1000.0, 45.0, 15.0)
    # Use C_obs=100 so both C0_min=500 and C0_max=2000 exceed it → two positive d values
    d_min, d_max = distance_band_from_obs(100.0, k, 500.0, 2000.0, 2.0, 1000.0)
    assert 0 < d_min < d_max
    assert d_min >= 2.0
    assert d_max <= 1000.0


def test_distance_band_above_c0_max_clamps_to_near():
    k = decay_k(1000.0, 45.0, 15.0)
    d_min, d_max = distance_band_from_obs(5000.0, k, 500.0, 2000.0, 2.0, 1000.0)
    assert d_min >= 2.0
    assert d_max > d_min


def test_distance_band_very_low_obs_clamps_to_global_max():
    k = decay_k(1000.0, 45.0, 15.0)
    d_min, d_max = distance_band_from_obs(0.001, k, 500.0, 2000.0, 2.0, 1000.0)
    assert d_max <= 1000.0


def test_distance_band_invalid_zero_obs():
    k = decay_k(1000.0, 45.0, 15.0)
    with pytest.raises(ValueError):
        distance_band_from_obs(0.0, k, 500.0, 2000.0, 2.0, 1000.0)


# ---------------------------------------------------------------------------
# interpolate_point_linear
# ---------------------------------------------------------------------------

def test_interpolate_midpoint():
    xs = np.array([0.0, 10.0, 20.0])
    ys = np.array([0.0, 10.0, 20.0])
    ds = np.array([0.0, 10.0, 20.0])
    x, y = interpolate_point_linear(xs, ys, ds, 5.0)
    assert abs(x - 5.0) < 1e-9
    assert abs(y - 5.0) < 1e-9


def test_interpolate_clamps_before_start():
    xs = np.array([0.0, 10.0])
    ys = np.array([0.0, 10.0])
    ds = np.array([0.0, 10.0])
    x, y = interpolate_point_linear(xs, ys, ds, -5.0)
    assert x == 0.0 and y == 0.0


def test_interpolate_clamps_after_end():
    xs = np.array([0.0, 10.0])
    ys = np.array([0.0, 10.0])
    ds = np.array([0.0, 10.0])
    x, y = interpolate_point_linear(xs, ys, ds, 100.0)
    assert x == 10.0 and y == 10.0


def test_interpolate_at_exact_node():
    xs = np.array([0.0, 5.0, 10.0])
    ys = np.array([0.0, 3.0, 6.0])
    ds = np.array([0.0, 5.0, 10.0])
    x, y = interpolate_point_linear(xs, ys, ds, 5.0)
    assert abs(x - 5.0) < 1e-9
    assert abs(y - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# inside
# ---------------------------------------------------------------------------

def test_inside_true():
    from rasterio.coords import BoundingBox
    b = BoundingBox(left=0, bottom=0, right=100, top=100)
    assert inside(b, 50, 50)


def test_inside_false_outside():
    from rasterio.coords import BoundingBox
    b = BoundingBox(left=0, bottom=0, right=100, top=100)
    assert not inside(b, 150, 50)


# ---------------------------------------------------------------------------
# _validate_rasters
# ---------------------------------------------------------------------------

def _open_uniform_raster(path, value, crs_epsg, west, south, east, north, w=10, h=10):
    transform = from_bounds(west, south, east, north, w, h)
    data = np.full((1, h, w), value, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs=CRS.from_epsg(crs_epsg), transform=transform,
    ) as dst:
        dst.write(data)


def test_validate_rasters_ok(tmp_path):
    for name, val in [("vel.tif", 5.0), ("ang.tif", 270.0), ("dsm.tif", 500.0)]:
        _open_uniform_raster(str(tmp_path / name), val, 32632,
                             500000, 5440000, 501000, 5441000)
    with (rasterio.open(str(tmp_path / "vel.tif")) as v,
          rasterio.open(str(tmp_path / "ang.tif")) as a,
          rasterio.open(str(tmp_path / "dsm.tif")) as d):
        _validate_rasters(v, a, d)  # should not raise


def test_validate_rasters_crs_mismatch(tmp_path):
    _open_uniform_raster(str(tmp_path / "vel.tif"), 5.0, 32632,
                         500000, 5440000, 501000, 5441000)
    _open_uniform_raster(str(tmp_path / "ang.tif"), 270.0, 32633,  # wrong CRS
                         500000, 5440000, 501000, 5441000)
    _open_uniform_raster(str(tmp_path / "dsm.tif"), 500.0, 32632,
                         500000, 5440000, 501000, 5441000)
    with (rasterio.open(str(tmp_path / "vel.tif")) as v,
          rasterio.open(str(tmp_path / "ang.tif")) as a,
          rasterio.open(str(tmp_path / "dsm.tif")) as d):
        with pytest.raises(ValueError, match="ang CRS"):
            _validate_rasters(v, a, d)


def test_validate_rasters_shape_mismatch(tmp_path):
    _open_uniform_raster(str(tmp_path / "vel.tif"), 5.0,   32632,
                         500000, 5440000, 501000, 5441000, w=10, h=10)
    _open_uniform_raster(str(tmp_path / "ang.tif"), 270.0, 32632,
                         500000, 5440000, 501000, 5441000, w=20, h=20)  # different size
    _open_uniform_raster(str(tmp_path / "dsm.tif"), 500.0, 32632,
                         500000, 5440000, 501000, 5441000)
    with (rasterio.open(str(tmp_path / "vel.tif")) as v,
          rasterio.open(str(tmp_path / "ang.tif")) as a,
          rasterio.open(str(tmp_path / "dsm.tif")) as d):
        with pytest.raises(ValueError, match="ang grid"):
            _validate_rasters(v, a, d)


# ---------------------------------------------------------------------------
# run_inverse_streamline  (integration)
# ---------------------------------------------------------------------------

def _write_uniform(path, value, crs_epsg, west, south, east, north, w=30, h=30):
    transform = from_bounds(west, south, east, north, w, h)
    data = np.full((1, h, w), value, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs=CRS.from_epsg(crs_epsg), transform=transform,
    ) as dst:
        dst.write(data)


def test_run_inverse_streamline_uniform_westerly(tmp_path):
    crs_epsg = 32632
    west, south, east, north = 500000.0, 5440000.0, 501000.0, 5441000.0

    _write_uniform(str(tmp_path / "vel.tif"), 5.0,   crs_epsg, west, south, east, north)
    _write_uniform(str(tmp_path / "ang.tif"), 270.0, crs_epsg, west, south, east, north)
    _write_uniform(str(tmp_path / "dsm.tif"), 500.0, crs_epsg, west, south, east, north)

    # Sensor at domain centre
    back = Transformer.from_crs(crs_epsg, 4326, always_xy=True)
    sensor_lon, sensor_lat = back.transform(
        (west + east) / 2, (south + north) / 2
    )

    params = InverseParams(
        C0_ref=1000.0, C_ref=45.0, d_ref_m=15.0,
        C0_min_factor=0.5, C0_max_factor=2.0,
        d_min_global_m=2.0, d_max_global_m=400.0,
        step_length_m=2.0, min_wind_speed_ms=0.1,
        force_model_crs=f"EPSG:{crs_epsg}",
    )

    # Use C_obs=100 (below C0_min=500) so both C0_min and C0_max give positive
    # distances, ensuring a genuine band (min < max).
    result = run_inverse_streamline(
        sensor_lat=sensor_lat,
        sensor_lon=sensor_lon,
        pm25_obs=100.0,
        vel_path=str(tmp_path / "vel.tif"),
        ang_path=str(tmp_path / "ang.tif"),
        dsm_path=str(tmp_path / "dsm.tif"),
        params=params,
    )

    assert result["model"] == "streamline_back_trajectory"
    assert "source_band" in result
    assert "trace_geojson" in result

    # Wind FROM 270° (westerly) → back-trajectory steps west → lower longitude
    assert result["source_band"]["far"]["lon"] < sensor_lon
    assert result["source_band"]["near"]["lon"] < sensor_lon
    assert result["distance_band_m"]["min"] < result["distance_band_m"]["max"]


def test_run_inverse_streamline_sensor_outside_domain(tmp_path):
    crs_epsg = 32632
    west, south, east, north = 500000.0, 5440000.0, 501000.0, 5441000.0

    _write_uniform(str(tmp_path / "vel.tif"), 5.0,   crs_epsg, west, south, east, north)
    _write_uniform(str(tmp_path / "ang.tif"), 270.0, crs_epsg, west, south, east, north)
    _write_uniform(str(tmp_path / "dsm.tif"), 500.0, crs_epsg, west, south, east, north)

    params = InverseParams(force_model_crs=f"EPSG:{crs_epsg}")

    with pytest.raises(ValueError, match="outside WindNinja domain"):
        run_inverse_streamline(
            sensor_lat=0.0, sensor_lon=0.0,  # far outside domain
            pm25_obs=100.0,
            vel_path=str(tmp_path / "vel.tif"),
            ang_path=str(tmp_path / "ang.tif"),
            dsm_path=str(tmp_path / "dsm.tif"),
            params=params,
        )
