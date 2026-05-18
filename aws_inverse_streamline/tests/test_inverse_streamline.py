import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pyproj import Transformer

from app.inverse_streamline import (
    sigma_y,
    sigma_z,
    gaussian_centerline_conc,
    gaussian_distance_band,
    interpolate_point_linear,
    inside,
    _validate_rasters,
    InverseParams,
    run_inverse_streamline,
)


# ---------------------------------------------------------------------------
# sigma_y / sigma_z
# ---------------------------------------------------------------------------

def test_sigma_y_increases_with_distance():
    assert sigma_y(100.0) < sigma_y(500.0)


def test_sigma_z_increases_with_distance():
    assert sigma_z(100.0) < sigma_z(500.0)


def test_sigma_y_class_D_100m():
    # class D: a=0.08, b=0.9  → σy = 0.08 * 100^0.9 ≈ 5.05 m
    val = sigma_y(100.0, "D")
    assert 4.0 < val < 7.0


def test_sigma_z_class_D_100m():
    # class D: a=0.06, b=0.87 → σz = 0.06 * 100^0.87 ≈ 3.45 m
    val = sigma_z(100.0, "D")
    assert 2.0 < val < 5.0


def test_unstable_class_wider_than_neutral():
    # Class A (unstable) should give larger σy than class D at same distance
    assert sigma_y(200.0, "A") > sigma_y(200.0, "D")
    assert sigma_z(200.0, "A") > sigma_z(200.0, "D")


def test_sigma_unknown_class_raises():
    with pytest.raises(KeyError):
        sigma_y(100.0, "Z")


# ---------------------------------------------------------------------------
# gaussian_centerline_conc
# ---------------------------------------------------------------------------

def test_conc_decreases_with_distance():
    assert gaussian_centerline_conc(100_000.0, 5.0, 100.0) > gaussian_centerline_conc(100_000.0, 5.0, 500.0)


def test_conc_increases_with_emission_rate():
    assert gaussian_centerline_conc(200_000.0, 5.0, 200.0) > gaussian_centerline_conc(100_000.0, 5.0, 200.0)


def test_conc_decreases_with_wind_speed():
    assert gaussian_centerline_conc(100_000.0, 3.0, 200.0) > gaussian_centerline_conc(100_000.0, 8.0, 200.0)


# ---------------------------------------------------------------------------
# gaussian_distance_band
# ---------------------------------------------------------------------------

def test_gaussian_distance_band_typical():
    d_min, d_max = gaussian_distance_band(
        pm25_obs=25.0, u=5.0,
        Q_min=10_000.0, Q_max=1_000_000.0,
        stability="D",
        d_min_global=2.0, d_max_global=2000.0,
    )
    assert 0 < d_min < d_max
    assert d_min >= 2.0
    assert d_max <= 2000.0


def test_gaussian_distance_band_Q_max_gives_farther_source():
    # A stronger emitter produces the same observed concentration farther away
    d_min, d_max = gaussian_distance_band(
        pm25_obs=50.0, u=5.0,
        Q_min=50_000.0, Q_max=500_000.0,
        stability="D",
        d_min_global=2.0, d_max_global=2000.0,
    )
    assert d_max > d_min


def test_gaussian_distance_band_high_obs_clamps_to_near():
    # Very high PM2.5 → source must be very close (d_min ≈ d_min_global)
    d_min, d_max = gaussian_distance_band(
        pm25_obs=500_000.0, u=5.0,
        Q_min=10_000.0, Q_max=1_000_000.0,
        stability="D",
        d_min_global=2.0, d_max_global=2000.0,
    )
    assert d_min == pytest.approx(2.0)


def test_gaussian_distance_band_low_obs_clamps_to_far():
    # Very low PM2.5 → source is at the far limit
    d_min, d_max = gaussian_distance_band(
        pm25_obs=0.001, u=5.0,
        Q_min=10_000.0, Q_max=1_000_000.0,
        stability="D",
        d_min_global=2.0, d_max_global=2000.0,
    )
    assert d_max == pytest.approx(2000.0)


def test_gaussian_distance_band_zero_obs_raises():
    with pytest.raises(ValueError):
        gaussian_distance_band(0.0, 5.0, 10_000.0, 1_000_000.0, "D", 2.0, 2000.0)


def test_gaussian_distance_band_zero_wind_raises():
    with pytest.raises(ValueError):
        gaussian_distance_band(25.0, 0.0, 10_000.0, 1_000_000.0, "D", 2.0, 2000.0)


def test_gaussian_distance_band_stability_affects_result():
    # Unstable (class A) → wider σ → plume disperses faster → same concentration
    # reached at a CLOSER distance than for stable class F (narrow σ, less dilution)
    d_min_A, d_max_A = gaussian_distance_band(50.0, 5.0, 100_000.0, 100_000.0, "A", 2.0, 2000.0)
    d_min_F, d_max_F = gaussian_distance_band(50.0, 5.0, 100_000.0, 100_000.0, "F", 2.0, 2000.0)
    assert d_min_A < d_min_F


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
    west, south, east, north = 500000.0, 5440000.0, 503000.0, 5443000.0

    _write_uniform(str(tmp_path / "vel.tif"), 5.0,   crs_epsg, west, south, east, north)
    _write_uniform(str(tmp_path / "ang.tif"), 270.0, crs_epsg, west, south, east, north)
    _write_uniform(str(tmp_path / "dsm.tif"), 500.0, crs_epsg, west, south, east, north)

    back = Transformer.from_crs(crs_epsg, 4326, always_xy=True)
    sensor_lon, sensor_lat = back.transform(
        (west + east) / 2, (south + north) / 2
    )

    params = InverseParams(
        Q_ref=100_000.0,
        Q_min_factor=0.1,
        Q_max_factor=10.0,
        stability_class="D",
        d_min_global_m=2.0,
        d_max_global_m=2000.0,
        step_length_m=5.0,
        min_wind_speed_ms=0.1,
        force_model_crs=f"EPSG:{crs_epsg}",
    )

    result = run_inverse_streamline(
        sensor_lat=sensor_lat,
        sensor_lon=sensor_lon,
        pm25_obs=25.0,
        vel_path=str(tmp_path / "vel.tif"),
        ang_path=str(tmp_path / "ang.tif"),
        dsm_path=str(tmp_path / "dsm.tif"),
        params=params,
    )

    assert result["model"] == "streamline_back_trajectory"
    assert "source_band" in result
    assert "trace_geojson" in result

    # Wind FROM 270° (westerly) → back-trajectory goes west → lower longitude
    assert result["source_band"]["far"]["lon"] < sensor_lon
    assert result["source_band"]["near"]["lon"] < sensor_lon
    assert result["distance_band_m"]["min"] < result["distance_band_m"]["max"]

    # DSM elevation at sensor should be ~500m
    assert result["sensor_elevation_m"] == pytest.approx(500.0, abs=1.0)

    # trace_geojson must be a valid GeoJSON FeatureCollection with a LineString
    gj = result["trace_geojson"]
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 1
    geom = gj["features"][0]["geometry"]
    assert geom["type"] == "LineString"
    coords = geom["coordinates"]
    assert len(coords) >= 2
    for lon, lat in coords:
        assert -180 <= lon <= 180
        assert -90 <= lat <= 90


def test_run_inverse_streamline_gaussian_distances_realistic(tmp_path):
    """Gaussian model should place source in a plausible 50–2000m range."""
    crs_epsg = 32632
    west, south, east, north = 500000.0, 5440000.0, 504000.0, 5444000.0

    _write_uniform(str(tmp_path / "vel.tif"), 6.0,   crs_epsg, west, south, east, north)
    _write_uniform(str(tmp_path / "ang.tif"), 235.0, crs_epsg, west, south, east, north)
    _write_uniform(str(tmp_path / "dsm.tif"), 500.0, crs_epsg, west, south, east, north)

    back = Transformer.from_crs(crs_epsg, 4326, always_xy=True)
    sensor_lon, sensor_lat = back.transform((west + east) / 2, (south + north) / 2)

    params = InverseParams(
        Q_ref=100_000.0,
        Q_min_factor=0.1,
        Q_max_factor=10.0,
        stability_class="D",
        d_min_global_m=2.0,
        d_max_global_m=2000.0,
        step_length_m=5.0,
        force_model_crs=f"EPSG:{crs_epsg}",
    )

    result = run_inverse_streamline(
        sensor_lat=sensor_lat,
        sensor_lon=sensor_lon,
        pm25_obs=25.0,
        vel_path=str(tmp_path / "vel.tif"),
        ang_path=str(tmp_path / "ang.tif"),
        dsm_path=str(tmp_path / "dsm.tif"),
        params=params,
    )

    d_min = result["distance_band_m"]["min"]
    d_max = result["distance_band_m"]["max"]
    assert 50.0 < d_min < 2000.0
    assert d_max > d_min


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
