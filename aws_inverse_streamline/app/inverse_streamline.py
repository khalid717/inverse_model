from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import rasterio
from pyproj import Transformer


# ---------------------------------------------------------------------------
# Pasquill-Gifford power-law dispersion coefficients
# σy(d) = a_y * d^b_y  |  σz(d) = a_z * d^b_z   (d in m → σ in m)
# ---------------------------------------------------------------------------
_PG_SIGMA: Dict[str, Tuple[float, float, float, float]] = {
    #       a_y    b_y    a_z    b_z
    "A": (0.22,  0.900, 0.20,  0.950),  # very unstable
    "B": (0.16,  0.900, 0.12,  0.920),  # unstable
    "C": (0.11,  0.900, 0.08,  0.900),  # slightly unstable
    "D": (0.08,  0.900, 0.06,  0.870),  # neutral (default)
    "E": (0.06,  0.900, 0.03,  0.840),  # slightly stable
    "F": (0.04,  0.900, 0.016, 0.810),  # stable
}


@dataclass(frozen=True)
class InverseParams:
    """Parameters for the inverse model (Gaussian plume + streamline back-trajectory)."""

    # Gaussian plume: emission rate reference and uncertainty range
    # Q_ref calibrated from field measurement: 7695 µg/m³ at source corresponds
    # to ~7695 µg/s emission (small wood fire, ~100 g/min fuel consumption).
    Q_ref: float = 7_695.0         # µg/s — reference source emission rate
    Q_min_factor: float = 0.1      # Q_min = Q_ref * Q_min_factor (smoldering / weak)
    Q_max_factor: float = 10.0     # Q_max = Q_ref * Q_max_factor (large active fire)

    # Pasquill-Gifford stability class ("A"–"F")
    stability_class: str = "D"

    # Distance band clamps
    d_min_global_m: float = 2.0
    d_max_global_m: float = 2000.0

    # Streamline integration
    step_length_m: float = 2.0
    min_wind_speed_ms: float = 0.1

    # If you know your model CRS, you can force it; else we read from DSM.
    force_model_crs: Optional[str] = None  # e.g. "EPSG:25832"

    @staticmethod
    def from_env() -> "InverseParams":
        def f(name, default):
            v = os.getenv(name)
            return default if v is None else float(v)
        def s(name, default):
            v = os.getenv(name)
            return default if v is None else v

        return InverseParams(
            Q_ref=f("Q_REF", 7_695.0),
            Q_min_factor=f("Q_MIN_FACTOR", 0.1),
            Q_max_factor=f("Q_MAX_FACTOR", 10.0),
            stability_class=s("STABILITY_CLASS", "D"),
            d_min_global_m=f("D_MIN_GLOBAL_M", 2.0),
            d_max_global_m=f("D_MAX_GLOBAL_M", 2000.0),
            step_length_m=f("STEP_LENGTH_M", 2.0),
            min_wind_speed_ms=f("MIN_WIND_SPEED_MS", 0.1),
            force_model_crs=s("FORCE_MODEL_CRS", None),
        )


def inside(bounds, x, y) -> bool:
    return (bounds.left <= x <= bounds.right) and (bounds.bottom <= y <= bounds.top)


def sample_wind(vel_ds, ang_ds, x, y) -> Tuple[float, float]:
    (vel_val,) = list(vel_ds.sample([(x, y)]))[0]
    (ang_val,) = list(ang_ds.sample([(x, y)]))[0]
    return float(vel_val), float(ang_val)


def sigma_y(d_m: float, stability: str = "D") -> float:
    """Lateral dispersion coefficient σy (m) at downwind distance d_m (m)."""
    a, b, _, _ = _PG_SIGMA[stability.upper()]
    return a * (d_m ** b)


def sigma_z(d_m: float, stability: str = "D") -> float:
    """Vertical dispersion coefficient σz (m) at downwind distance d_m (m)."""
    _, _, a, b = _PG_SIGMA[stability.upper()]
    return a * (d_m ** b)


def gaussian_centerline_conc(Q: float, u: float, d_m: float, stability: str = "D") -> float:
    """Gaussian plume centerline ground-level concentration (µg/m³).

    C(d) = Q / (π × σy(d) × σz(d) × u)

    Assumes: ground-level source, ground-level receptor, centerline, no reflection.
    """
    sy = sigma_y(d_m, stability)
    sz = sigma_z(d_m, stability)
    return Q / (np.pi * sy * sz * u)


def _bisect(f, lo: float, hi: float, tol: float = 0.1, max_iter: int = 60) -> float:
    """Bisection root-finder: f(lo) > 0, f(hi) < 0 assumed on entry."""
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break
    return 0.5 * (lo + hi)


def gaussian_distance_band(pm25_obs: float, u: float,
                           Q_min: float, Q_max: float,
                           stability: str,
                           d_min_global: float, d_max_global: float) -> Tuple[float, float]:
    """Invert Gaussian plume equation to find source distance band (m).

    Concentration decreases monotonically with distance → one solution per Q.
    Q_min (weak source) → closer distance; Q_max (strong source) → farther distance.
    """
    if pm25_obs <= 0:
        raise ValueError("Observed concentration must be > 0.")
    if u <= 0:
        raise ValueError("Wind speed must be > 0.")

    def _solve_d(Q: float) -> float:
        def residual(d):
            return gaussian_centerline_conc(Q, u, d, stability) - pm25_obs

        c_near = gaussian_centerline_conc(Q, u, d_min_global, stability)
        c_far  = gaussian_centerline_conc(Q, u, d_max_global, stability)

        if pm25_obs >= c_near:
            return d_min_global  # observation is very high → source very close
        if pm25_obs <= c_far:
            return d_max_global  # observation is low → source at or beyond max range
        return _bisect(residual, d_min_global, d_max_global)

    d_for_Q_min = _solve_d(Q_min)
    d_for_Q_max = _solve_d(Q_max)

    d_min_band = float(max(min(d_for_Q_min, d_for_Q_max), d_min_global))
    d_max_band = float(min(max(d_for_Q_min, d_for_Q_max), d_max_global))
    return d_min_band, d_max_band

def trace_curved_back_trajectory(sensor_x: float, sensor_y: float,
                                 vel_ds, ang_ds,
                                 d_max_band: float,
                                 step_length: float,
                                 min_wind_speed: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [sensor_x]
    ys = [sensor_y]
    ds = [0.0]
    b = vel_ds.bounds

    while ds[-1] < d_max_band:
        x, y = xs[-1], ys[-1]
        if not inside(b, x, y):
            break

        vel_val, ang_val = sample_wind(vel_ds, ang_ds, x, y)
        if vel_val < min_wind_speed:
            break

        # Convert direction FROM -> direction TO
        dir_to_deg = (ang_val + 180.0) % 360.0
        theta = np.deg2rad(dir_to_deg)
        u = vel_val * np.sin(theta)  # east
        v = vel_val * np.cos(theta)  # north
        speed = float(np.hypot(u, v))
        if speed < min_wind_speed:
            break

        u_hat, v_hat = u / speed, v / speed

        # Upstream step (opposite to flow direction)
        x_new = x - u_hat * step_length
        y_new = y - v_hat * step_length
        d_new = ds[-1] + step_length

        xs.append(x_new)
        ys.append(y_new)
        ds.append(d_new)

    return np.array(xs), np.array(ys), np.array(ds)


def interpolate_point_linear(xs: np.ndarray, ys: np.ndarray, ds: np.ndarray, target_d: float) -> Tuple[float, float]:
    """
    Linear interpolation along ds, so results are smooth.
    """
    if target_d <= ds[0]:
        return float(xs[0]), float(ys[0])
    if target_d >= ds[-1]:
        return float(xs[-1]), float(ys[-1])

    # Find segment where ds[i] <= target < ds[i+1]
    i = int(np.searchsorted(ds, target_d) - 1)
    i = max(0, min(i, len(ds) - 2))
    d0, d1 = ds[i], ds[i + 1]
    t = 0.0 if d1 == d0 else (target_d - d0) / (d1 - d0)
    x = xs[i] + t * (xs[i + 1] - xs[i])
    y = ys[i] + t * (ys[i + 1] - ys[i])
    return float(x), float(y)


def _validate_rasters(vel_ds, ang_ds, dsm_ds) -> None:
    """Check that rasters are mutually compatible before any sampling."""
    # ASC files (WindNinja output) carry no embedded CRS — FORCE_MODEL_CRS handles projection.
    # Only enforce CRS consistency when vel actually has a CRS.
    if vel_ds.crs is not None:
        for name, ds in [("ang", ang_ds), ("dsm", dsm_ds)]:
            if str(ds.crs) != str(vel_ds.crs):
                raise ValueError(f"{name} CRS {ds.crs} != vel CRS {vel_ds.crs}")

    # vel and ang are from the same WindNinja run and must align exactly.
    if vel_ds.shape != ang_ds.shape or vel_ds.transform != ang_ds.transform:
        raise ValueError(
            f"ang grid (shape={ang_ds.shape}, transform={ang_ds.transform}) "
            f"does not match vel grid (shape={vel_ds.shape}, transform={vel_ds.transform})"
        )


def _to_geojson_line(lons: List[float], lats: List[float]) -> Dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "back_trajectory"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[lo, la] for lo, la in zip(lons, lats)]
            }
        }]
    }


def run_inverse_streamline(*,
                           sensor_lat: float,
                           sensor_lon: float,
                           pm25_obs: float,
                           vel_path: str,
                           ang_path: str,
                           dsm_path: str,
                           params: InverseParams) -> Dict[str, Any]:
    """
    Main callable that runs the streamline back-trajectory inverse model.
    Returns a machine-readable dict with min/mid/max source band and optional trace GeoJSON.
    """
    with rasterio.open(dsm_path) as dsm_ds, rasterio.open(vel_path) as vel_ds, rasterio.open(ang_path) as ang_ds:
        _validate_rasters(vel_ds, ang_ds, dsm_ds)
        model_crs = params.force_model_crs or str(dsm_ds.crs)
        if model_crs is None or model_crs == "None":
            raise RuntimeError("DSM has no CRS; set FORCE_MODEL_CRS env var (e.g. EPSG:25832).")

        # Project sensor WGS84 -> model CRS
        transformer = Transformer.from_crs("EPSG:4326", model_crs, always_xy=True)
        sensor_x, sensor_y = transformer.transform(sensor_lon, sensor_lat)

        if not inside(vel_ds.bounds, sensor_x, sensor_y):
            raise ValueError("Sensor outside WindNinja domain bounds.")

        # Sample at sensor (for reporting)
        vel_val, ang_val = sample_wind(vel_ds, ang_ds, sensor_x, sensor_y)
        (z_val,) = list(dsm_ds.sample([(sensor_x, sensor_y)]))[0]

        # Gaussian plume inversion: find source distance band
        u = max(float(vel_val), params.min_wind_speed_ms)
        Q_min = params.Q_ref * params.Q_min_factor
        Q_max = params.Q_ref * params.Q_max_factor

        d_min_band, d_max_band = gaussian_distance_band(
            pm25_obs, u, Q_min, Q_max,
            params.stability_class,
            params.d_min_global_m, params.d_max_global_m,
        )
        d_mid_band = 0.5 * (d_min_band + d_max_band)

        # Trace streamline back
        xs, ys, ds = trace_curved_back_trajectory(
            sensor_x, sensor_y, vel_ds, ang_ds,
            d_max_band=d_max_band,
            step_length=params.step_length_m,
            min_wind_speed=params.min_wind_speed_ms,
        )
        if len(ds) < 2:
            raise RuntimeError("Back-trajectory too short (calm wind or domain edge).")

        # Interpolate band points
        src_min_x, src_min_y = interpolate_point_linear(xs, ys, ds, d_min_band)
        src_mid_x, src_mid_y = interpolate_point_linear(xs, ys, ds, d_mid_band)
        src_max_x, src_max_y = interpolate_point_linear(xs, ys, ds, d_max_band)

        # Convert to lat/lon
        back = Transformer.from_crs(model_crs, "EPSG:4326", always_xy=True)
        src_min_lon, src_min_lat = back.transform(src_min_x, src_min_y)
        src_mid_lon, src_mid_lat = back.transform(src_mid_x, src_mid_y)
        src_max_lon, src_max_lat = back.transform(src_max_x, src_max_y)

        # Also convert entire trace for GIS display (optional)
        lons, lats = [], []
        for x, y in zip(xs.tolist(), ys.tolist()):
            lo, la = back.transform(x, y)
            lons.append(float(lo)); lats.append(float(la))

        # Report local flow direction at sensor (TO direction)
        dir_to_deg = (ang_val + 180.0) % 360.0

        return {
            "model": "streamline_back_trajectory",
            "sensor_elevation_m": float(z_val),
            "distance_band_m": {
                "min": float(d_min_band),
                "mid": float(d_mid_band),
                "max": float(d_max_band),
                "trace_length": float(ds[-1]),
                "n_steps": int(len(ds)),
                "step_length": float(params.step_length_m),
            },
            "source_band": {
                "near": {"lat": float(src_min_lat), "lon": float(src_min_lon)},
                "mid": {"lat": float(src_mid_lat), "lon": float(src_mid_lon)},
                "far": {"lat": float(src_max_lat), "lon": float(src_max_lon)},
            },
            "trace_geojson": _to_geojson_line(lons, lats),
        }        