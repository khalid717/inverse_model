from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import rasterio
from pyproj import Transformer


@dataclass(frozen=True)
class InverseParams:
    """
    Parameters for the inverse model.
    Defaults match your notebook prototype, but can be overridden via env vars.
    """
    # Empirical decay calibration: C(d) = C0 * exp(-k d)
    C0_ref: float = 1000.0
    C_ref: float = 45.0
    d_ref_m: float = 15.0

    # Fire scale uncertainty around C0_ref
    C0_min_factor: float = 0.5
    C0_max_factor: float = 2.0

    # Distance band clamps
    d_min_global_m: float = 2.0
    d_max_global_m: float = 1000.0

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
            C0_ref=f("C0_REF", 1000.0),
            C_ref=f("C_REF", 45.0),
            d_ref_m=f("D_REF_M", 15.0),
            C0_min_factor=f("C0_MIN_FACTOR", 0.5),
            C0_max_factor=f("C0_MAX_FACTOR", 2.0),
            d_min_global_m=f("D_MIN_GLOBAL_M", 2.0),
            d_max_global_m=f("D_MAX_GLOBAL_M", 1000.0),
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


def decay_k(C0_ref: float, C_ref: float, d_ref_m: float) -> float:
    if not (0.0 < C_ref < C0_ref):
        raise ValueError("Need 0 < C_ref < C0_ref to calibrate k.")
    return float(np.log(C0_ref / C_ref) / d_ref_m)


def distance_band_from_obs(C_obs: float, k: float,
                           C0_min: float, C0_max: float,
                           d_min_global: float, d_max_global: float) -> Tuple[float, float]:
    if C_obs <= 0:
        raise ValueError("Observed concentration must be > 0.")

    # If observation is extremely high (near-source), clamp it slightly below C0_max
    C_obs_eff = float(np.clip(C_obs, 1e-6, C0_max * 0.99))

    d1 = (1.0 / k) * np.log(C0_min / C_obs_eff)
    d2 = (1.0 / k) * np.log(C0_max / C_obs_eff)

    d_candidates = [d for d in (d1, d2) if d > 0]

    if not d_candidates:
        # If both are <=0, the observation is at/above plausible C0 range.
        # Operationally: treat as "very close", but keep a non-zero band.
        return float(d_min_global), float(min(5.0 * d_min_global, d_max_global))

    d_min_band = max(min(d_candidates), d_min_global)
    d_max_band = min(max(d_candidates), d_max_global)
    if d_min_band > d_max_band:
        d_min_band, d_max_band = d_min_global, d_max_global

    return float(d_min_band), float(d_max_band)

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

        # Calibrate decay & band
        k = decay_k(params.C0_ref, params.C_ref, params.d_ref_m)
        C0_min = params.C0_min_factor * params.C0_ref
        C0_max = params.C0_max_factor * params.C0_ref

        d_min_band, d_max_band = distance_band_from_obs(
            pm25_obs, k, C0_min, C0_max, params.d_min_global_m, params.d_max_global_m
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
        }        