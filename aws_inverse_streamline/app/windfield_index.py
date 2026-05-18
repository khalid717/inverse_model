from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


def _circ_dist(a: float, b: float) -> float:
    """Circular distance on 0..360 degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


@dataclass
class WindfieldIndex:
    """
    Lightweight in-memory index.
    Each record must include:
      - wspd_ms (float)
      - dir_deg (int)
      - vel_ref (str)  (S3 key or local path)
      - ang_ref (str)
      - dsm_ref (str)
      - domain_id (str, optional)
    """
    records: List[Dict[str, Any]]

    @staticmethod
    def from_jsonl(txt: str) -> "WindfieldIndex":
        recs = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            recs.append(__import__("json").loads(line))
        # Normalize required fields
        norm = []
        for r in recs:
            norm.append({
                "domain_id": r.get("domain_id"),
                "wspd_ms": float(r["wspd_ms"]),
                "dir_deg": int(r["dir_deg"]) % 360,
                "vel_ref": r.get("vel_s3_key") or r.get("vel_ref") or r.get("vel_local_path"),
                "ang_ref": r.get("ang_s3_key") or r.get("ang_ref") or r.get("ang_local_path"),
                "dsm_ref": r.get("dsm_s3_key") or r.get("dsm_ref") or r.get("dsm_local_path"),
            })
        return WindfieldIndex(records=norm)


def match_windfield(index: WindfieldIndex,
                    wspd_obs: float,
                    wdir_from_obs: float,
                    *,
                    domain_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Finds the best matching record for observed wind.

    Matching strategy:
    1) Try exact bin match after rounding:
       - wspd rounded to 0.1 m/s
       - dir rounded to nearest integer degree
    2) If not found, choose minimal weighted distance:
       - speed distance in m/s
       - circular direction distance in degrees

    Returns a record dict with match_type.
    """
    wspd_bin = round(wspd_obs * 10.0) / 10.0
    dir_bin = int(round(wdir_from_obs)) % 360

    candidates = index.records
    if domain_hint:
        dom = domain_hint.lower()
        dom_cand = [r for r in candidates if (r.get("domain_id") or "").lower() == dom]
        if dom_cand:
            candidates = dom_cand

    # Exact match first
    for r in candidates:
        if abs(r["wspd_ms"] - wspd_bin) < 1e-9 and int(r["dir_deg"]) == dir_bin:
            out = dict(r)
            out["match_type"] = "exact_bin"
            return out

    # Nearest neighbor fallback
    best = None
    best_score = float("inf")
    for r in candidates:
        ds = abs(r["wspd_ms"] - wspd_obs)              # m/s
        dd = _circ_dist(float(r["dir_deg"]), wdir_from_obs)  # deg
        score = ds * 5.0 + dd * 1.0  # heuristic: 1 m/s ~ 5 degrees
        if score < best_score:
            best_score = score
            best = r

    if best is None:
        raise RuntimeError("Windfield index empty or no candidates.")
    out = dict(best)
    out["match_type"] = "nearest_neighbor"
    return out
