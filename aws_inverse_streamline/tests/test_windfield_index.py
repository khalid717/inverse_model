import json
import pytest

from app.windfield_index import WindfieldIndex, match_windfield, _circ_dist


# ---------------------------------------------------------------------------
# _circ_dist
# ---------------------------------------------------------------------------

def test_circ_dist_identical():
    assert _circ_dist(90.0, 90.0) == 0.0


def test_circ_dist_opposite():
    assert _circ_dist(0.0, 180.0) == 180.0


def test_circ_dist_wraparound():
    assert _circ_dist(350.0, 10.0) == 20.0


def test_circ_dist_symmetric():
    assert _circ_dist(10.0, 350.0) == _circ_dist(350.0, 10.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(wspd, dir_deg, domain=None):
    return {
        "wspd_ms": float(wspd),
        "dir_deg": int(dir_deg),
        "vel_ref": "vel.asc",
        "ang_ref": "ang.asc",
        "dsm_ref": "dsm.tif",
        "domain_id": domain,
    }


def _idx(*args):
    return WindfieldIndex(records=list(args))


# ---------------------------------------------------------------------------
# match_windfield
# ---------------------------------------------------------------------------

def test_exact_bin_match():
    idx = _idx(_rec(5.0, 270), _rec(3.0, 90))
    r = match_windfield(idx, 5.0, 270.0)
    assert r["match_type"] == "exact_bin"
    assert r["wspd_ms"] == 5.0
    assert r["dir_deg"] == 270


def test_nearest_neighbor_chosen_when_no_exact():
    idx = _idx(_rec(5.0, 270), _rec(3.0, 90))
    r = match_windfield(idx, 4.8, 265.0)
    assert r["match_type"] == "nearest_neighbor"
    assert r["wspd_ms"] == 5.0


def test_nearest_neighbor_prefers_closer_direction():
    idx = _idx(_rec(5.0, 270), _rec(5.0, 180))
    r = match_windfield(idx, 5.0, 260.0)
    assert r["match_type"] == "nearest_neighbor"
    assert r["dir_deg"] == 270


def test_domain_hint_restricts_candidates():
    idx = _idx(_rec(5.0, 270, domain="alps"), _rec(5.0, 270, domain="valley"))
    r = match_windfield(idx, 5.0, 270.0, domain_hint="valley")
    assert r["domain_id"] == "valley"
    assert r["match_type"] == "exact_bin"


def test_domain_hint_ignored_when_no_match():
    idx = _idx(_rec(5.0, 270, domain="alps"))
    r = match_windfield(idx, 5.0, 270.0, domain_hint="unknown_domain")
    assert r["domain_id"] == "alps"


def test_empty_index_raises():
    with pytest.raises(RuntimeError):
        match_windfield(WindfieldIndex(records=[]), 5.0, 270.0)


def test_dir_360_normalised_to_0():
    idx = _idx(_rec(5.0, 0))
    r = match_windfield(idx, 5.0, 360.0)
    assert r["match_type"] == "exact_bin"


# ---------------------------------------------------------------------------
# WindfieldIndex.from_jsonl
# ---------------------------------------------------------------------------

def test_from_jsonl_vel_s3_key():
    line = json.dumps({
        "wspd_ms": 5.0, "dir_deg": 270,
        "vel_s3_key": "v.asc", "ang_s3_key": "a.asc", "dsm_s3_key": "d.tif",
    })
    idx = WindfieldIndex.from_jsonl(line)
    assert idx.records[0]["vel_ref"] == "v.asc"


def test_from_jsonl_vel_ref_key():
    line = json.dumps({
        "wspd_ms": 3.0, "dir_deg": 90,
        "vel_ref": "v2.asc", "ang_ref": "a2.asc", "dsm_ref": "d2.tif",
    })
    idx = WindfieldIndex.from_jsonl(line)
    assert idx.records[0]["vel_ref"] == "v2.asc"


def test_from_jsonl_skips_blank_lines():
    lines = "\n".join([
        json.dumps({"wspd_ms": 5.0, "dir_deg": 270,
                    "vel_ref": "v.asc", "ang_ref": "a.asc", "dsm_ref": "d.tif"}),
        "",
        json.dumps({"wspd_ms": 3.0, "dir_deg": 90,
                    "vel_ref": "v2.asc", "ang_ref": "a2.asc", "dsm_ref": "d2.tif"}),
    ])
    idx = WindfieldIndex.from_jsonl(lines)
    assert len(idx.records) == 2


def test_from_jsonl_dir_normalised_mod360():
    line = json.dumps({
        "wspd_ms": 5.0, "dir_deg": 450,
        "vel_ref": "v.asc", "ang_ref": "a.asc", "dsm_ref": "d.tif",
    })
    idx = WindfieldIndex.from_jsonl(line)
    assert idx.records[0]["dir_deg"] == 90
