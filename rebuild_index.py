"""Rebuild windfield_index.jsonl by scanning the local runs directory."""
import json
import os

LIBRARY_ROOT = r"D:\ClaudeCodeTest\windninjalibrary"
RUNS_ROOT    = os.path.join(LIBRARY_ROOT, "runs", "Feuchtwangen_1x1")
DSM_REF      = "domains/Feuchtwangen_1x1/dsm.tif"
OUTPUT       = os.path.join(LIBRARY_ROOT, "index", "windfield_index.jsonl")
DOMAIN_ID    = "Feuchtwangen_1x1"

# Wind speed range to include
WS_MIN, WS_MAX = 2.0, 8.0

records = []
missing = []

for ws_dir in sorted(os.listdir(RUNS_ROOT)):
    if not ws_dir.startswith("ws_"):
        continue
    wspd = float(ws_dir.replace("ws_", ""))
    if wspd < WS_MIN or wspd > WS_MAX:
        continue

    ws_path = os.path.join(RUNS_ROOT, ws_dir)
    for dir_dir in sorted(os.listdir(ws_path)):
        if not dir_dir.startswith("dir_"):
            continue
        dir_deg = int(dir_dir.replace("dir_", ""))
        dir_path = os.path.join(ws_path, dir_dir)

        files = os.listdir(dir_path)
        vel_file = next((f for f in files if f.endswith("_vel.asc")), None)
        ang_file = next((f for f in files if f.endswith("_ang.asc")), None)

        if not vel_file or not ang_file:
            missing.append(f"{ws_dir}/{dir_dir}")
            continue

        vel_ref = f"runs/Feuchtwangen_1x1/{ws_dir}/{dir_dir}/{vel_file}"
        ang_ref = f"runs/Feuchtwangen_1x1/{ws_dir}/{dir_dir}/{ang_file}"

        records.append({
            "domain_id": DOMAIN_ID,
            "wspd_ms":   wspd,
            "dir_deg":   dir_deg,
            "vel_s3_key": vel_ref,
            "ang_s3_key": ang_ref,
            "dsm_s3_key": DSM_REF,
        })

with open(OUTPUT, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"Written {len(records)} records to {OUTPUT}")
speeds = sorted(set(r["wspd_ms"] for r in records))
print(f"Wind speeds: {speeds[0]} - {speeds[-1]} m/s ({len(speeds)} bins)")
print(f"Directions : 0-359° ({len(set(r['dir_deg'] for r in records))} unique)")
if missing:
    print(f"Missing files ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
