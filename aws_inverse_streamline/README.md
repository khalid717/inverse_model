# AWS Inverse Streamline Runner

This is a refactor of your notebook "Curved back-trajectory inverse test with distance band"
into an automated pipeline:

Alert → nearest metmast wind → match closest pre-sim windfield → streamline inverse → result JSON (+ GeoJSON trace)

## 1) Expected alert JSON

```json
{
  "timestamp_utc": "2026-02-02T10:12:31Z",
  "sensor_id": "SENSOR_04",
  "lat": 49.179924,
  "lon": 10.336225,
  "pm25": 800.0
}
```

## 2) Storage modes

### S3 (AWS)
Set:
- STORE_MODE=s3
- S3_BUCKET=<bucket>
- S3_PREFIX=windninja_library  (default)
- WINDFIELD_INDEX_KEY=index/windfield_index.jsonl  (default)

Windfield index records should provide:
- wspd_ms, dir_deg, vel_s3_key, ang_s3_key, dsm_s3_key

### Local (dev)
Set:
- STORE_MODE=local
- LOCAL_LIBRARY_ROOT=/path/to/library_root

Index expected at:
- <root>/index/windfield_index.jsonl

## 3) Metmast modes

### local_csv (dev)
Set:
- METMAST_MODE=local_csv
- METMAST_TABLE_PATH=metmasts.csv
- METMAST_WIND_PATH=metmast_wind.csv

### dynamodb (AWS)
Set:
- METMAST_MODE=dynamodb
- METMAST_TABLE_DDB=<mast metadata table>
- METMAST_WIND_DDB=<wind time series table>

## 4) Run locally

```bash
python -m app.handler --alert alert.json
```

## 5) Output

- source_band near/mid/far (lat/lon)
- trace_geojson line for GIS/QGIS
- includes the windfield chosen and metmast wind used
