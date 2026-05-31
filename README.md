<div align="center">

<img src="assets/logo.svg" width="92" alt="Eye in frame logo" />

# Venue Analytics MVP

### Real-time retail analytics that never identifies anyone.

A FastAPI service that turns CCTV footage into demographics, behavior patterns, and engagement scores, on-prem or at the edge. Faces become 512-dim vectors, not photos. Visitors become hashed IDs, not identities.

<br/>

![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-1A1A1A)
![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-1A1A1A)
![Railway](https://img.shields.io/badge/deploy-Railway-0B0D0E?logo=railway&logoColor=white)

</div>

---

## What it does

Point it at a CCTV stream, a YouTube link, or an uploaded clip. It runs three on-device ML stages, writes counts and aggregates to SQLite or Postgres, and serves the result as JSON or as interactive HTML dashboards.

```
Video ──► YOLO11 detection + BoT-SORT tracking
       ──► InsightFace SCRFD face detection
       ──► InsightFace buffalo_l: 512-dim embedding + age + gender
       ──► aggregated demographics, dwell time, return visits
```

**No raw video is stored. No PII leaves the box.** Every "person" downstream is a hashed track ID plus a face vector you can search against, but cannot reverse into an image.

## Why privacy-preserving

- **Faces are stored as 512-dim ArcFace embeddings.** You can recognise a returning visitor without ever keeping their photo.
- **Visitor IDs are salted hashes.** No phone, no name, no government ID, nothing personally identifying.
- **Edge-first.** The service is built to run inside the venue, behind your firewall. Cloud is optional.
- Designed to fit **GCC PDPL and GDPR** expectations by construction, not by policy.

## What's inside

- **45 JSON endpoints** across 13 routers: ingest, demographics, behavior, re-identification, engagement, alerts, exports.
- **8 HTML dashboards** for live KPIs, heat maps, dwell-time histograms, returning-visitor curves.
- **One ML pipeline** (`app/pipelines/`) with a video processing queue, async by default.
- **One SQL schema** (7 tables) that bootstraps on first run via SQLAlchemy.

## Quickstart

```bash
git clone https://github.com/sunnybhara/cctv-analytics-mvp.git
cd cctv-analytics-mvp
pip install -r requirements.txt
python main.py
# → http://localhost:8000  (dashboards)
# → http://localhost:8000/docs  (OpenAPI explorer)
```

## Deploy to Railway

```bash
npm install -g @railway/cli
railway login
railway init
railway add --database postgres
railway up
railway domain
```

Health check is wired to `/health`. Startup is `uvicorn main:app --host 0.0.0.0 --port $PORT` (see `railway.toml`).

## API at a glance

| Group | Endpoints |
|---|---|
| Ingest | `POST /ingest/video`, `POST /ingest/youtube`, `POST /ingest/stream` |
| People | `GET /people`, `GET /people/{id}`, `GET /people/{id}/visits` |
| Demographics | `GET /demographics/summary`, `/by-hour`, `/age-gender` |
| Behavior | `GET /behavior/dwell`, `/zones`, `/heatmap` |
| Re-identification | `POST /reid/search`, `GET /reid/returns` |
| Dashboards | `GET /dashboard/*` (live HTML pages) |
| Ops | `GET /health`, `GET /metrics`, `GET /version` |

Full spec at `/docs` once running.

## Status

MVP. Single-camera, single-venue, single-node. Production hardening (multi-camera fan-out, role-based access, retention policies) is the next phase. Architecture notes in [`ARCHITECTURE.md`](ARCHITECTURE.md), product framing in [`PRODUCT_VISION.md`](PRODUCT_VISION.md).

---

<div align="center">
<sub>Built for venues that want the analytics without the surveillance.</sub>
</div>
