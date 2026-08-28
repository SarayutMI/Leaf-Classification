# Yamwisdom Phase 2 — Leaf Classification AI Service

## Overview

A FastAPI service that classifies cassava/yam leaf characteristics from a single
photo, and maps the result to a named leaf group using a rule base stored in
MySQL and edited from a built-in admin page.

One request does five things:

1. **Detect** the leaf with YOLO11s and crop it to 4:3 without distortion
2. **Slice** the crop into the bands each classifier was trained on
3. **Classify** shape / apex / base / margin with four independent ResNet50V2 heads
4. **Match** the four predicted traits against the rule base to name a group
   (and return its id, so the caller can look up the yam varieties under it)
5. **Log** the call and upload the cropped regions to S3 in the background

The rule base is data, not code: `/admin/` is a small static page where a
non-developer creates and edits rules, and changes take effect within 30
seconds without a redeploy.

---

## Input contract — leaf orientation

**The leaf must be photographed upright with its tip (apex) pointing DOWN and
the stalk (petiole) up.** This is not a preference; it is what the classifiers
were trained on.

`slice_leaf()` cuts the crop at fixed fractions of image height and hands each
band to a different model. Because the apex points down, the apex model reads
the **bottom** band and the base model reads the **top** one:

| band | fraction of height | model | region file |
|---|---|---|---|
| full | 0 – 100% | `R50_Shape` | `Full-leaf.jpg` |
| top | 0 – 30% | `R50_Base` | `Top-leaf.jpg` |
| middle | 30 – 60% | `R50_Margin` | `Middle-leaf.jpg` |
| *(gap)* | 60 – 70% | *(unused — transition zone)* | — |
| bottom | 70 – 100% | `R50_Apex` | `Bottom-leaf.jpg` |

These bounds come from `Etc/Extractimage-Test-Process.py`, the script that
produced the training crops from the `Testset-*-Shadow-output` photos. A leaf
that arrives sideways or flipped gives every head a region it never saw in
training, and the API answers with confident, wrong labels — it cannot detect
that anything is off.

The API does **not** correct orientation, deliberately. Guessing wrong destroys
framing that was already correct, and nothing in these photos settles it: the
petiole — the one botanically reliable marker of the base — is cut off before
the leaf is photographed. Enforce the framing in the client instead.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r Requirement.txt
cp .env.example .env          # then fill in DB + AWS values

# JWT_SECRET is REQUIRED — the app refuses to start without it
python -c "import secrets; print(secrets.token_urlsafe(48))"

python -m src.main

# or with Docker (the override file publishes the host port):
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
pytest tests -q
```

Then open **<http://localhost:8000/admin/>** (dev container: `:15780`) and log in
with an account from the platform's `users` table that has `role = 'admin'` and
`is_active = 1`. `/` redirects there.

`SEED_USERNAME` / `SEED_PASSWORD` are **not** the admin login — they seed the
`classify_user` row that backs `/api/genToken`.

Tables are created and migrated on every startup by `src/seed.py` — there is no
separate migration step and no ORM.

---

## Project structure

```
Leaf-Classification/
├── src/
│   ├── main.py               # FastAPI app, lifespan (seed + model load/warmup), /admin mount
│   ├── config.py             # Settings (pydantic BaseSettings) -> `settings`
│   ├── database.py           # MySQL connection pool, get_conn()
│   ├── seed.py               # CREATE TABLE + schema migrations + first admin user
│   │
│   ├── auth/                 # Feature module: authentication
│   │   ├── router.py         # POST /api/genToken, /api/admin/{login,logout,me}
│   │   ├── schemas.py        # Request models
│   │   ├── service.py        # get_or_create_api_key()
│   │   ├── repository.py     # SQL for classify_user / classify_token / users
│   │   ├── security.py       # bcrypt hashing, API key + JWT bearer tokens
│   │   ├── dependencies.py   # require_api_key (X-API-Key), require_session (Bearer)
│   │   └── exceptions.py
│   │
│   ├── classify/             # Feature module: leaf classification
│   │   ├── router.py         # POST /api/classify
│   │   ├── service.py        # YOLO detect, slice, TFLite/Keras predict
│   │   ├── storage.py        # S3 upload of cropped regions
│   │   └── repository.py     # SQL for classify_api_logs / image_dataset
│   │
│   ├── rules/                # Feature module: the rule base
│   │   ├── router.py         # /api/rules CRUD + /varieties + /test + /vocab
│   │   ├── service.py        # match_group() + 30s rule cache
│   │   ├── schemas.py        # Validation against the configured class lists
│   │   └── repository.py     # SQL for classify_rule_group + varieties
│   │
│   ├── static/               # Admin page (no build step, no CDN)
│   │   ├── index.html        # login + rules CRUD + varieties + scoring test bench
│   │   ├── app.js
│   │   └── style.css         # follows DESIGN.md
│   │
│   └── health/router.py      # GET /health (DB + model readiness), /health/live
│
├── tests/                    # conftest.py + tests/auth, tests/classify, tests/rules
├── scripts/                  # convert_tflite.py, export_openvino.py
├── Model-Leaf/               # YOLO weights (runtime volume)
├── Model_Classification/     # Keras/TFLite classifiers (runtime volume)
├── DESIGN.md                 # Design system for the admin page
├── .env.example
├── Requirement.txt
├── Dockerfile                # base -> test / runtime targets
├── docker-compose.yml        # shared service (no published port)
├── docker-compose.dev.yml    # dev override  — host :15780
├── docker-compose.prod.yml   # prod override — host :16780
├── Jenkinsfile               # Vault -> build -> test -> deploy -> health check
└── entrypoint.sh             # fetches/converts models, then `python -m src.main`
```

---

# System Architecture

```text
Input Image
     │
     ▼
YOLO Leaf Detection
     │
     ▼
4:3 Aspect Crop
(No Resize / No Squeeze)
     │
     ▼
Image Region Extraction
     │
     ├── Full Leaf
     ├── Apex Region
     ├── Base Region
     └── Margin Region
     │
     ▼
Classification Services
     │
     ├── Shape Model
     ├── Apex Model
     ├── Base Model
     └── Margin Model
     │
     ▼
Rule Base Matching  ◄── MySQL (groups + varieties, edited at /admin/)
     │
     ▼
Classification API Response
     │
     ├──► Laravel Backend
     └──► S3 region upload + MySQL log (background)
```

---

# Process 1 — Leaf Detection

## Model

```text
YOLO11s
```

Model file:

```text
yolo11s_leaf.pt
```

Target class:

```text
leaf
```

The detector is responsible for locating the leaf within the image.
Detection dominates request time (~72% at 1600×1200 input) and scales with
`YOLO_IMGSZ` (default 640), not with the uploaded resolution.

---

## Detection Workflow

```text
Input Image
      │
      ▼
YOLO Detection
      │
      ▼
Bounding Box
      │
      ▼
Expand Bounding Box
to 4:3 Aspect Ratio
      │
      ▼
Crop Image
```

---

## Cropping Rules

The crop operation:

* Uses YOLO bounding box
* Maintains original image proportions
* Does NOT stretch image
* Does NOT squeeze image
* Does NOT resize image before cropping

Target ratio:

```text
4 : 3
```

The ratio was measured, not assumed. Portrait (3:4) frames an upright leaf more
tightly and was tried, but scored 96.7% against 4:3's 98.3% on the 241 labelled
photos; the raw bounding box the dataset generator produces came last at 94.6%.
Expanding the box helps the shape head whichever ratio is used. Re-run the
comparison with `scripts/compare_crop_aspect.py`.

Example:

```text
Original BBox
 ┌─────────┐
 │  Leaf   │
 └─────────┘

Expanded Crop Area (4:3)
 ┌─────────────────┐
 │                 │
 │      Leaf       │
 │                 │
 └─────────────────┘
```

---

# Process 2 — Region Extraction

After cropping, the image is sliced into separate regions. Remember the
orientation contract: **apex points down**, so the apex region is the bottom of
the image and the base region is the top.

| Region | Purpose | Fraction | Output |
|---|---|---|---|
| Full | Leaf Shape Classification | 0% – 100% | `Full-leaf.jpg` |
| Top | Leaf **Base** Classification | 0% – 30% | `Top-leaf.jpg` |
| Middle | Leaf Margin Classification | 30% – 60% | `Middle-leaf.jpg` |
| Bottom | Leaf **Apex** Classification | 70% – 100% | `Bottom-leaf.jpg` |

## Region Diagram

```text
     stalk (petiole) up
┌─────────────────────┐
│                     │
│    BASE  0-30%      │  -> Top-leaf.jpg
│                     │
├─────────────────────┤
│                     │
│   MARGIN  30-60%    │  -> Middle-leaf.jpg
│                     │
├─────────────────────┤
│   (unused 60-70%)   │
├─────────────────────┤
│                     │
│    APEX  70-100%    │  -> Bottom-leaf.jpg
│                     │
└─────────────────────┘
      tip (apex) down
```

---

# Process 3 — Image Storage

Generated regions are uploaded to S3 **in the background** after the response is
sent, so storage latency never delays the caller.

```text
Full-leaf.jpg
Top-leaf.jpg
Middle-leaf.jpg
Bottom-leaf.jpg
```

Each upload is linked to the `classify_api_logs` row through
`classify_image_dataset`.

Set `AWS_ALLOWED_UPLOADED=false` and the regions are written to
`LOCAL_UPLOAD_DIR` (default `./uploads`) instead of S3. The layout under it
mirrors the S3 keys — `<S3_DATASET_PREFIX>/<region>/<file>.jpg` — so a local
run can be synced to the bucket later as-is, and the local paths land in the
same `classify_image_dataset.cdn_url` column the CDN URLs use. Either way the
work happens in the background task and the caller sees no difference.

---

# Process 4 — Classification

## Model

```text
Resnet50V2
```

Model file:

```text
Model_Classification/R50_Apex_final_V1.keras
Model_Classification/R50_Base_final_V1.keras
Model_Classification/R50_Margin_final_V1.keras
Model_Classification/R50_Shape_final_V0.keras
```

Each region is classified independently. At runtime `entrypoint.sh` may convert
these to TFLite or export an OpenVINO model for faster CPU inference; the class
lists are unchanged either way.

| Model | Input | Classes | Total |
|---|---|---|---|
| Shape | `Full-leaf.jpg` | Ovate, Cordate, Sagittate, Lanceolate | 4 |
| Apex | `Bottom-leaf.jpg` | Acute, Caudate, Cuspidate, Obtuse | 4 |
| Base | `Top-leaf.jpg` | Auriculate, Caudate, Cuneate, Obtuse | 4 |
| Margin | `Middle-leaf.jpg` | Crenate, Entire | 2 |

The class lists live in `src/config.py` (`SHAPE_CLASSES`, `APEX_CLASSES`,
`BASE_CLASSES`, `MARGIN_CLASSES`) and are the single source of truth — the rule
base validates against them and the admin page builds its dropdowns from them.

Thai names for every class are in `THAI_LABELS` (`src/classify/service.py`) and
are returned alongside the English ones.

---

# Current Classification Architecture

The system uses four independent models.

```text
Shape Model
Apex Model
Base Model
Margin Model
```

Advantages:

* Independent training
* Easier dataset management
* Easier retraining
* Easier deployment
* Better debugging
* Individual confidence scores

Predictions are serialized behind a lock: the TFLite interpreters are shared
module state and are not thread-safe.

---

# Process 5 — Rule Base

The four predicted traits are matched against a table of rules to produce the
group name returned in `prediction.label`.

## Data model

One row in `classify_rule_group` = one exact combination of the four traits
mapped to a group name.

```text
id | code | name          | shape      | apex    | base       | margin  | is_active
 1 | G1   | กลุ่มใบหัวใจ    | Cordate    | Acute   | Auriculate | Entire  | 1
 2 | G2   | กลุ่มใบไข่      | Ovate      | Obtuse  | Cuneate    | Crenate | 1
```

Every trait holds exactly one class, so the combination is the natural key:
`UNIQUE (shape, apex, base, margin)` stops two groups claiming the same leaf.
The total space is 4 × 4 × 4 × 2 = **128 combinations**.

## Scoring

`match_group()` scores every active group and returns the best. Ranking is
count-first:

| Case | Matched | Score |
|---|---|---|
| Trait matches, model is confident | +1 | + the model's confidence |
| Model is unsure (below `conf_th`, default 0.6) | +0.5 | +0.5 |
| Trait does not match, model is confident | 0 | 0 |

Groups are ordered by `matched`, then `score`, then `code`. An unsure model
never rules a group out, and a group matching more traits never loses to one
matching fewer.

`/api/classify` returns only the winning group's name. The full Top-3 is
available on the admin test bench.

If the rule base is empty — or a query fails — `prediction.label` comes back
`null` and the trait predictions are still returned. A rule problem never turns
a successful classification into a 500.

## Varieties

Each group can hold named yam varieties — "มันเสือ", "มันขาว" — in
`classify_rule_variety`. A variety belongs to **exactly one** group, and its name
is unique across the whole table.

```text
classify_rule_variety
 id | group_id | name     | is_active
  1 |        1 | มันเสือ   | 1
  2 |        1 | มันขาว    | 0
```

Varieties do **not** affect scoring. `/api/classify` returns the matched group's
`group_id` and the consumer fetches the varieties itself — a group with no
varieties still classifies normally. Deleting a group deletes its varieties
(`ON DELETE CASCADE`).

Managed from the **ชนิดมัน** tab in the admin page. The add form takes **one group
and any number of names at once**, each with its own active flag; the edit form
handles a single variety and can move it to another group.

### Indexed validation

The add form submits many rows, so "a name is invalid" would not be actionable.
Every validation failure names the row it came from:

```json
{
  "code": 422,
  "status": "error",
  "message": "Validation failed",
  "errors": {
    "type": "VALIDATION_ERROR",
    "fields": [
      {"index": 1, "field": "name", "message": "กรุณากรอกชื่อชนิดมัน"},
      {"index": 2, "field": "name", "message": "ชื่อซ้ำกับบรรทัดที่ 1"},
      {"index": null, "field": "group_id", "message": "ไม่พบกลุ่มนี้"}
    ]
  }
}
```

`index` is the position in `items`, or `null` for a form-level field such as
`group_id`. The admin page paints each message under the input it belongs to.

Checked per row: empty name, a name repeated within the same submission, and a
name already in the database — all matched case-insensitively. A batch is
all-or-nothing: one bad row and nothing is written.

This shape also covers type errors. `src/main.py` installs a
`RequestValidationError` handler that rewrites FastAPI's default flat `detail`
list into the same envelope, turning pydantic's `loc` path into `index` + `field`
— so **every** 422 from this service now looks like the example above, including
the ones from `/api/classify`.

## Caching

The active rule set is cached in-process for 30 seconds and invalidated
immediately on any write through `/api/rules`, so an edit in the admin page
takes effect on the next request.

---

# Admin page

Served from `src/static/` at `/admin/` (`/` and `/admin` both redirect there).
Plain HTML/CSS/JS — no build step, no CDN, no external fonts.

* **Login** — username/password checked against the platform `users` table
* **Rules tab** — list, create, edit, delete rules; every trait is a dropdown
  built from the configured class lists, so a typo is impossible
* **ชนิดมัน tab** — pick a group, then add as many variety names as you like in
  one go; bad rows are flagged in place
* **Test tab** — enter four traits and confidences by hand and see the Top-3
  with the same `match_group()` the API uses

## Authentication

`POST /api/admin/login` checks `users.username` / `users.password` (Laravel
bcrypt, `$2y$`) and returns a JWT:

```json
{"access_token": "…", "token_type": "bearer", "expires_in": 28800,
 "user": {"id": "019ea0de-…", "username": "admin", "name": "Administrator", "role": "admin"}}
```

Send it as `Authorization: Bearer <token>` — the same token works from the admin
page and from any other service.

**Only `role = 'admin'` with `is_active = 1` gets a token.** A wrong password, an
unknown username, a non-admin role and a disabled account all return the same
401 with the same message, so a caller cannot enumerate admin accounts.

This service **reads** `users` and never writes to it. `/api/genToken` is
unaffected and still uses `classify_user`.

Every `/api/rules` route is protected at the router level, not per route, so a
route added later is protected by default. Authenticated responses carry
`Cache-Control: no-store, private`.

> There is no server-side session, so **logout is client-side only** and a token
> stays valid until it expires (`JWT_EXPIRE_MINUTES`, 8h by default). There is no
> revocation. Shorten the lifetime if that matters. The admin page keeps its
> token in `localStorage`, which any script on the page can read — do not add
> third-party scripts to `src/static/`.

> In `/docs`, use the **Authorize** button and paste the token from
> `/api/admin/login`.

---

# API Reference

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/genToken` | none | Issue/return an API key (registers unknown users) |
| POST | `/api/classify` | `X-API-Key` | Classify one leaf image |
| POST | `/classify` | — | Legacy alias, 308 redirect to `/api/classify` |
| POST | `/api/admin/login` | none | Exchange credentials for a Bearer token |
| POST | `/api/admin/logout` | none | No-op — the client discards its token |
| GET | `/api/admin/me` | Bearer | Identity carried by the token |
| GET | `/api/rules` | Bearer | List rule groups (with a variety count) |
| POST | `/api/rules` | Bearer | Create a rule group |
| GET | `/api/rules/{id}` | Bearer | One rule group |
| PUT | `/api/rules/{id}` | Bearer | Update a rule group |
| DELETE | `/api/rules/{id}` | Bearer | Delete a rule group |
| GET | `/api/rules/varieties` | Bearer | List varieties |
| POST | `/api/rules/varieties` | Bearer | Create several varieties in one group |
| GET | `/api/rules/varieties/{id}` | Bearer | One variety |
| PUT | `/api/rules/varieties/{id}` | Bearer | Update a variety (name, group, active) |
| DELETE | `/api/rules/varieties/{id}` | Bearer | Delete a variety |
| POST | `/api/rules/test` | Bearer | Score a hand-entered prediction (Top-N) |
| GET | `/api/rules/vocab` | Bearer | The class list per trait |
| GET | `/health` | none | DB + model readiness (200 / 503) |
| GET | `/health/live` | none | Liveness only, touches nothing external |

Interactive docs at `/docs`.

## POST /api/classify

Endpoint:

```http
POST /api/classify
Content-Type: multipart/form-data
X-API-Key: <api key>
```

Field name:

```text
image
```

Requirements:

* Single image
* JPEG or PNG
* Max **10 MB**

## Success Response

```http
200 OK
```

```json
{
  "code": 200,
  "status": "success",
  "message": "Image processed successfully",
  "data": {
    "shape": "Cordate",
    "apex": "Acute",
    "base": "Auriculate",
    "margin": "Entire",
    "shape_th": "รูปหัวใจ",
    "apex_th": "แหลม",
    "base_th": "รูปติ่งหู",
    "margin_th": "เรียบ",
    "prediction": {
      "group_id": 1,
      "code": "G1",
      "label": "กลุ่มใบหัวใจ",
      "confidence": 97.42
    }
  }
}
```

`prediction.label` is the **name of the matched rule group**; `group_id` and
`code` identify it, so the consumer can fetch that group's varieties. All three
are `null` when no rule matched. `prediction.confidence` is the mean of the four
model confidences — it describes the trait predictions, not the group match.

## Confidence Rules

Expected format:

```json
{ "confidence": 95.50 }
```

Scale:

```text
0 - 100
```

NOT:

```json
{ "confidence": 0.955 }
```

Internally `match_group()` works on 0–1 probabilities; `src/classify/router.py`
divides by 100 before calling it. The API contract stays 0–100.

## Validation Error

```http
400 Bad Request
```

```json
{
  "code": 400,
  "status": "error",
  "message": "Image too large",
  "errors": {
    "type": "VALIDATION_ERROR",
    "details": "Maximum image size is 10 MB"
  }
}
```

## Rate Limit

Classification is serialized, so past `MAX_CONCURRENT_CLASSIFY` (default 4)
requests in flight the API sheds load instead of letting clients time out.

```http
429 Too Many Requests
Retry-After: 5
```

```json
{
  "code": 429,
  "status": "error",
  "message": "Server busy — too many classifications in progress",
  "errors": {
    "type": "RATE_LIMIT",
    "details": "At most 4 images are processed at a time. Retry in 5s."
  }
}
```

## Processing Error

```http
500 Internal Server Error
```

```json
{
  "code": 500,
  "status": "error",
  "message": "Image invalid or cannot be processed",
  "errors": {
    "type": "PROCESSING_ERROR",
    "details": "Unsupported image format"
  }
}
```

## Calling the admin API from a terminal

```bash
TOKEN=$(curl -s -X POST 'http://localhost:15780/api/admin/login' \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"<password>"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["access_token"])')

curl -H "Authorization: Bearer $TOKEN" 'http://localhost:15780/api/rules/varieties'
```

---

# Database Schema

All tables are created by `src/seed.py` on startup. Raw SQL over a
`mysql-connector` pool — no ORM, no alembic.

| Table | Purpose |
|---|---|
| `classify_user` | Username + bcrypt hash. Used by both API keys and admin login |
| `classify_token` | One API key per user |
| `classify_api_logs` | Every call: key, IP, status, traits, confidence, duration, group |
| `classify_image_dataset` | CDN URLs of the four uploaded regions per log row |
| `classify_rule_group` | The rule base — one trait combination per group |
| `classify_rule_variety` | Named yam varieties, one group each |
| `users` | **Read-only here.** Laravel-managed; backs the admin login |

`seed.py` also migrates in place: missing columns are added with
`_ensure_column`, rules stored under the older child-table layout are folded into
the trait columns, and the short-lived many-to-many variety link table is folded
into `group_id` and dropped once empty.

---

# Configuration

Everything is read from the environment / `.env` through `src/config.py`. See
`.env.example` for the full list.

| Variable | Default | Notes |
|---|---|---|
| `JWT_SECRET` | — | **Required.** The app refuses to start without it |
| `JWT_EXPIRE_MINUTES` | `480` | Bearer token lifetime; there is no revocation |
| `SEED_USERNAME` / `SEED_PASSWORD` | `admin` / random | Seeds `classify_user` for `/api/genToken` — **not** the admin login |
| `DB_HOST` … `DB_NAME` | | MySQL connection |
| `S3_BUCKET`, `AWS_*` | | Model download + region upload |
| `AWS_ALLOWED_UPLOADED` | `true` | `false` writes regions to `LOCAL_UPLOAD_DIR` instead of S3 |
| `LOCAL_UPLOAD_DIR` | `./uploads` | Where regions go when `AWS_ALLOWED_UPLOADED=false` |
| `MAX_CONCURRENT_CLASSIFY` | `4` | Load-shedding threshold |
| `YOLO_IMGSZ` | `640` | Detection cost scales with this |
| `NUM_THREADS` | `2` | Match the deploy target's core count |

In CI the values come from Vault; `Jenkinsfile` fails the build early if any
required key — `JWT_SECRET` included — is missing.

---

# Backend Integration Notes

Backend System:

```text
Laravel
```

Consumer:

```text
ClassificationService
```

Behavior:

```text
2xx
 → processed

429
 → retry after Retry-After

4xx / 5xx
 → failed

timeout
 → timeout
```

Backend timeout:

```text
300 seconds
```

---

# Testing

```bash
pytest tests -q          # local
docker compose build --target test && docker compose run --rm app   # in CI
```

Tests never touch a real database or model: repositories and the model service
are patched. `tests/rules/test_router.py` enumerates the router's own routes to
assert every one of them requires a session, so a new endpoint is covered
without editing a list.

---

# Recommended Future Improvements

* Species Classification Model
* Top-K predictions on `/api/classify` (already available on the admin test bench)
* Batch Processing
* Model Version Tracking
* Explainable AI Visualization
* Leaf Segmentation Mask Storage
* Feature Confidence Analytics
* Rule base import/export (YAML or CSV)
* Token revocation — logout cannot currently invalidate an issued token

---

# Technology Stack

```text
Python 3.12
TensorFlow 2.20
Keras 3.13
OpenVINO 2025.3
NumPy 2.0
OpenCV 4.12
Scikit-Learn 1.6
Matplotlib 3.10

YOLO11s (ultralytics 8.4)
ResNet50V2

FastAPI 0.136
PyJWT 2.10 + bcrypt 5.0
MySQL (mysql-connector-python 9.3)
boto3 1.38

Laravel (consumer)
```

---

# Project Status

Current Pipeline Version:

```text
v1.1
```

Architecture:

```text
YOLO Detection
+
4 Independent Classification Models
+
Rule Base in MySQL (admin CRUD page)
+
REST API
+
Laravel Integration
```
