# Leaf-Classification

# Yamwisdom Phase 2 - Leaf Classification AI Service

## Overview


## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r Requirement.txt
cp .env.example .env          # then fill in DB + AWS values
python -m src.main

# or with Docker (the override file publishes the host port):
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
pytest tests -q
```

## Project structure

```
Leaf-Classification/
├── src/
│   ├── main.py               # FastAPI app, lifespan (seed + model load/warmup)
│   ├── config.py             # Settings (pydantic BaseSettings) -> `settings`
│   ├── database.py           # MySQL connection pool, get_conn()
│   ├── seed.py               # CREATE TABLE + first admin user / API key
│   │
│   ├── auth/                 # Feature module: authentication
│   │   ├── router.py         # POST /api/genToken
│   │   ├── schemas.py        # Request/response models
│   │   ├── service.py        # get_or_create_api_key()
│   │   ├── repository.py     # SQL for classify_user / classify_token
│   │   ├── security.py       # bcrypt hashing, API key generation
│   │   ├── dependencies.py   # require_api_key (X-API-Key)
│   │   └── exceptions.py
│   │
│   ├── classify/             # Feature module: leaf classification
│   │   ├── router.py         # POST /api/classify
│   │   ├── service.py        # YOLO detect, slice, TFLite/Keras predict
│   │   ├── storage.py        # S3 upload of cropped regions
│   │   └── repository.py     # SQL for classify_api_logs / image_dataset
│   │
│   └── health/router.py      # GET /health (DB + model readiness)
│
├── tests/                    # conftest.py + tests/auth, tests/classify
├── scripts/                  # convert_tflite.py, export_openvino.py
├── Model-Leaf/               # YOLO weights (runtime volume)
├── Model_Classification/     # Keras/TFLite classifiers (runtime volume)
├── .env.example
├── Requirement.txt
├── Dockerfile                # base -> test / runtime targets
├── docker-compose.yml        # shared service (no published port)
├── docker-compose.dev.yml    # dev override  — host :15780
├── docker-compose.prod.yml   # prod override — host :16780
├── Jenkinsfile               # Vault -> build -> test -> deploy -> health check
└── entrypoint.sh             # fetches/converts models, then `python -m src.main`
```


This project is part of the Yamwisdom Phase 2 platform.

The objective is to classify cassava leaf characteristics using Computer Vision and Deep Learning models.

The system consists of:

1. Leaf Detection (YOLO)
2. Image Cropping (4:3 aspect ratio)
3. Region Extraction
4. Feature Classification
5. API Service Integration
6. Laravel Backend Integration

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
Save Images
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
Combine Results
     │
     ▼
Classification API Response
     │
     ▼
Laravel Backend
```

---

# Process 1 — Leaf Detection

## Model

```text
YOLO11x
```

Model file:

```text
yolo11x_leaf.pt
```

Target class:

```text
leaf
```

The detector is responsible for locating the leaf within the image.

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

Example:

```text
Original BBox
 ┌─────────┐
 │  Leaf   │
 └─────────┘

Expanded Crop Area
 ┌─────────────────┐
 │                 │
 │      Leaf       │
 │                 │
 └─────────────────┘
```

---

# Process 2 — Region Extraction

After cropping, the image is sliced into separate regions.

---

## Full Image

Purpose:

```text
Leaf Shape Classification
```

Region:

```text
0% - 100%
```

Output:

```text
Full-leaf.jpg
```

---

## Apex Region

Purpose:

```text
Leaf Apex Classification
```

Region:

```text
0% - 30%
```

Output:

```text
Top-leaf.jpg
```

---

## Margin Region

Purpose:

```text
Leaf Margin Classification
```

Region:

```text
30% - 60%
```

Output:

```text
Middle-leaf.jpg
```

---

## Base Region

Purpose:

```text
Leaf Base Classification
```

Region:

```text
70% - 100%
```

Output:

```text
Bottom-leaf.jpg
```

---

## Region Diagram

```text
┌─────────────────────┐
│                     │
│       APEX          │
│       0-30%         │
│                     │
├─────────────────────┤
│                     │
│      MARGIN         │
│      30-60%         │
│                     │
├─────────────────────┤
│                     │
│                     │
│                     │
├─────────────────────┤
│                     │
│       BASE          │
│      70-100%        │
│                     │
└─────────────────────┘
```

---

# Process 3 — Image Storage

Generated regions are stored separately.

Stored images:

```text
Full-leaf.jpg
Top-leaf.jpg
Middle-leaf.jpg
Bottom-leaf.jpg
```

Each image can be:

* Saved to local storage
* Saved to object storage
* Saved via API
* Linked to database records

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

Each region is classified independently.

---

## Shape Model

Input:

```text
Full-leaf.jpg
```

Classes:

```text
Ovate
Cordate
Sagittate
Lanceolate
```

Total Classes:

```text
4
```

---

## Apex Model

Input:

```text
Top-leaf.jpg
```

Classes:

```text
Acute
Caudate
Cuspidate
Obtuse
```

Total Classes:

```text
4
```

---

## Base Model

Input:

```text
Bottom-leaf.jpg
```

Classes:

```text
Auriculate
Caudate
Cuneate
Obtuse
```

Total Classes:

```text
4
```

---

## Margin Model

Input:

```text
Middle-leaf.jpg
```

Classes:

```text
Crenate
Entire
```

Total Classes:

```text
2
```

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

---

# Detection & Slicing Reference Code

Current implementation:

```python
YOLO Detection
    ↓
Crop 4:3
    ↓
Generate:

Full-leaf.jpg
Top-leaf.jpg
Middle-leaf.jpg
Bottom-leaf.jpg
```

Cropping method:

* Bounding-box based
* Aspect ratio expansion
* No distortion
* No image squeezing

---

# API Service Specification

Endpoint:

```http
POST /classify
```

Content-Type:

```http
multipart/form-data
```

---

## Request

Field name:

```text
image
```

Example:

```http
POST /classify

Content-Type: multipart/form-data

image=<leaf.jpg>
```

Requirements:

* Single image
* JPEG
* PNG
* Max 5 MB

---

# Success Response

HTTP Status:

```http
200 OK
```

Example:

```json
{
  "code": 200,
  "status": "success",
  "message": "Image processed successfully",
  "data": {
    "shape": "Ovate",
    "apex": "Acute",
    "base": "Cuneate",
    "margin": "Entire",
    "prediction": {
      "label": "Manihot esculenta",
      "confidence": 97.42
    }
  }
}
```

---

# Required Response Fields

```json
{
  "data": {
    "shape": "...",
    "apex": "...",
    "base": "...",
    "margin": "...",
    "prediction": {
      "label": "...",
      "confidence": 95.50
    }
  }
}
```

---

# Confidence Rules

Expected format:

```json
{
  "confidence": 95.50
}
```

Scale:

```text
0 - 100
```

NOT:

```json
{
  "confidence": 0.955
}
```

---

# Validation Error

HTTP:

```http
400 Bad Request
```

Example:

```json
{
  "code": 400,
  "status": "error",
  "message": "Validation failed",
  "errors": {
    "type": "VALIDATED_ERROR",
    "details": "Image file is required"
  }
}
```

---

# Processing Error

HTTP:

```http
500 Internal Server Error
```

Example:

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

# Recommended Future Improvements

* Species Classification Model
* Top-K Predictions
* Confidence per Feature
* Batch Processing
* Model Version Tracking
* Inference Logging
* Explainable AI Visualization
* Leaf Segmentation Mask Storage
* Feature Confidence Analytics

---

# Technology Stack

```text
Python 3.12
TensorFlow 2.20
Keras 3.13
NumPy 2.0
OpenCV 4.13
Scikit-Learn 1.6
Matplotlib 3.10

YOLO11x
ResNet50

FastAPI
Laravel
MySQL
```

---

# Project Status

Current Pipeline Version:

```text
v1.0
```

Architecture:

```text
YOLO Detection
+
4 Independent Classification Models
+
REST API
+
Laravel Integration
```
