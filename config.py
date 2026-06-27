# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ── Server ───────────────────────────────────────────────────
PORT = int(os.getenv("PORT", "8000"))

# ── Database ──────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "3306"))
DB_USER     = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME     = os.getenv("DB_NAME", "leaf_db")

# ── AWS / S3 ──────────────────────────────────────────────────
S3_BUCKET          = os.getenv("S3_BUCKET", "")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
CDN_BASE           = os.getenv("CDN_BASE", "https://d3w0s75zdffeft.cloudfront.net")
S3_DATASET_PREFIX  = "datasets"

# ── Model paths ───────────────────────────────────────────────
YOLO_MODEL_PATH   = "./Model-Leaf/yolo11x_leaf.pt"
SHAPE_MODEL_PATH  = "./Model_Classification/R50_Shape_final_V0.keras"
APEX_MODEL_PATH   = "./Model_Classification/R50_Apex_final_V1.keras"
BASE_MODEL_PATH   = "./Model_Classification/R50_Base_final_V1.keras"
MARGIN_MODEL_PATH = "./Model_Classification/R50_Margin_final_V1.keras"

# ── Class labels ─────────────────────────────────────────────
SHAPE_CLASSES  = ["Ovate", "Cordate", "Sagittate", "Lanceolate"]
APEX_CLASSES   = ["Acute", "Caudate", "Cuspidate", "Obtuse"]
BASE_CLASSES   = ["Auriculate", "Caudate", "Cuneate", "Obtuse"]
MARGIN_CLASSES = ["Crenate", "Entire"]

# ── Detection ─────────────────────────────────────────────────
TARGET_CLASS = "leaf"
ASPECT_W     = 4
ASPECT_H     = 3
IMG_SIZE     = 256
