# app.py
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi.responses import JSONResponse

from tensorflow.keras.models import load_model


# ============================================================
# CONFIG
# ============================================================

YOLO_MODEL_PATH = "./Model-Leaf/yolo11x_leaf.pt"

SHAPE_MODEL_PATH = "./Model-Leaf/shape.keras"
APEX_MODEL_PATH = "./Model-Leaf/apex.keras"
BASE_MODEL_PATH = "./Model-Leaf/base.keras"
MARGIN_MODEL_PATH = "./Model-Leaf/margin.keras"

TARGET_CLASS = "leaf"

ASPECT_W = 4
ASPECT_H = 3

IMG_SIZE = 224


# ============================================================
# CLASS MAP
# ============================================================

SHAPE_CLASSES = [
    "Ovate",
    "Cordate",
    "Sagittate",
    "Lanceolate"
]

APEX_CLASSES = [
    "Acute",
    "Caudate",
    "Cuspidate",
    "Obtuse"
]

BASE_CLASSES = [
    "Auriculate",
    "Caudate",
    "Cuneate",
    "Obtuse"
]

MARGIN_CLASSES = [
    "Crenate",
    "Entire"
]


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

yolo_model = YOLO(YOLO_MODEL_PATH)

shape_model = load_model(SHAPE_MODEL_PATH)
apex_model = load_model(APEX_MODEL_PATH)
base_model = load_model(BASE_MODEL_PATH)
margin_model = load_model(MARGIN_MODEL_PATH)

print("All models loaded")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Leaf Classification API",
    version="1.0"
)


# ============================================================
# PROCESS 1
# ============================================================

def crop_to_aspect(
        image,
        x1,
        y1,
        x2,
        y2,
        aspect_w=4,
        aspect_h=3
):
    ih, iw = image.shape[:2]

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w / box_h > aspect_w / aspect_h:
        new_w = box_w
        new_h = int(new_w * aspect_h / aspect_w)
    else:
        new_h = box_h
        new_w = int(new_h * aspect_w / aspect_h)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    nx1 = max(0, cx - new_w // 2)
    ny1 = max(0, cy - new_h // 2)

    nx2 = min(iw, nx1 + new_w)
    ny2 = min(ih, ny1 + new_h)

    if nx2 - nx1 < new_w:
        nx1 = max(0, nx2 - new_w)

    if ny2 - ny1 < new_h:
        ny1 = max(0, ny2 - new_h)

    return image[ny1:ny2, nx1:nx2]


def detect_leaf(image):

    results = yolo_model(image)[0]

    for box in results.boxes:

        cls_name = yolo_model.names[int(box.cls)]

        if cls_name.lower() == TARGET_CLASS.lower():

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cropped = crop_to_aspect(
                image,
                x1,
                y1,
                x2,
                y2,
                ASPECT_W,
                ASPECT_H
            )

            return cropped

    return None


# ============================================================
# PROCESS 2
# ============================================================

def slice_leaf(cropped):

    h = cropped.shape[0]

    top_end = int(h * 0.30)

    mid_start = int(h * 0.30)
    mid_end = int(h * 0.60)

    bot_start = int(h * 0.70)

    return {
        "full": cropped,
        "top": cropped[0:top_end],
        "middle": cropped[mid_start:mid_end],
        "bottom": cropped[bot_start:h]
    }


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(img):

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    return img


# ============================================================
# PREDICTION
# ============================================================

def predict_class(
        model,
        image,
        classes
):

    image = preprocess(image)

    pred = model.predict(
        image,
        verbose=0
    )[0]

    idx = int(np.argmax(pred))

    confidence = float(pred[idx] * 100)

    return (
        classes[idx],
        round(confidence, 2)
    )


# ============================================================
# API
# ============================================================

@app.post("/classify")
async def classify(
        image: UploadFile = File(...)
):

    try:

        file_bytes = await image.read()

        np_arr = np.frombuffer(
            file_bytes,
            np.uint8
        )

        img = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if img is None:

            return JSONResponse(
                status_code=500,
                content={
                    "code": 500,
                    "status": "error",
                    "message": "Image invalid or cannot be processed",
                    "errors": {
                        "type": "PROCESSING_ERROR",
                        "details": "Unsupported image format"
                    }
                }
            )

        # ==================================================
        # PROCESS 1
        # ==================================================

        cropped = detect_leaf(img)

        if cropped is None:

            return JSONResponse(
                status_code=500,
                content={
                    "code": 500,
                    "status": "error",
                    "message": "Leaf not detected",
                    "errors": {
                        "type": "PROCESSING_ERROR",
                        "details": "No leaf found"
                    }
                }
            )

        # ==================================================
        # PROCESS 2
        # ==================================================

        regions = slice_leaf(cropped)

        # ==================================================
        # PROCESS 3
        # ==================================================

        shape_label, shape_conf = predict_class(
            shape_model,
            regions["full"],
            SHAPE_CLASSES
        )

        apex_label, apex_conf = predict_class(
            apex_model,
            regions["top"],
            APEX_CLASSES
        )

        base_label, base_conf = predict_class(
            base_model,
            regions["bottom"],
            BASE_CLASSES
        )

        margin_label, margin_conf = predict_class(
            margin_model,
            regions["middle"],
            MARGIN_CLASSES
        )

        overall_conf = round(
            (
                shape_conf +
                apex_conf +
                base_conf +
                margin_conf
            ) / 4,
            2
        )

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "status": "success",
                "message": "Image processed successfully",
                "data": {
                    "shape": shape_label,
                    "apex": apex_label,
                    "base": base_label,
                    "margin": margin_label,
                    "prediction": {
                        "label": None,
                        "confidence": overall_conf
                    }
                }
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "status": "error",
                "message": str(e),
                "errors": {
                    "type": "PROCESSING_ERROR",
                    "details": str(e)
                }
            }
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )

