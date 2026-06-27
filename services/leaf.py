# services/leaf.py
import threading
import cv2
import numpy as np
import config

yolo_model   = None
shape_model  = None
apex_model   = None
base_model   = None
margin_model = None

_load_lock = threading.Lock()


def load_models():
    global yolo_model, shape_model, apex_model, base_model, margin_model
    if yolo_model is None:
        with _load_lock:
            if yolo_model is None:
                from ultralytics import YOLO
                from tensorflow.keras.models import load_model
                print("Loading models...")
                yolo_model   = YOLO(config.YOLO_MODEL_PATH)
                shape_model  = load_model(config.SHAPE_MODEL_PATH)
                apex_model   = load_model(config.APEX_MODEL_PATH)
                base_model   = load_model(config.BASE_MODEL_PATH)
                margin_model = load_model(config.MARGIN_MODEL_PATH)
                print("All models loaded")


def _crop_to_aspect(image, x1, y1, x2, y2, aspect_w=4, aspect_h=3):
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


def detect_leaf(image: np.ndarray) -> np.ndarray | None:
    results = yolo_model(image)[0]
    for box in results.boxes:
        cls_name = yolo_model.names[int(box.cls)]
        if cls_name.lower() == config.TARGET_CLASS.lower():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return _crop_to_aspect(image, x1, y1, x2, y2, config.ASPECT_W, config.ASPECT_H)
    return None


def slice_leaf(cropped: np.ndarray) -> dict:
    h = cropped.shape[0]
    return {
        "full":   cropped,
        "top":    cropped[0:int(h * 0.30)],
        "middle": cropped[int(h * 0.30):int(h * 0.65)],
        "bottom": cropped[int(h * 0.65):h],
    }


def preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_class(model, image: np.ndarray, classes: list[str]) -> tuple[str, float]:
    pred = model.predict(preprocess(image), verbose=0)[0]
    idx = int(np.argmax(pred))
    return classes[idx], round(float(pred[idx] * 100), 2)


def warmup_models() -> None:
    import time
    print("Warmup [1/5] YOLO — starting...")
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    t = time.perf_counter()
    yolo_model(dummy)
    print(f"Warmup [1/5] YOLO — done ({time.perf_counter() - t:.1f}s)")

    inp = preprocess(dummy)
    for i, (model, name) in enumerate(
        zip(
            (shape_model, apex_model, base_model, margin_model),
            ("Shape", "Apex", "Base", "Margin"),
        ),
        start=2,
    ):
        print(f"Warmup [{i}/5] {name} — starting...")
        t = time.perf_counter()
        model.predict(inp, verbose=0)
        print(f"Warmup [{i}/5] {name} — done ({time.perf_counter() - t:.1f}s)")

    print("Warmup complete — all models ready")
