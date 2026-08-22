# src/classify/service.py
import logging
import os
import threading
import cv2
import numpy as np
from src.config import settings

logger = logging.getLogger(__name__)

yolo_model   = None
shape_model  = None
apex_model   = None
base_model   = None
margin_model = None

_load_lock = threading.Lock()

# TFLite interpreters are not thread-safe and are shared process-wide, so every
# prediction — across concurrent requests, not just within one — must hold this.
_predict_lock = threading.Lock()

THAI_LABELS: dict[str, dict[str, str]] = {
    "shape": {          # รูปร่างแผ่นใบ
        "Ovate":      "รูปไข่",
        "Cordate":    "รูปหัวใจ",
        "Sagittate":  "รูปหัวลูกศร",
        "Lanceolate": "รูปใบหอก",
    },
    "apex": {           # ปลายใบ
        "Acute":     "แหลม",
        "Caudate":   "ยาวคล้ายหาง",
        "Cuspidate": "ติ่งแหลม",
        "Obtuse":    "มน",
    },
    "base": {           # โคนใบ
        "Auriculate": "รูปติ่งหู",
        "Caudate":    "รูปหัวใจ",
        "Cuneate":    "รูปลิ่ม",
        "Obtuse":     "มน",
    },
    "margin": {         # ขอบใบ
        "Crenate": "หยักมน",
        "Entire":  "เรียบ",
    },
}


class TFLiteModel:
    """Drop-in replacement for keras Model.predict backed by a TFLite
    interpreter (XNNPACK). NOT thread-safe — every caller must hold
    _predict_lock (see predict_all)."""

    def __init__(self, path: str):
        import tensorflow as tf
        self._interpreter = tf.lite.Interpreter(
            model_path=path, num_threads=settings.NUM_THREADS
        )
        self._interpreter.allocate_tensors()
        self._input_index  = self._interpreter.get_input_details()[0]["index"]
        self._output_index = self._interpreter.get_output_details()[0]["index"]

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        self._interpreter.set_tensor(self._input_index, x.astype("float32"))
        self._interpreter.invoke()
        return self._interpreter.get_tensor(self._output_index)


def _load_classifier(keras_path: str):
    """Prefer a sibling .tflite file (see scripts/convert_tflite.py);
    fall back to the original .keras model."""
    tflite_path = os.path.splitext(keras_path)[0] + ".tflite"
    if os.path.exists(tflite_path):
        try:
            model = TFLiteModel(tflite_path)
            print(f"  using TFLite: {tflite_path}")
            return model
        except Exception as e:
            # Truncated or incompatible artifact — the .keras original still works.
            print(f"  TFLite load failed ({tflite_path}): {e} — falling back to Keras")
    from tensorflow.keras.models import load_model
    print(f"  using Keras:  {keras_path}")
    return load_model(keras_path)


def load_models():
    global yolo_model, shape_model, apex_model, base_model, margin_model
    if yolo_model is None:
        with _load_lock:
            if yolo_model is None:
                import tensorflow as tf
                try:
                    # Must run before TF initializes its thread pools.
                    tf.config.threading.set_intra_op_parallelism_threads(settings.NUM_THREADS)
                    tf.config.threading.set_inter_op_parallelism_threads(1)
                except RuntimeError:
                    pass  # TF context already initialized (e.g. in tests)

                from ultralytics import YOLO
                print("Loading models...")
                yolo_model   = YOLO(settings.YOLO_MODEL_PATH)
                shape_model  = _load_classifier(settings.SHAPE_MODEL_PATH)
                apex_model   = _load_classifier(settings.APEX_MODEL_PATH)
                base_model   = _load_classifier(settings.BASE_MODEL_PATH)
                margin_model = _load_classifier(settings.MARGIN_MODEL_PATH)
                _warn_on_class_count_mismatch()
                print("All models loaded")


def _warn_on_class_count_mismatch() -> None:
    """Report heads whose output width does not match their label list.

    Extra outputs are silently ignored at predict time (see predict_class),
    which keeps the API up but means those classes can never be reported —
    surface it at startup instead of letting it hide in production.
    """
    heads = (
        ("shape",  shape_model,  settings.SHAPE_CLASSES),
        ("apex",   apex_model,   settings.APEX_CLASSES),
        ("base",   base_model,   settings.BASE_CLASSES),
        ("margin", margin_model, settings.MARGIN_CLASSES),
    )
    for name, model, classes in heads:
        try:
            if isinstance(model, TFLiteModel):
                outputs = int(model._interpreter.get_output_details()[0]["shape"][-1])
            else:
                outputs = int(model.output_shape[-1])
        except Exception:
            continue  # never let a diagnostic stop startup

        if outputs != len(classes):
            logger.warning(
                "%s model outputs %d classes but %s_CLASSES names %d (%s) — "
                "the extra outputs are ignored and can never be returned",
                name, outputs, name.upper(), len(classes), ", ".join(classes),
            )


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
    # Detection dominates request time (~72% at 1600x1200 input), and it scales
    # with the letterboxed input, not the source resolution. YOLO_IMGSZ caps it;
    # the crop below still comes from the full-resolution original because the
    # box coordinates are rescaled back by ultralytics.
    results = yolo_model(image, imgsz=settings.YOLO_IMGSZ)[0]
    for box in results.boxes:
        cls_name = yolo_model.names[int(box.cls)]
        if cls_name.lower() == settings.TARGET_CLASS.lower():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return _crop_to_aspect(image, x1, y1, x2, y2, settings.ASPECT_W, settings.ASPECT_H)
    return None


def slice_leaf(cropped: np.ndarray) -> dict:
    """Cut the crop into the bands each head was trained on.

    **Input contract: the leaf must be upright with its tip (apex) pointing
    DOWN and the stalk (petiole) up** — the framing used to produce the
    training crops (Etc/Extractimage-Test-Process.py against the
    Testset-*-Shadow-output photos). The bands are fixed fractions of image
    height, so a leaf that arrives sideways or flipped hands every head a
    region it was never trained on and the API answers confident nonsense.

    Nothing here detects or corrects orientation, and that is deliberate: a
    wrong guess destroys framing that was already correct, and no cue in these
    photos settles it — the petiole, the one botanically reliable marker, is
    cut off before the leaf is photographed.

    The 0.60-0.70 gap is also deliberate: the transition between the middle of
    the blade and the far end belongs to neither the margin nor the base
    classifier. develop had closed it (0.65/0.65), widening both inputs by 5%
    of the leaf against what the models saw in training; these bounds match
    the extraction script and main:app.py.
    """
    h = cropped.shape[0]
    return {
        "full":   cropped,
        "top":    cropped[0:int(h * 0.30)],
        "middle": cropped[int(h * 0.30):int(h * 0.60)],
        "bottom": cropped[int(h * 0.70):h],
    }


def preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (settings.IMG_SIZE, settings.IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_class(model, image: np.ndarray, classes: list[str]) -> tuple[str, float]:
    pred = model.predict(preprocess(image), verbose=0)[0]

    # The margin model emits 4 logits while MARGIN_CLASSES names 2, so a raw
    # argmax can index past the end of the list and 500 the whole request.
    # Ignore the unnamed outputs rather than guessing a label for them;
    # load_models() logs the mismatch at startup.
    if len(pred) > len(classes):
        pred = pred[:len(classes)]

    idx = int(np.argmax(pred))
    return classes[idx], round(float(pred[idx] * 100), 2)


def predict_all(regions: dict) -> dict[str, tuple[str, float]]:
    """Run the four classifiers sequentially under a global lock.

    The interpreters are shared module state and are not thread-safe, and
    routes/classify.py dispatches this to a thread pool — so concurrent
    requests would otherwise interleave set_tensor/invoke on the same
    interpreter and return each other's logits (or segfault). On a 2-core
    host parallel predictions fight over cores anyway, so serializing costs
    nothing."""
    with _predict_lock:
        return {
            "shape":  predict_class(shape_model,  regions["full"],   settings.SHAPE_CLASSES),
            "base":   predict_class(base_model,   regions["top"], settings.BASE_CLASSES),
            "margin": predict_class(margin_model, regions["middle"], settings.MARGIN_CLASSES),
            "apex":   predict_class(apex_model,   regions["bottom"],    settings.APEX_CLASSES),
        }


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
    

def mapping_predict_thai_name(regions: str, label: str) -> str:
    """Thai name for a predicted class, e.g. ("apex", "Caudate") -> ยาวคล้ายหาง.

    `regions` is the head the label came from — shape, apex, base or margin —
    and it matters: "Caudate" and "Obtuse" appear under more than one head with
    different Thai names.

    An unknown region or label returns the English label unchanged and logs it,
    rather than raising: a missing translation should not turn a successful
    classification into a 500.
    """
    if not label:
        return label

    table = THAI_LABELS.get((regions or "").strip().lower())
    if table is None:
        logger.warning("No Thai label table for region %r", regions)
        return label

    thai = table.get(label)
    if thai is None:   # tolerate casing differences from callers
        thai = next((v for k, v in table.items() if k.lower() == label.lower()), None)

    if thai is None:
        logger.warning("No Thai name for %s label %r", regions, label)
        return label

    return thai
    
