# scripts/convert_tflite.py
"""
Convert the four Keras classification models to TFLite (FP32) for fast
CPU inference via the XNNPACK delegate. Weights are unchanged, so
predictions are identical to the .keras originals.

Run inside the API environment (same deps as the server):

    python scripts/convert_tflite.py                 # all four
    python scripts/convert_tflite.py path/to/a.keras # only the given models

Each `<name>.keras` gains a sibling `<name>.tflite`; services/leaf.py
automatically prefers the .tflite file when it exists.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from src.config import settings

MODEL_PATHS = [
    settings.SHAPE_MODEL_PATH,
    settings.APEX_MODEL_PATH,
    settings.BASE_MODEL_PATH,
    settings.MARGIN_MODEL_PATH,
]


def convert(keras_path: str) -> None:
    tflite_path = os.path.splitext(keras_path)[0] + ".tflite"
    print(f"Converting {keras_path} -> {tflite_path}")

    model = tf.keras.models.load_model(keras_path)

    # Keras 3 models convert most reliably through the SavedModel format.
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.export(tmp_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
        tflite_bytes = converter.convert()

    # Write atomically: a kill mid-write would otherwise leave a truncated
    # .tflite that services/leaf.py would happily try to load forever.
    tmp_path = tflite_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(tflite_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, tflite_path)

    keras_mb = os.path.getsize(keras_path) / 1024 / 1024
    tflite_mb = os.path.getsize(tflite_path) / 1024 / 1024
    print(f"  done ({keras_mb:.0f} MB -> {tflite_mb:.0f} MB)")


def main() -> None:
    paths = sys.argv[1:] or MODEL_PATHS
    for path in paths:
        convert(path)
    print("All models converted — restart the API to pick up the .tflite files.")


if __name__ == "__main__":
    main()
