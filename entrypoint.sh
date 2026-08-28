#!/bin/bash
set -e

MODEL_LEAF_DIR="/app/Model-Leaf"
MODEL_CLASS_DIR="/app/Model_Classification"

# Override with YOLO_WEIGHTS to swap detectors — no code change needed.
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolo11s_leaf.pt}"
YOLO_NAME="${YOLO_WEIGHTS%.pt}"
OPENVINO_DIR="$MODEL_LEAF_DIR/${YOLO_NAME}_openvino_model"
OPENVINO_TAR="${YOLO_NAME}_openvino_model.tar.gz"

KERAS_MODELS=(
    "R50_Shape_final_V0"
    "R50_Apex_final_V1"
    "R50_Base_final_V1"
    "R50_Margin_final_V1"
)

download_if_missing() {
    local dest="$1"
    local s3_key="$2"
    local size
    size=$([ -f "$dest" ] && wc -c < "$dest" || echo 0)
    if [ ! -f "$dest" ] || [ "$size" -lt 1000000 ]; then
        echo "Downloading $(basename $dest) from S3..."
        aws s3 cp "s3://${S3_BUCKET}/${s3_key}" "${dest}.tmp" && mv "${dest}.tmp" "$dest" || { rm -f "${dest}.tmp"; exit 1; }
        echo "Done: $(basename $dest)"
    else
        echo "Found: $(basename $dest) (cached)"
    fi
}

# Optional artifact: return non-zero instead of exiting when absent on S3.
try_download() {
    local dest="$1"
    local s3_key="$2"
    local size
    size=$([ -f "$dest" ] && wc -c < "$dest" || echo 0)
    # A truncated artifact (killed mid-download/conversion) must not be trusted.
    if [ -f "$dest" ] && [ "$size" -ge 1000000 ]; then
        return 0
    fi
    rm -f "$dest"
    if aws s3 cp "s3://${S3_BUCKET}/${s3_key}" "${dest}.tmp" 2>/dev/null; then
        mv "${dest}.tmp" "$dest"
        return 0
    fi
    rm -f "${dest}.tmp"
    return 1
}

# Cache converted artifacts back to S3 so other/new hosts skip conversion.
# Best-effort: a failed upload must not block the API from starting.
upload_artifact() {
    local src="$1"
    local s3_key="$2"
    aws s3 cp "$src" "s3://${S3_BUCKET}/${s3_key}" \
        && echo "Uploaded to S3: ${s3_key}" \
        || echo "WARN: failed to upload ${s3_key} — conversion will re-run on fresh hosts"
}

mkdir -p "$MODEL_LEAF_DIR" "$MODEL_CLASS_DIR"

# ── 1. Original models ───────────────────────────────────────────
download_if_missing "$MODEL_LEAF_DIR/$YOLO_WEIGHTS" "ml_models/$YOLO_WEIGHTS"
for name in "${KERAS_MODELS[@]}"; do
    download_if_missing "$MODEL_CLASS_DIR/${name}.keras" "ml_models/${name}.keras"
done

# ── 2. Optimized artifacts (download from S3, or convert once & upload) ──
# Best-effort: .keras / .pt originals already work, so an optimization
# failure must degrade performance, never block startup.
if [ "${SKIP_MODEL_OPTIMIZE:-0}" != "1" ]; then
    set +e

    # TFLite versions of the 4 classifiers (services/leaf.py prefers them,
    # and falls back to .keras per model when one is missing).
    missing_tflite=()
    for name in "${KERAS_MODELS[@]}"; do
        try_download "$MODEL_CLASS_DIR/${name}.tflite" "ml_models/${name}.tflite" \
            || missing_tflite+=("$MODEL_CLASS_DIR/${name}.keras")
    done
    if [ "${#missing_tflite[@]}" -gt 0 ]; then
        echo "Converting ${#missing_tflite[@]} classifier(s) to TFLite (one-time)..."
        if python scripts/convert_tflite.py "${missing_tflite[@]}"; then
            for keras in "${missing_tflite[@]}"; do
                name="$(basename "${keras%.keras}")"
                upload_artifact "$MODEL_CLASS_DIR/${name}.tflite" "ml_models/${name}.tflite"
            done
        else
            echo "WARN: TFLite conversion failed — serving those classifiers from .keras"
        fi
    fi

    # OpenVINO export of the YOLO detector
    if [ ! -d "$OPENVINO_DIR" ]; then
        if try_download "/tmp/$OPENVINO_TAR" "ml_models/$OPENVINO_TAR"; then
            echo "Extracting cached OpenVINO model..."
            tar -xzf "/tmp/$OPENVINO_TAR" -C "$MODEL_LEAF_DIR"
            rm -f "/tmp/$OPENVINO_TAR"
        else
            echo "Exporting $YOLO_WEIGHTS to OpenVINO (one-time)..."
            if python scripts/export_openvino.py --model "$MODEL_LEAF_DIR/$YOLO_WEIGHTS"; then
                tar -czf "/tmp/$OPENVINO_TAR" -C "$MODEL_LEAF_DIR" "$(basename "$OPENVINO_DIR")" \
                    && upload_artifact "/tmp/$OPENVINO_TAR" "ml_models/$OPENVINO_TAR"
                rm -f "/tmp/$OPENVINO_TAR"
            else
                echo "WARN: OpenVINO export failed — falling back to $YOLO_WEIGHTS"
            fi
        fi
    else
        echo "Found: $(basename "$OPENVINO_DIR") (cached)"
    fi

    set -e

    # The archive may have been built from a different weights stem, so only
    # trust the directory once it actually exists.
    if [ -d "$OPENVINO_DIR" ]; then
        export YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-$OPENVINO_DIR/}"
    else
        echo "WARN: $OPENVINO_DIR not found — using $YOLO_WEIGHTS"
        export YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-$MODEL_LEAF_DIR/$YOLO_WEIGHTS}"
    fi
else
    # Keep YOLO_WEIGHTS authoritative here too, so config.py never falls back
    # to its hardcoded default for weights that were never downloaded.
    export YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-$MODEL_LEAF_DIR/$YOLO_WEIGHTS}"
fi
echo "Using YOLO model: $YOLO_MODEL_PATH"

exec python -m src.main
