#!/bin/bash
set -e

MODEL_LEAF_DIR="/app/Model-Leaf"
MODEL_CLASS_DIR="/app/Model_Classification"

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

mkdir -p "$MODEL_LEAF_DIR" "$MODEL_CLASS_DIR"

download_if_missing "$MODEL_LEAF_DIR/yolo11x_leaf.pt"           "ml_models/yolo11x_leaf.pt"
download_if_missing "$MODEL_CLASS_DIR/R50_Shape_final_V0.keras"  "ml_models/R50_Shape_final_V0.keras"
download_if_missing "$MODEL_CLASS_DIR/R50_Apex_final_V1.keras"   "ml_models/R50_Apex_final_V1.keras"
download_if_missing "$MODEL_CLASS_DIR/R50_Base_final_V1.keras"   "ml_models/R50_Base_final_V1.keras"
download_if_missing "$MODEL_CLASS_DIR/R50_Margin_final_V1.keras" "ml_models/R50_Margin_final_V1.keras"

exec python app.py
