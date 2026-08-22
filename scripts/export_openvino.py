# scripts/export_openvino.py
"""
Export a trained YOLO .pt checkpoint to OpenVINO for fast CPU inference.

Run inside the API environment (same deps as the server):

    python scripts/export_openvino.py --model Model-Leaf/yolo11n_leaf.pt

This produces `Model-Leaf/yolo11n_leaf_openvino_model/` — point
YOLO_MODEL_PATH (env or src/config.py) at that folder and restart the API.
Note: imgsz is fixed at export time; inference must use the same size.
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the .pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default 640)")
    args = parser.parse_args()

    model = YOLO(args.model)
    out_path = model.export(format="openvino", imgsz=args.imgsz)
    print(f"Exported: {out_path}")
    print("Set YOLO_MODEL_PATH to that folder and restart the API.")


if __name__ == "__main__":
    main()
