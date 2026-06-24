import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_PATH   = "Model-Leaf/yolo11x_leaf.pt"          # path to your YOLO .pt
IMAGE_PATH   = "Input-Image/Test-Image.png"         # path to input image
OUTPUT_DIR   = Path("output")      # folder to save results
TARGET_CLASS = "leaf"              # class name in your model

ASPECT_W, ASPECT_H = 4, 3         # crop aspect ratio


# ──────────────────────────────────────────────
# PROCESS 1 — Detect & Crop leaf (4:3, no squeeze)
# ──────────────────────────────────────────────
def crop_to_aspect(image: np.ndarray, x1, y1, x2, y2,
                   aspect_w=4, aspect_h=3) -> np.ndarray:
    """
    Expand the bounding box to the target aspect ratio (centered),
    then crop — no resizing / squeezing involved.
    """
    ih, iw = image.shape[:2]

    box_w = x2 - x1
    box_h = y2 - y1

    # Decide which dimension to expand
    if box_w / box_h > aspect_w / aspect_h:
        # Width is the limiting side → expand height
        new_w = box_w
        new_h = int(new_w * aspect_h / aspect_w)
    else:
        # Height is the limiting side → expand width
        new_h = box_h
        new_w = int(new_h * aspect_w / aspect_h)

    # Center the new box over the original bounding box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    nx1 = max(0, cx - new_w // 2)
    ny1 = max(0, cy - new_h // 2)
    nx2 = min(iw, nx1 + new_w)
    ny2 = min(ih, ny1 + new_h)

    # Re-anchor if the box was clamped to image borders
    if nx2 - nx1 < new_w:
        nx1 = max(0, nx2 - new_w)
    if ny2 - ny1 < new_h:
        ny1 = max(0, ny2 - new_h)

    return image[ny1:ny2, nx1:nx2]


def detect_and_crop(model_path: str, image_path: str) -> np.ndarray | None:
    model  = YOLO(model_path)
    image  = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    results = model(image)[0]

    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        if cls_name.lower() == TARGET_CLASS.lower():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"[Process 1] Leaf detected — bbox: ({x1},{y1}) → ({x2},{y2})")
            cropped = crop_to_aspect(image, x1, y1, x2, y2, ASPECT_W, ASPECT_H)
            return cropped

    print("[Process 1] No leaf detected.")
    return None


# ──────────────────────────────────────────────
# PROCESS 2 — Slice cropped image into 4 regions
# ──────────────────────────────────────────────
def slice_leaf(cropped: np.ndarray, output_dir: Path) -> None:
    """
    Slice the cropped leaf image into 4 parts and save them.

    Slices (based on image height H):
        Full     → 0 % – 100 %   (full image)
        Top      → 0 % –  30 %   (leaf tip)
        Bottom   → 70 % – 100 %  (leaf base)
        Middle   → 30 % –  60 %  (leaf edge / mid-section)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    H = cropped.shape[0]

    top_end    = int(H * 0.30)
    mid_start  = int(H * 0.30)
    mid_end    = int(H * 0.60)
    bot_start  = int(H * 0.70)

    slices = {
        "Full-leaf.jpg"   : cropped,                          # 100 % full
        "Top-leaf.jpg"    : cropped[0        : top_end  ],    # 0–30 %
        "Bottom-leaf.jpg" : cropped[bot_start: H        ],    # 70–100 %
        "Middle-leaf.jpg" : cropped[mid_start: mid_end  ],    # 30–60 %
    }

    for filename, region in slices.items():
        if region.size == 0:
            print(f"[Process 2] WARNING — empty region for {filename}, skipped.")
            continue
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), region)
        print(f"[Process 2] Saved → {save_path}  "
              f"(size: {region.shape[1]}×{region.shape[0]})")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # ── Process 1 ──────────────────────────────
    cropped_leaf = detect_and_crop(MODEL_PATH, IMAGE_PATH)

    if cropped_leaf is None:
        print("Stopping — no leaf found.")
        exit(1)

    # (Optional) preview the cropped leaf
    # cv2.imshow("Cropped Leaf", cropped_leaf)
    # cv2.waitKey(0)

    # ── Process 2 ──────────────────────────────
    slice_leaf(cropped_leaf, OUTPUT_DIR)

    print("\nDone! All files saved to:", OUTPUT_DIR.resolve())