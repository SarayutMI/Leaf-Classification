import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_PATH   = "Model-Leaf/yolo11x_leaf.pt"   # path to YOLO .pt
INPUT_PATH   = Path("/Volumes/SSD_M/LeafClassification_A7/Leaf_Shape/Cordate")            # image file OR folder
OUTPUT_DIR   = Path("/Volumes/SSD_M/LeafClassification_A7/output-Shape-Cordate")                 # root output folder
TARGET_CLASS = "leaf"                         # class name in your model

# Final square output size (all saved images will be this size)
SQUARE_SIZE = 640

# padding color (B, G, R)
PAD_COLOR = (255, 255, 255)

# supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ──────────────────────────────────────────────
# UTIL — Convert any image to square without distortion
# ──────────────────────────────────────────────
def to_square_letterbox(
    image: np.ndarray,
    out_size: int = 640,
    pad_color=(255, 255, 255)
) -> np.ndarray:
    """
    Resize image while keeping aspect ratio, then pad to square (out_size x out_size).
    No stretching -> preserves features.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image with zero width/height.")

    scale = min(out_size / w, out_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((out_size, out_size, 3), pad_color, dtype=np.uint8)
    x = (out_size - nw) // 2
    y = (out_size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def crop_bbox(image: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    """
    Safe bbox crop (clamped to image border).
    """
    ih, iw = image.shape[:2]
    x1 = max(0, min(iw - 1, int(x1)))
    y1 = max(0, min(ih - 1, int(y1)))
    x2 = max(0, min(iw, int(x2)))
    y2 = max(0, min(ih, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)

    return image[y1:y2, x1:x2]


def list_images(input_path: Path) -> list[Path]:
    """
    Return list of image files from a file or folder.
    """
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTS else []

    if input_path.is_dir():
        files = [p for p in sorted(input_path.iterdir())
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        return files

    return []


# ──────────────────────────────────────────────
# PROCESS 1 — Detect & Crop leaf from bbox
# ──────────────────────────────────────────────
def detect_and_crop(model: YOLO, image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[Process 1] Cannot open image: {image_path}")
        return None

    results = model(image)[0]

    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        if cls_name.lower() == TARGET_CLASS.lower():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"[Process 1] {image_path.name} — Leaf detected bbox: ({x1},{y1}) → ({x2},{y2})")
            cropped = crop_bbox(image, x1, y1, x2, y2)
            if cropped.size == 0:
                print(f"[Process 1] {image_path.name} — WARNING invalid crop from bbox.")
                return None
            return cropped

    print(f"[Process 1] {image_path.name} — No leaf detected.")
    return None


# ──────────────────────────────────────────────
# PROCESS 2 — Slice + save into category folders
# ──────────────────────────────────────────────
def ensure_output_folders(root: Path) -> dict[str, Path]:
    """
    Create output subfolders:
      output/Full-leaf, Top-leaf, Middle-leaf, Bottom-leaf
    """
    folders = {
        "Full-leaf": root / "Full-leaf",
        "Top-leaf": root / "Top-leaf",
        "Middle-leaf": root / "Middle-leaf",
        "Bottom-leaf": root / "Bottom-leaf",
    }
    for p in folders.values():
        p.mkdir(parents=True, exist_ok=True)
    return folders


def save_slices(cropped: np.ndarray, src_name: str, output_root: Path) -> None:
    """
    Slice cropped image and save square outputs to category folders.
    Filename includes source name, e.g. source.png -> source.jpg
    """
    H, W = cropped.shape[:2]
    top_end   = int(H * 0.40)
    mid_start = int(H * 0.30)
    mid_end   = int(H * 0.60)
    bot_start = int(H * 0.70)

    slices = {
        "Full-leaf":   cropped,
        "Top-leaf":    cropped[0:top_end, :],
        "Middle-leaf": cropped[mid_start:mid_end, :],
        "Bottom-leaf": cropped[bot_start:H, :],
    }

    out_dirs = ensure_output_folders(output_root)
    stem = Path(src_name).stem
    out_filename = f"{stem}.jpg"

    for label, region in slices.items():
        if region.size == 0:
            print(f"[Process 2] {src_name} — WARNING empty region for {label}, skipped.")
            continue

        square = to_square_letterbox(region, out_size=SQUARE_SIZE, pad_color=PAD_COLOR)
        save_path = out_dirs[label] / out_filename
        cv2.imwrite(str(save_path), square)
        print(f"[Process 2] Saved → {save_path} (square: {square.shape[1]}×{square.shape[0]})")


# ──────────────────────────────────────────────
# MAIN — process file or folder
# ──────────────────────────────────────────────
if __name__ == "__main__":
    image_files = list_images(INPUT_PATH)

    if not image_files:
        print(f"No valid images found in: {INPUT_PATH}")
        exit(1)

    print(f"Found {len(image_files)} image(s). Loading model once...")
    model = YOLO(MODEL_PATH)

    success = 0
    fail = 0

    for img_path in image_files:
        cropped_leaf = detect_and_crop(model, img_path)
        if cropped_leaf is None:
            fail += 1
            continue

        save_slices(cropped_leaf, img_path.name, OUTPUT_DIR)
        success += 1

    print("\nDone!")
    print(f"Processed successfully: {success}")
    print(f"Failed / no leaf:       {fail}")
    print("Output root:", OUTPUT_DIR.resolve())