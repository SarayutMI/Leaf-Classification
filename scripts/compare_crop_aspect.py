# scripts/compare_crop_aspect.py
"""Measure which crop target the classifier heads actually want.

Production expands the YOLO bounding box to a fixed aspect ratio before
`slice_leaf()` cuts it into bands. That target was 4:3 (landscape) and is now
3:4 (portrait) — but the dataset the heads were trained on was produced by
`Etc/Extractimage-Test-Process-Folder.py`, which forces **no** aspect at all:
it crops the raw bbox. So neither 4:3 nor 3:4 is training parity, and the
choice has to be measured rather than argued.

The same script also varies the two other divergences between that generator
and production, so their effect can be read off the same table:

    generator                       production
    raw bbox                        _crop_to_aspect(W, H)
    top band = 0.40 of height       top band = 0.30
    letterbox to 640, white pad     cv2.resize stretch to IMG_SIZE

Nothing in src/ is modified — the variants are applied around the real
functions, so the numbers describe the deployed code path.

    python scripts/compare_crop_aspect.py
    python scripts/compare_crop_aspect.py --limit 40      # quick smoke run
    python scripts/compare_crop_aspect.py --photos "Image for Example/Testset-Shape-Shadow-output"

Writes a markdown report next to the default output path and prints it.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.classify import service
from src.config import settings

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PHOTOS = REPO / "Image for Example" / "Testset-Shape-Shadow-output"
DEFAULT_REPORT = REPO / "docs" / "experiments" / f"{date.today():%Y-%m-%d}-crop-aspect.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# git-lfs pointer files are a few hundred bytes; a real model is megabytes.
LFS_POINTER_MAX_BYTES = 1000

CROP_VARIANTS = ["raw_bbox", "4:3", "3:4"]
PREPROCESS_VARIANTS = ["stretch", "letterbox"]


# ── variants ──────────────────────────────────────────────────────────

def crop_variant(image, box, variant):
    x1, y1, x2, y2 = box
    if variant == "raw_bbox":
        # Matches Extractimage-Test-Process-Folder.py:52-65 — clamp, no expand.
        ih, iw = image.shape[:2]
        x1, y1 = max(0, min(iw - 1, x1)), max(0, min(ih - 1, y1))
        x2, y2 = max(0, min(iw, x2)), max(0, min(ih, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    w, h = (int(n) for n in variant.split(":"))
    return service._crop_to_aspect(image, x1, y1, x2, y2, w, h)


def to_square_letterbox(image, out_size=640, pad_color=(255, 255, 255)):
    """Port of Etc/Extractimage-Test-Process-Folder.py:28-49 — the resize the
    training images actually went through."""
    h, w = image.shape[:2]
    scale = min(out_size / w, out_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_size, out_size, 3), pad_color, dtype=np.uint8)
    x, y = (out_size - nw) // 2, (out_size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def predict(model, region, classes, preprocess_variant):
    if preprocess_variant == "letterbox":
        region = to_square_letterbox(region)
    return service.predict_class(model, region, classes)


# ── run ───────────────────────────────────────────────────────────────

def detect_box(image):
    results = service.yolo_model(image, imgsz=settings.YOLO_IMGSZ, verbose=False)[0]
    for box in results.boxes:
        if service.yolo_model.names[int(box.cls)].lower() == settings.TARGET_CLASS.lower():
            return tuple(int(v) for v in box.xyxy[0])
    return None


def check_models_present():
    required = [
        settings.YOLO_MODEL_PATH,
        settings.SHAPE_MODEL_PATH, settings.APEX_MODEL_PATH,
        settings.BASE_MODEL_PATH, settings.MARGIN_MODEL_PATH,
    ]
    problems = []
    for path in required:
        p = Path(path)
        tflite = p.with_suffix(".tflite")
        if p.suffix == ".keras" and tflite.exists() and tflite.stat().st_size > LFS_POINTER_MAX_BYTES:
            continue  # service._load_classifier will use the .tflite sibling
        if not p.exists():
            problems.append(f"{path} — missing")
        elif p.stat().st_size <= LFS_POINTER_MAX_BYTES:
            problems.append(f"{path} — git-lfs pointer, run: git lfs pull --include '{path}'")
    if problems:
        sys.exit("Cannot run — model artifacts unusable:\n  " + "\n  ".join(problems))


def run(photos_dir: Path, limit: int | None):
    files = sorted(p for p in photos_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit:
        files = files[:limit]
    if not files:
        sys.exit(f"No images found in {photos_dir}")

    correct = Counter()          # (crop, preprocess) -> n correct shape
    confusion = defaultdict(Counter)   # (crop, preprocess) -> (true, pred)
    other_heads = defaultdict(Counter)  # (crop, preprocess, head) -> label
    scored = 0
    undetected = []

    print(f"{len(files)} photo(s) from {photos_dir}\n")
    for i, path in enumerate(files, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  [{i}/{len(files)}] {path.name}: unreadable, skipped")
            continue

        box = detect_box(image)
        if box is None:
            undetected.append(path.name)
            print(f"  [{i}/{len(files)}] {path.name}: no leaf detected")
            continue

        truth = path.stem.rsplit("_", 1)[0]   # same convention as Test-Model.py:46-47
        scored += 1

        for crop_name in CROP_VARIANTS:
            cropped = crop_variant(image, box, crop_name)
            if cropped is None or cropped.size == 0:
                continue
            regions = service.slice_leaf(cropped)
            for pre in PREPROCESS_VARIANTS:
                key = (crop_name, pre)
                shape_label, _ = predict(service.shape_model, regions["full"],
                                         settings.SHAPE_CLASSES, pre)
                confusion[key][(truth, shape_label)] += 1
                if shape_label == truth:
                    correct[key] += 1
                # No ground truth for these — distribution only.
                for head, region, classes, model in (
                    ("base", "top", settings.BASE_CLASSES, service.base_model),
                    ("margin", "middle", settings.MARGIN_CLASSES, service.margin_model),
                    ("apex", "bottom", settings.APEX_CLASSES, service.apex_model),
                ):
                    label, _ = predict(model, regions[region], classes, pre)
                    other_heads[(crop_name, pre, head)][label] += 1

        if i % 20 == 0:
            print(f"  [{i}/{len(files)}] ...")

    return correct, confusion, other_heads, scored, undetected


# ── report ────────────────────────────────────────────────────────────

def build_report(correct, confusion, other_heads, scored, undetected, photos_dir, files_seen):
    lines = [
        f"# Crop aspect comparison — {date.today():%Y-%m-%d}",
        "",
        f"- Photos: `{photos_dir.relative_to(REPO)}` ({files_seen} files, {scored} with a leaf detected)",
        f"- Detector: `{settings.YOLO_MODEL_PATH}` at imgsz {settings.YOLO_IMGSZ}",
        f"- Production settings at run time: crop {settings.ASPECT_W}:{settings.ASPECT_H}, "
        f"IMG_SIZE {settings.IMG_SIZE}",
        "",
        "Detection runs once per photo; every crop variant is taken from the same",
        "bounding box, so the undetected count is shared and cannot favour a variant.",
        "",
        "## Shape head — top-1 accuracy (ground truth from filenames)",
        "",
        "| crop | preprocess | accuracy | correct / scored |",
        "|---|---|---|---|",
    ]
    ranked = []
    for crop_name in CROP_VARIANTS:
        for pre in PREPROCESS_VARIANTS:
            key = (crop_name, pre)
            n = correct[key]
            acc = (n / scored * 100) if scored else 0.0
            ranked.append((acc, crop_name, pre))
            note = " *(production today)*" if crop_name == f"{settings.ASPECT_W}:{settings.ASPECT_H}" \
                                              and pre == "stretch" else ""
            lines.append(f"| {crop_name}{note} | {pre} | {acc:.1f}% | {n} / {scored} |")

    ranked.sort(reverse=True)
    if ranked:
        acc, crop_name, pre = ranked[0]
        lines += ["", f"**Best: `{crop_name}` + `{pre}` at {acc:.1f}%.**", ""]

    lines += ["## Shape head — misclassifications per variant", ""]
    for crop_name in CROP_VARIANTS:
        for pre in PREPROCESS_VARIANTS:
            wrong = {k: v for k, v in confusion[(crop_name, pre)].items() if k[0] != k[1]}
            if not wrong:
                lines += [f"- `{crop_name}` + `{pre}`: none", ""]
                continue
            lines += [f"- `{crop_name}` + `{pre}`:", ""]
            for (truth, pred), n in sorted(wrong.items(), key=lambda kv: -kv[1]):
                lines.append(f"  - {truth} → {pred}: {n}")
            lines.append("")

    lines += [
        "## Apex / base / margin — no ground truth, distribution only",
        "",
        "These photos are labelled for shape only. The numbers below are **not**",
        "accuracy: they are how often each head emitted each label. They are here",
        "because the symptom under investigation is a head collapsing onto one",
        "class (apex → Caudate, base → Auriculate); a variant that spreads the",
        "distribution is evidence even without labels.",
        "",
        "| crop | preprocess | head | distribution |",
        "|---|---|---|---|",
    ]
    for crop_name in CROP_VARIANTS:
        for pre in PREPROCESS_VARIANTS:
            for head in ("apex", "base", "margin"):
                dist = other_heads[(crop_name, pre, head)]
                shown = ", ".join(f"{k} {v}" for k, v in dist.most_common())
                lines.append(f"| {crop_name} | {pre} | {head} | {shown or '—'} |")

    lines += [
        "",
        "## Gaps",
        "",
        "- Apex, base and margin have no full-frame ground truth in the repo. The",
        "  `Testset-Apex/Base/Margin-Shadow-output` folders are already-sliced band",
        "  crops, so they test the heads but never touch `_crop_to_aspect`. Measuring",
        "  the crop change against those heads needs the source photos they were cut",
        "  from, which live outside the repo (`/Volumes/SSD_M/...`, see",
        "  `Etc/Extractimage-Test-Process-Folder.py:11-12`).",
        "- The band bounds are the production ones (top 0.30). The dataset generator",
        "  used 0.40 for the top band; that divergence is not varied here.",
    ]
    if undetected:
        lines += ["", f"- YOLO found no leaf in {len(undetected)} photo(s): "
                      + ", ".join(undetected[:10]) + ("..." if len(undetected) > 10 else "")]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", type=Path, default=DEFAULT_PHOTOS,
                    help="folder of full-frame labelled photos (<Class>_<n>.jpg)")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--limit", type=int, default=None, help="only the first N photos")
    args = ap.parse_args()

    if not args.photos.is_dir():
        sys.exit(f"Not a directory: {args.photos}")

    check_models_present()
    service.load_models()

    files_seen = len([p for p in sorted(args.photos.iterdir())
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if args.limit:
        files_seen = min(files_seen, args.limit)

    correct, confusion, other_heads, scored, undetected = run(args.photos, args.limit)
    report = build_report(correct, confusion, other_heads, scored, undetected,
                          args.photos, files_seen)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print("\n" + report)
    print(f"Written to {args.report}")


if __name__ == "__main__":
    main()
