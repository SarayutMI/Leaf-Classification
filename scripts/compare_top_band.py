# scripts/compare_top_band.py
"""Does the base head want a 0.30 or a 0.40 top band?

The two extraction scripts disagree. `Etc/Extractimage-Test-Process-Folder.py:134`
— the batch script that produced the dataset — cuts the top band at 0.40 of the
crop height. `Etc/Extractimage-Test-Process.py:98` and production cut it at 0.30,
and `tests/classify/test_slice_leaf.py` pins 0.30 citing the latter. Only the top
band differs; middle (0.30-0.60) and bottom (0.70-1.0) are identical in both, so
this question touches the base head alone.

There is no full-frame photo set with base labels in the repo, so the comparison
is run the other way round: `Testset-Base-Shadow-output` already holds 0.40 bands
(letterboxed onto a 640 white square). Undo the letterbox, keep the top 75% of the
band — 0.75 x 0.40 = 0.30 of the original leaf height — and letterbox it back.
That is what the same photo would have produced under a 0.30 cut.

A `full_band` control re-letterboxes the recovered band without trimming. It must
land on the same accuracy as the untouched files; if it does not, the letterbox
recovery is lossy and every number here is suspect. Check it first.

    python scripts/compare_top_band.py
    python scripts/compare_top_band.py --limit 20
"""
import argparse
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.classify import service
from src.config import settings

REPO = Path(__file__).resolve().parent.parent
BASE_TESTSET = REPO / "Image for Example" / "Testset-Base-Shadow-output"
DEFAULT_REPORT = REPO / "docs" / "experiments" / f"{date.today():%Y-%m-%d}-top-band.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SQUARE = 640
PAD = (255, 255, 255)

# The dataset bands are 0-0.40 of leaf height. Keeping this fraction of one
# yields the band a 0.30 cut would have produced: 0.30 / 0.40.
GENERATED_TOP = 0.40
PRODUCTION_TOP = 0.30


def to_square_letterbox(image, out_size=SQUARE, pad_color=PAD):
    """Port of Etc/Extractimage-Test-Process-Folder.py:28-49."""
    h, w = image.shape[:2]
    scale = min(out_size / w, out_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_size, out_size, 3), pad_color, dtype=np.uint8)
    x, y = (out_size - nw) // 2, (out_size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def recover_band(square):
    """Strip the white letterbox padding, returning the band itself.

    The bands are wider than tall, so the letterbox scaled them to the full
    width and padded top and bottom symmetrically. Symmetry is what makes the
    recovery safe: a band whose own top rows happen to be pure white would fool
    a scan from one side, so both sides are scanned and the smaller pad wins.
    """
    non_white = ~np.all(square == 255, axis=(1, 2))
    rows = np.flatnonzero(non_white)
    if rows.size == 0:
        return None
    pad = min(int(rows[0]), int(square.shape[0] - 1 - rows[-1]))
    if pad <= 0:
        return square
    return square[pad:square.shape[0] - pad]


def images(folder, limit=None):
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return files[:limit] if limit else files


def score(files, classes, transform, label):
    correct = 0
    scored = 0
    confusion = Counter()
    skipped = 0

    for i, path in enumerate(files, 1):
        truth = path.stem.rsplit("_", 1)[0]
        if truth not in classes:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        region = transform(image)
        if region is None or region.size == 0:
            skipped += 1
            continue
        pred, _ = service.predict_class(service.base_model, region, classes)
        scored += 1
        confusion[(truth, pred)] += 1
        if pred == truth:
            correct += 1
        if i % 40 == 0:
            print(f"    {label}: {i}/{len(files)}")

    acc = (correct / scored * 100) if scored else 0.0
    return {"correct": correct, "scored": scored, "accuracy": acc,
            "confusion": confusion, "skipped": skipped}


def trim_to(fraction):
    """Keep the top `fraction` of the recovered band, then letterbox back."""
    def transform(square):
        band = recover_band(square)
        if band is None:
            return None
        if fraction >= 1.0:
            return to_square_letterbox(band)
        keep = max(1, int(round(band.shape[0] * fraction)))
        return to_square_letterbox(band[0:keep])
    return transform


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not BASE_TESTSET.is_dir():
        sys.exit(f"Missing test set: {BASE_TESTSET}")

    service.load_models()
    classes = settings.BASE_CLASSES
    files = images(BASE_TESTSET, args.limit)
    print(f"{len(files)} base band(s) from {BASE_TESTSET.name}\n")

    ratio = PRODUCTION_TOP / GENERATED_TOP
    variants = [
        ("untouched (0.40 band, as shipped)", lambda img: img),
        ("full_band control (recovered, re-letterboxed)", trim_to(1.0)),
        (f"trimmed to 0.30 of leaf height (top {ratio:.0%} of band)", trim_to(ratio)),
    ]

    results = []
    for label, transform in variants:
        print(f"  {label}")
        results.append((label, score(files, classes, transform, label)))

    L = [
        f"# Top band 0.30 vs 0.40 — base head — {date.today():%Y-%m-%d}",
        "",
        f"- Data: `Image for Example/{BASE_TESTSET.name}` ({len(files)} labelled bands)",
        "- Only the top band differs between the two extraction scripts; middle and",
        "  bottom are identical, so this affects the base head alone.",
        "- The shipped bands are 0.40 of leaf height. The 0.30 variant is produced by",
        f"  un-letterboxing each band, keeping its top {ratio:.0%}, and letterboxing back.",
        "",
        "| variant | accuracy | correct / scored |",
        "|---|---|---|",
    ]
    for label, r in results:
        L.append(f"| {label} | {r['accuracy']:.1f}% | {r['correct']} / {r['scored']} |")

    control = results[1][1]["accuracy"]
    shipped = results[0][1]["accuracy"]
    L += [
        "",
        f"**Read the control first.** The recovery round-trip scores {control:.1f}% against "
        f"{shipped:.1f}% for the untouched files. A gap here is loss introduced by the "
        "reconstruction, and the 0.30 row has to be read against the control, not against "
        "the untouched row.",
        "",
        "## Confusions",
        "",
    ]
    for label, r in results:
        wrong = {k: v for k, v in r["confusion"].items() if k[0] != k[1]}
        line = " · ".join(f"{t} → {p}: {n}"
                          for (t, p), n in sorted(wrong.items(), key=lambda kv: -kv[1]))
        L.append(f"- **{label}**: {line or 'none'}")
        if r["skipped"]:
            L.append(f"  - {r['skipped']} file(s) skipped (band could not be recovered)")

    L += [
        "",
        "## Caveat",
        "",
        "This trims an already-downscaled band rather than re-slicing the source photo,",
        "so the 0.30 variant carries the resolution the 0.40 band was saved at. It answers",
        "which framing the head prefers, not what a full-resolution 0.30 pipeline would",
        "score. Re-running from the source photos on `/Volumes/SSD_M/...` would settle that.",
    ]

    report = "\n".join(L) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print("\n" + report)
    print(f"Written to {args.report}")


if __name__ == "__main__":
    main()
