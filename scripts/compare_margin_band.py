# scripts/compare_margin_band.py
"""Should the middle band be 0.30-0.60 or 0.30-0.70? — margin head.

Production cuts the middle band at 0.30-0.60 (`src/classify/service.py:206`),
leaving the 0.60-0.70 strip to neither the margin nor the base head. The
question here is whether widening it to 0.30-0.70 predicts better.

**There is no direct answer available in this repo, and that is a finding, not
a limitation of the script.** `Testset-Margin-Shadow-output` holds bands that
are already cut at 0.30-0.60; the 0.60-0.70 rows are in no test set at all
(base bands are 0-0.40, apex bands 0.70-1.0). A 0.30-0.70 band cannot be
reconstructed from labelled data, so its accuracy cannot be measured. The
`compare_top_band.py` trick — trim a wider shipped band down to the narrower
candidate — only runs in the direction of removing rows.

Two things that CAN be measured stand in for it:

  A. Sensitivity (has ground truth). Trim the shipped 0.30-0.60 bands to
     narrower middles and score them. This is the slope of accuracy against
     band extent. A head that shrugs off a 17% trim is a head unlikely to be
     hurt or helped much by a 33% widening; one that collapses is a head whose
     accuracy is tied to seeing exactly the band it trained on.

  B. Production impact (no ground truth). Run the margin head over the 241
     full-frame crops in `Testset-Shape-Shadow-output` sliced both ways and
     count how many predictions FLIP. Flips are the real blast radius of the
     change; the confidence and class-distribution shift say which direction
     the widened band pulls.

    python scripts/compare_margin_band.py
    python scripts/compare_margin_band.py --limit 20
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
TESTSETS = REPO / "Image for Example"
MARGIN_TESTSET = TESTSETS / "Testset-Margin-Shadow-output"
FULLFRAME_TESTSET = TESTSETS / "Testset-Shape-Shadow-output"
DEFAULT_REPORT = REPO / "docs" / "experiments" / f"{date.today():%Y-%m-%d}-margin-band.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SQUARE = 640
PAD = (255, 255, 255)

# The shipped middle band spans this fraction of leaf height.
BAND_TOP, BAND_BOTTOM = 0.30, 0.60
CANDIDATE_BOTTOM = 0.70


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


# JPEG turns the white letterbox into near-white, so an exact ==255 test (what
# compare_top_band.py uses) finds no padding at all and silently returns the
# padded square. Every row whose darkest channel is above this is padding.
WHITE = 245


def recover_band(square):
    """Strip the letterbox padding, returning the band content itself.

    The padding is NOT reliably symmetric in this dataset — measured across the
    80 margin bands, content spans anywhere from 562 to 640 of the 640 rows and
    is not centred — so each edge is found independently. Getting this wrong
    matters: trimming "the top 83%" of a padded square is not the top 83% of the
    band, and the fraction it really keeps varies per image.
    """
    non_white = square.min(axis=2) <= WHITE
    rows = np.flatnonzero(non_white.any(axis=1))
    if rows.size == 0:
        return None
    return square[int(rows[0]):int(rows[-1]) + 1]


def images(folder, limit=None):
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return files[:limit] if limit else files


# ── A. sensitivity to band extent, against the margin ground truth ──────────

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


def score(files, classes, transform, label):
    correct = 0
    scored = 0
    conf_sum = 0.0
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
        pred, conf = service.predict_class(service.margin_model, region, classes)
        scored += 1
        conf_sum += conf
        confusion[(truth, pred)] += 1
        if pred == truth:
            correct += 1
        if i % 40 == 0:
            print(f"    {label}: {i}/{len(files)}")

    acc = (correct / scored * 100) if scored else 0.0
    return {"correct": correct, "scored": scored, "accuracy": acc,
            "mean_conf": (conf_sum / scored) if scored else 0.0,
            "confusion": confusion, "skipped": skipped}


# ── B. production impact on full-frame crops, no ground truth ───────────────

def middle_band(crop, top, bottom):
    """The production slice, verbatim from service.slice_leaf."""
    h = crop.shape[0]
    return crop[int(h * top):int(h * bottom)]


def paired_predictions(files, classes):
    """Predict margin on each full crop under both band bounds."""
    rows = []
    for i, path in enumerate(files, 1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        crop = recover_band(image)
        if crop is None or crop.size == 0:
            continue
        narrow = middle_band(crop, BAND_TOP, BAND_BOTTOM)
        wide = middle_band(crop, BAND_TOP, CANDIDATE_BOTTOM)
        if narrow.size == 0 or wide.size == 0:
            continue
        n_pred, n_conf = service.predict_class(service.margin_model, narrow, classes)
        w_pred, w_conf = service.predict_class(service.margin_model, wide, classes)
        rows.append((path.name, n_pred, n_conf, w_pred, w_conf))
        if i % 40 == 0:
            print(f"    paired: {i}/{len(files)}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not MARGIN_TESTSET.is_dir():
        sys.exit(f"Missing test set: {MARGIN_TESTSET}")

    # Only the margin head is needed. `service.load_models()` also pulls in YOLO
    # and the other three, which this comparison never calls.
    service.margin_model = service._load_classifier(settings.MARGIN_MODEL_PATH)
    classes = settings.MARGIN_CLASSES

    # ── A ──
    band_files = images(MARGIN_TESTSET, args.limit)
    print(f"A. {len(band_files)} labelled margin band(s) from {MARGIN_TESTSET.name}")
    span = BAND_BOTTOM - BAND_TOP
    fractions = [1.0, 0.833, 0.667, 0.5]
    variants = [("untouched (0.30-0.60, as shipped)", lambda img: img)]
    for f in fractions:
        end = BAND_TOP + span * f
        name = ("full_band control (recovered, re-letterboxed)" if f >= 1.0
                else f"0.30-{end:.2f} (top {f:.0%} of band)")
        variants.append((name, trim_to(f)))

    sensitivity = []
    for label, transform in variants:
        print(f"  {label}")
        sensitivity.append((label, score(band_files, classes, transform, label)))

    # ── B ──
    paired = []
    if FULLFRAME_TESTSET.is_dir():
        full_files = images(FULLFRAME_TESTSET, args.limit)
        print(f"B. {len(full_files)} full-frame crop(s) from {FULLFRAME_TESTSET.name}")
        paired = paired_predictions(full_files, classes)

    flips = [r for r in paired if r[1] != r[3]]
    n_dist = Counter(r[1] for r in paired)
    w_dist = Counter(r[3] for r in paired)
    n_conf = (sum(r[2] for r in paired) / len(paired)) if paired else 0.0
    w_conf = (sum(r[4] for r in paired) / len(paired)) if paired else 0.0

    # ── report ────────────────────────────────────────────────────
    L = [
        f"# Middle band 0.30-0.60 vs 0.30-0.70 — margin head — {date.today():%Y-%m-%d}",
        "",
        "## The direct comparison cannot be run",
        "",
        "`Testset-Margin-Shadow-output` is already cut at 0.30-0.60. The 0.60-0.70",
        "strip appears in no test set (base bands are 0-0.40, apex bands 0.70-1.0),",
        "so a labelled 0.30-0.70 band cannot be reconstructed and its accuracy cannot",
        "be measured. Everything below is indirect evidence.",
        "",
        "## A. How much does the margin head care about band extent?",
        "",
        f"- Data: `Image for Example/{MARGIN_TESTSET.name}` ({len(band_files)} labelled bands)",
        "- Each variant un-letterboxes the shipped band, keeps its top fraction, and",
        "  letterboxes back — the same recovery `compare_top_band.py` uses.",
        "- Trimming only shrinks the band. It cannot simulate the widened candidate;",
        "  it measures the slope of accuracy against extent.",
        "",
        "| variant | accuracy | correct / scored | mean confidence |",
        "|---|---|---|---|",
    ]
    for label, r in sensitivity:
        L.append(f"| {label} | {r['accuracy']:.1f}% | {r['correct']} / {r['scored']} "
                 f"| {r['mean_conf']:.1f}% |")

    shipped_acc = sensitivity[0][1]["accuracy"]
    control_acc = sensitivity[1][1]["accuracy"]
    L += [
        "",
        f"Control check: the recovered-and-re-letterboxed band scores {control_acc:.1f}% "
        f"against the untouched {shipped_acc:.1f}%. If those differ, the letterbox",
        "recovery is lossy and the trimmed rows below are suspect.",
        "",
        "### Per-variant confusions",
        "",
    ]
    for label, r in sensitivity:
        parts = [f"{t} → {p}: {n}" for (t, p), n in r["confusion"].most_common() if t != p]
        L.append(f"- **{label}**: " + (" · ".join(parts) if parts else "none"))

    L += [
        "",
        "## B. What the change actually does in production",
        "",
        f"- Data: `Image for Example/{FULLFRAME_TESTSET.name}` "
        f"({len(paired)} full-frame crops, sliced by this script the way "
        "`slice_leaf` slices them)",
        "- These crops carry SHAPE labels, not margin labels, so there is no accuracy",
        "  here. What is countable is how many margin predictions change.",
        "",
        f"- Predictions that flip when the band widens to 0.70: "
        f"**{len(flips)} / {len(paired)}** "
        f"({(len(flips) / len(paired) * 100) if paired else 0:.1f}%)",
        f"- Mean confidence: 0.30-0.60 {n_conf:.1f}% · 0.30-0.70 {w_conf:.1f}%",
        "",
        "| class | predicted at 0.30-0.60 | predicted at 0.30-0.70 |",
        "|---|---|---|",
    ]
    for c in classes:
        L.append(f"| {c} | {n_dist[c]} | {w_dist[c]} |")

    if flips:
        L += ["", "### Flipped crops", "",
              "| file | 0.30-0.60 | 0.30-0.70 |", "|---|---|---|"]
        for name, n_pred, nc, w_pred, wc in flips[:40]:
            L.append(f"| {name} | {n_pred} ({nc:.0f}%) | {w_pred} ({wc:.0f}%) |")
        if len(flips) > 40:
            L.append(f"| … | {len(flips) - 40} more | |")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
