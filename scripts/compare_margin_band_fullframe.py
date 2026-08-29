# scripts/compare_margin_band_fullframe.py
"""Middle band 0.30-0.55 vs 0.30-0.60 vs 0.30-0.70, scored on full-frame crops.

`Testset-Shape-Shadow-output` carries shape labels, not margin labels, so
`compare_margin_band.py` could only count prediction flips there. This script
uses the ground truth the user supplied instead:

    every leaf in that folder EXCEPT the Ovate ones has an entire margin.

That makes 169 of the 241 crops (Cordate, Lanceolate, Sagittate) labelled
`Entire`, and the margin head can be scored end-to-end on the band it is
actually fed in production. The 72 Ovate crops have no supplied truth — they
are reported separately as a distribution, never mixed into the accuracy.

One caveat the numbers cannot escape: an all-`Entire` truth set only measures
false Crenate. A band that made the head answer `Entire` for everything would
score 100% here and be worthless. Read this against the Crenate recall in
`docs/experiments/2026-08-29-margin-band.md` (section A, real two-class truth),
not on its own.

    python scripts/compare_margin_band_fullframe.py
    python scripts/compare_margin_band_fullframe.py --limit 20
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
TESTSET = REPO / "Image for Example" / "Testset-Shape-Shadow-output"
DEFAULT_REPORT = REPO / "docs" / "experiments" / f"{date.today():%Y-%m-%d}-margin-band-fullframe.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Shape classes whose leaves the user states have an entire margin.
ENTIRE_SHAPES = {"Cordate", "Lanceolate", "Sagittate"}
UNLABELLED_SHAPES = {"Ovate"}

BAND_TOP = 0.30
CANDIDATES = [0.55, 0.60, 0.70]
SHIPPED = 0.60


# JPEG turns the white letterbox into near-white, so an exact ==255 test finds
# no padding and silently returns the padded square. Every row whose darkest
# channel is above this is padding.
WHITE = 245


def recover_crop(square):
    """Strip the letterbox padding, returning the leaf crop itself.

    The shape testset is letterboxed onto a 640 white square. Slicing the square
    directly would hand the head rows of padding instead of leaf, and would put
    the 0.30 and 0.60 bounds somewhere other than 0.30 and 0.60 of the leaf. The
    padding is not reliably symmetric, so each edge is found independently.
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


def middle_band(crop, bottom):
    """The production slice from service.slice_leaf, with the bottom bound open."""
    h = crop.shape[0]
    return crop[int(h * BAND_TOP):int(h * bottom)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not TESTSET.is_dir():
        sys.exit(f"Missing test set: {TESTSET}")

    # Only the margin head is needed; load_models() also pulls in YOLO and the
    # other three heads, none of which this comparison calls.
    service.margin_model = service._load_classifier(settings.MARGIN_MODEL_PATH)
    classes = settings.MARGIN_CLASSES

    files = images(TESTSET, args.limit)
    print(f"{len(files)} crop(s) from {TESTSET.name}")

    # per bound: predictions keyed by filename, so every bound is scored on
    # exactly the same set of crops.
    preds = {b: {} for b in CANDIDATES}
    shapes = {}

    for i, path in enumerate(files, 1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        crop = recover_crop(image)
        if crop is None or crop.size == 0:
            continue
        shape = path.stem.rsplit("_", 1)[0]
        for bottom in CANDIDATES:
            band = middle_band(crop, bottom)
            if band.size == 0:
                continue
            preds[bottom][path.name] = service.predict_class(
                service.margin_model, band, classes)
        shapes[path.name] = shape
        if i % 40 == 0:
            print(f"  {i}/{len(files)}")

    labelled = [n for n, s in shapes.items() if s in ENTIRE_SHAPES]
    ovate = [n for n, s in shapes.items() if s in UNLABELLED_SHAPES]

    def summarise(bottom):
        p = preds[bottom]
        correct = [n for n in labelled if p[n][0] == "Entire"]
        wrong = [n for n in labelled if p[n][0] != "Entire"]
        conf = [p[n][1] for n in labelled]
        by_shape = defaultdict(lambda: [0, 0])
        for n in labelled:
            by_shape[shapes[n]][1] += 1
            if p[n][0] == "Entire":
                by_shape[shapes[n]][0] += 1
        return {
            "accuracy": len(correct) / len(labelled) * 100 if labelled else 0.0,
            "correct": len(correct), "scored": len(labelled),
            "mean_conf": sum(conf) / len(conf) if conf else 0.0,
            "wrong": wrong,
            "by_shape": dict(by_shape),
            "ovate": Counter(p[n][0] for n in ovate),
            "ovate_conf": (sum(p[n][1] for n in ovate) / len(ovate)) if ovate else 0.0,
        }

    results = {b: summarise(b) for b in CANDIDATES}

    L = [
        f"# Middle band 0.30-0.55 vs 0.30-0.60 vs 0.30-0.70 — margin head — {date.today():%Y-%m-%d}",
        "",
        f"- Data: `Image for Example/{TESTSET.name}` ({len(shapes)} crops)",
        "- Ground truth (supplied): every non-Ovate leaf has an **Entire** margin —",
        f"  {len(labelled)} crops across {', '.join(sorted(ENTIRE_SHAPES))}.",
        f"- The {len(ovate)} Ovate crops carry no supplied margin truth and are excluded",
        "  from accuracy; their prediction split is reported separately below.",
        "- Each crop is un-letterboxed, then sliced `0.30-x` the way `slice_leaf` does.",
        "",
        "> An all-`Entire` truth set can only catch false Crenate. A band that answered",
        "> `Entire` for every input would score 100% here. Read these numbers together",
        "> with the two-class Crenate recall in `2026-08-29-margin-band.md`.",
        "",
        "## Accuracy on the 169 Entire-truth crops",
        "",
        "| band | accuracy | correct / scored | mean confidence | false Crenate |",
        "|---|---|---|---|---|",
    ]
    for b in CANDIDATES:
        r = results[b]
        tag = " (shipped)" if b == SHIPPED else ""
        L.append(f"| 0.30-{b:.2f}{tag} | {r['accuracy']:.1f}% | {r['correct']} / {r['scored']} "
                 f"| {r['mean_conf']:.1f}% | {len(r['wrong'])} |")

    L += ["", "### Per shape class", "",
          "| band | " + " | ".join(sorted(ENTIRE_SHAPES)) + " |",
          "|---|" + "---|" * len(ENTIRE_SHAPES)]
    for b in CANDIDATES:
        cells = []
        for s in sorted(ENTIRE_SHAPES):
            hit, seen = results[b]["by_shape"].get(s, (0, 0))
            cells.append(f"{hit}/{seen}" + (f" ({hit / seen * 100:.0f}%)" if seen else ""))
        L.append(f"| 0.30-{b:.2f} | " + " | ".join(cells) + " |")

    L += ["", "### Crops predicted Crenate against an Entire truth", ""]
    any_wrong = False
    for b in CANDIDATES:
        w = results[b]["wrong"]
        if w:
            any_wrong = True
        detail = ", ".join(f"{n} ({preds[b][n][1]:.0f}%)" for n in sorted(w)) or "none"
        L.append(f"- **0.30-{b:.2f}**: {detail}")
    if not any_wrong:
        L.append("")
        L.append("No band produces a false Crenate on this set, so accuracy alone cannot")
        L.append("separate them — the tie breaks on confidence and on Ovate behaviour.")

    L += ["", "## Ovate crops — no supplied truth, distribution only", "",
          "| band | " + " | ".join(classes) + " | mean confidence |",
          "|---|" + "---|" * (len(classes) + 1)]
    for b in CANDIDATES:
        r = results[b]
        cells = " | ".join(str(r["ovate"][c]) for c in classes)
        L.append(f"| 0.30-{b:.2f} | {cells} | {r['ovate_conf']:.1f}% |")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
