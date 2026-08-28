# scripts/eval_heads.py
"""Score each classifier head against its own labelled test set.

The `Image for Example/Testset-*-Shadow-output` folders hold the region crops
the heads were trained and tested on — already sliced and letterboxed, so they
are the model's input verbatim. Running them here separates two questions that
the end-to-end API conflates:

    is the head itself accurate?   →  measured directly, below
    is the pipeline feeding it the right region?  →  the cross-evaluation

The cross-evaluation exists because `Etc/Extractimage-Test-Process.py:88-92`
labels the top band as the leaf TIP and the bottom band as the BASE, while
`service.py:244-246` sends `top` to the base head and `bottom` to the apex head.
If those are swapped in production, the apex head scores well here on its own
data and poorly on base data — and vice versa. The numbers say which.

    python scripts/eval_heads.py
    python scripts/eval_heads.py --limit 20      # quick smoke run
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.classify import service
from src.config import settings

REPO = Path(__file__).resolve().parent.parent
TESTSETS = REPO / "Image for Example"
DEFAULT_REPORT = REPO / "docs" / "experiments" / f"{date.today():%Y-%m-%d}-head-accuracy.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# head -> (testset folder, region the API feeds it, class list attribute)
HEADS = {
    "shape":  ("Testset-Shape-Shadow-output",  "full",   "SHAPE_CLASSES"),
    "apex":   ("Testset-Apex-Shadow-output",   "bottom", "APEX_CLASSES"),
    "base":   ("Testset-Base-Shadow-output",   "top",    "BASE_CLASSES"),
    "margin": ("Testset-Margin-Shadow-output", "middle", "MARGIN_CLASSES"),
}

# Heads that share a class vocabulary can be scored on each other's data.
CROSS = [("apex", "base"), ("base", "apex")]


def images(folder: Path, limit=None):
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return files[:limit] if limit else files


def score(model, classes, files, label):
    """Top-1 accuracy over files whose truth label is in `classes`.

    Files labelled with a class this head cannot emit are counted separately —
    scoring them would penalise the head for a vocabulary it never had.
    """
    correct = 0
    scored = 0
    unscorable = Counter()
    confusion = Counter()
    conf_sum = 0.0

    for i, path in enumerate(files, 1):
        truth = path.stem.rsplit("_", 1)[0]
        if truth not in classes:
            unscorable[truth] += 1
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        pred, conf = service.predict_class(model, image, classes)
        scored += 1
        conf_sum += conf
        confusion[(truth, pred)] += 1
        if pred == truth:
            correct += 1
        if i % 40 == 0:
            print(f"    {label}: {i}/{len(files)}")

    acc = (correct / scored * 100) if scored else 0.0
    mean_conf = (conf_sum / scored) if scored else 0.0
    return {"correct": correct, "scored": scored, "accuracy": acc,
            "mean_conf": mean_conf, "confusion": confusion, "unscorable": unscorable}


def per_class(result, classes):
    seen = defaultdict(int)
    hit = defaultdict(int)
    for (truth, pred), n in result["confusion"].items():
        seen[truth] += n
        if truth == pred:
            hit[truth] += n
    return {c: (hit[c], seen[c]) for c in classes if seen[c]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    service.load_models()
    models = {"shape": service.shape_model, "apex": service.apex_model,
              "base": service.base_model, "margin": service.margin_model}

    own = {}
    for head, (folder, region, classes_attr) in HEADS.items():
        path = TESTSETS / folder
        if not path.is_dir():
            print(f"  {head}: {path} missing, skipped")
            continue
        classes = getattr(settings, classes_attr)
        files = images(path, args.limit)
        print(f"  {head}: {len(files)} file(s) from {folder}")
        own[head] = score(models[head], classes, files, head)

    cross = {}
    for head, other in CROSS:
        folder = TESTSETS / HEADS[other][0]
        if not folder.is_dir():
            continue
        classes = getattr(settings, HEADS[head][2])
        files = images(folder, args.limit)
        print(f"  cross: {head} model on {other} data ({len(files)} files)")
        cross[(head, other)] = score(models[head], classes, files, f"{head}<-{other}")

    # ── report ────────────────────────────────────────────────────
    L = [
        f"# Per-head accuracy on the labelled region crops — {date.today():%Y-%m-%d}",
        "",
        "The `Testset-*-Shadow-output` folders are the already-sliced region images",
        "the heads were tested on, so this measures the classifiers alone — no YOLO,",
        "no cropping, no band slicing.",
        "",
        "## Each head on its own test set",
        "",
        "| head | region fed in production | accuracy | correct / scored | mean confidence |",
        "|---|---|---|---|---|",
    ]
    for head in HEADS:
        if head not in own:
            continue
        r = own[head]
        L.append(f"| {head} | `{HEADS[head][1]}` | {r['accuracy']:.1f}% | "
                 f"{r['correct']} / {r['scored']} | {r['mean_conf']:.1f}% |")

    L += ["", "### Per class", ""]
    for head in HEADS:
        if head not in own:
            continue
        classes = getattr(settings, HEADS[head][2])
        pc = per_class(own[head], classes)
        parts = [f"{c} {h}/{n} ({h / n * 100:.0f}%)" for c, (h, n) in pc.items()]
        L.append(f"- **{head}**: " + " · ".join(parts))
        skipped = own[head]["unscorable"]
        if skipped:
            L.append(f"  - not scorable (label outside this head's classes): "
                     + ", ".join(f"{k} {v}" for k, v in skipped.most_common()))
    L.append("")

    L += ["### Confusions", ""]
    for head in HEADS:
        if head not in own:
            continue
        wrong = {k: v for k, v in own[head]["confusion"].items() if k[0] != k[1]}
        if not wrong:
            L.append(f"- **{head}**: none")
            continue
        L.append(f"- **{head}**: " + " · ".join(
            f"{t} → {p}: {n}" for (t, p), n in sorted(wrong.items(), key=lambda kv: -kv[1])))
    L.append("")

    if cross:
        L += [
            "## Cross-evaluation — is the pipeline feeding each head the right band?",
            "",
            "`Etc/Extractimage-Test-Process.py:88-92` calls the top band the leaf TIP and",
            "the bottom band the BASE. `service.py:244-246` sends `top` to the base head",
            "and `bottom` to the apex head. If production has them swapped, each head",
            "does better on the other head's data than on its own.",
            "",
            "| model | data | accuracy | correct / scored |",
            "|---|---|---|---|",
        ]
        for (head, other), r in cross.items():
            L.append(f"| {head} | {other} test set | {r['accuracy']:.1f}% | "
                     f"{r['correct']} / {r['scored']} |")
        L += ["", "Compare each row against the same model's own-data accuracy above.", ""]

    report = "\n".join(L) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print("\n" + report)
    print(f"Written to {args.report}")


if __name__ == "__main__":
    main()
