# Crop aspect comparison — 2026-08-28

- Photos: `Image for Example/Testset-Shape-Shadow-output` (241 files, 241 with a leaf detected)
- Detector: `./Model-Leaf/yolo11s_leaf.pt` at imgsz 640
- Production settings at run time: crop 3:4, IMG_SIZE 256

Detection runs once per photo; every crop variant is taken from the same
bounding box, so the undetected count is shared and cannot favour a variant.

## Shape head — top-1 accuracy (ground truth from filenames)

| crop | preprocess | accuracy | correct / scored |
|---|---|---|---|
| raw_bbox | stretch | 94.6% | 228 / 241 |
| raw_bbox | letterbox | 95.0% | 229 / 241 |
| 4:3 | stretch | 98.3% | 237 / 241 |
| 4:3 | letterbox | 97.9% | 236 / 241 |
| 3:4 *(production today)* | stretch | 96.7% | 233 / 241 |
| 3:4 | letterbox | 97.1% | 234 / 241 |

**Best: `4:3` + `stretch` at 98.3%.**

## Shape head — misclassifications per variant

- `raw_bbox` + `stretch`:

  - Ovate → Sagittate: 6
  - Lanceolate → Sagittate: 3
  - Cordate → Sagittate: 2
  - Ovate → Cordate: 2

- `raw_bbox` + `letterbox`:

  - Lanceolate → Sagittate: 4
  - Ovate → Cordate: 4
  - Ovate → Sagittate: 3
  - Cordate → Sagittate: 1

- `4:3` + `stretch`:

  - Lanceolate → Sagittate: 2
  - Cordate → Sagittate: 1
  - Ovate → Cordate: 1

- `4:3` + `letterbox`:

  - Lanceolate → Sagittate: 3
  - Cordate → Sagittate: 1
  - Ovate → Cordate: 1

- `3:4` + `stretch`:

  - Lanceolate → Sagittate: 3
  - Cordate → Sagittate: 2
  - Ovate → Cordate: 2
  - Ovate → Sagittate: 1

- `3:4` + `letterbox`:

  - Lanceolate → Sagittate: 4
  - Ovate → Cordate: 3

## Apex / base / margin — no ground truth, distribution only

These photos are labelled for shape only. The numbers below are **not**
accuracy: they are how often each head emitted each label. They are here
because the symptom under investigation is a head collapsing onto one
class (apex → Caudate, base → Auriculate); a variant that spreads the
distribution is evidence even without labels.

| crop | preprocess | head | distribution |
|---|---|---|---|
| raw_bbox | stretch | apex | Caudate 181, Acute 55, Obtuse 5 |
| raw_bbox | stretch | base | Auriculate 168, Obtuse 71, Cuneate 2 |
| raw_bbox | stretch | margin | Entire 237, Crenate 4 |
| raw_bbox | letterbox | apex | Caudate 164, Cuspidate 36, Acute 35, Obtuse 6 |
| raw_bbox | letterbox | base | Auriculate 95, Caudate 74, Obtuse 39, Cuneate 33 |
| raw_bbox | letterbox | margin | Entire 172, Crenate 69 |
| 4:3 | stretch | apex | Caudate 209, Acute 29, Obtuse 2, Cuspidate 1 |
| 4:3 | stretch | base | Auriculate 181, Obtuse 52, Cuneate 8 |
| 4:3 | stretch | margin | Entire 240, Crenate 1 |
| 4:3 | letterbox | apex | Caudate 147, Cuspidate 90, Obtuse 4 |
| 4:3 | letterbox | base | Caudate 104, Auriculate 84, Cuneate 32, Obtuse 21 |
| 4:3 | letterbox | margin | Entire 173, Crenate 68 |
| 3:4 | stretch | apex | Caudate 195, Acute 40, Obtuse 6 |
| 3:4 | stretch | base | Auriculate 174, Obtuse 62, Caudate 3, Cuneate 2 |
| 3:4 | stretch | margin | Entire 233, Crenate 8 |
| 3:4 | letterbox | apex | Caudate 190, Cuspidate 25, Acute 21, Obtuse 5 |
| 3:4 | letterbox | base | Auriculate 106, Caudate 67, Cuneate 36, Obtuse 32 |
| 3:4 | letterbox | margin | Entire 177, Crenate 64 |

## Gaps

- Apex, base and margin have no full-frame ground truth in the repo. The
  `Testset-Apex/Base/Margin-Shadow-output` folders are already-sliced band
  crops, so they test the heads but never touch `_crop_to_aspect`. Measuring
  the crop change against those heads needs the source photos they were cut
  from, which live outside the repo (`/Volumes/SSD_M/...`, see
  `Etc/Extractimage-Test-Process-Folder.py:11-12`).
- The band bounds are the production ones (top 0.30). The dataset generator
  used 0.40 for the top band; that divergence is not varied here.
