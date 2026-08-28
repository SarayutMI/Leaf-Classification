# Per-head accuracy on the labelled region crops — 2026-08-28

The `Testset-*-Shadow-output` folders are the already-sliced region images
the heads were tested on, so this measures the classifiers alone — no YOLO,
no cropping, no band slicing.

## Each head on its own test set

| head | region fed in production | accuracy | correct / scored | mean confidence |
|---|---|---|---|---|
| shape | `full` | 98.3% | 237 / 241 | 98.8% |
| apex | `bottom` | 86.2% | 138 / 160 | 95.3% |
| base | `top` | 86.2% | 138 / 160 | 92.9% |
| margin | `middle` | 88.8% | 71 / 80 | 95.1% |

### Per class

- **shape**: Cordate 40/40 (100%) · Lanceolate 37/40 (92%) · Ovate 71/72 (99%) · Sagittate 89/89 (100%)
- **apex**: Acute 33/40 (82%) · Caudate 39/40 (98%) · Cuspidate 27/40 (68%) · Obtuse 39/40 (98%)
- **base**: Auriculate 40/40 (100%) · Caudate 31/40 (78%) · Cuneate 35/40 (88%) · Obtuse 32/40 (80%)
- **margin**: Crenate 31/40 (78%) · Entire 40/40 (100%)

### Confusions

- **shape**: Lanceolate → Sagittate: 3 · Ovate → Cordate: 1
- **apex**: Cuspidate → Caudate: 13 · Acute → Caudate: 7 · Caudate → Acute: 1 · Obtuse → Caudate: 1
- **base**: Caudate → Auriculate: 8 · Obtuse → Auriculate: 8 · Cuneate → Auriculate: 4 · Caudate → Cuneate: 1 · Cuneate → Caudate: 1
- **margin**: Crenate → Entire: 9

## Cross-evaluation — is the pipeline feeding each head the right band?

`Etc/Extractimage-Test-Process.py:88-92` calls the top band the leaf TIP and
the bottom band the BASE. `service.py:244-246` sends `top` to the base head
and `bottom` to the apex head. If production has them swapped, each head
does better on the other head's data than on its own.

| model | data | accuracy | correct / scored |
|---|---|---|---|
| apex | base test set | 42.5% | 34 / 80 |
| base | apex test set | 3.8% | 3 / 80 |

Compare each row against the same model's own-data accuracy above.

