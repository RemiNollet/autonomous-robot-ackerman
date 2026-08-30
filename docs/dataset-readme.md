# Dataset — M1

**Location:** `data/dataset_v0/` (git-ignored — regenerate locally, see below)
**Status:** Generated, split, mirror-augmented, and validated programmatically. See [`docs/decisions.md`](decisions.md) ADR-7 through ADR-10 for the defects this process caught before training started.

---

## 1. What it is

4000 synthetic camera frames from a vehicle-mounted MuJoCo camera on `REFERENCE_TRACK` (a 45.13 m closed loop, see [`perception/dataset/track_definitions.py`](../perception/dataset/track_definitions.py)) — 2000 rendered poses plus their horizontally-mirrored twin (ADR-10) — each labeled with the ground-truth `/lane_state` the vehicle would have seen from that pose — the contract defined in [`docs/lane-state-contract.md`](lane-state-contract.md).

| | |
|---|---|
| Total samples | 4000 (2000 rendered + 2000 mirrored) |
| Images | `data/dataset_v0/images/img_NNNNN.png` + `img_NNNNN_mirror.png`, 320×240 RGB, ~40 MB total |
| Labels | `data/dataset_v0/labels.csv`, ~610 KB |
| Valid (`valid=True`) | 3600 (90%) — vehicle in-lane, lane visible ahead |
| Invalid (`valid=False`) | 400 (10%) — vehicle far off track, no lane marking anywhere in frame (confidence-head negatives) |
| Split | train 2842 (71.1%) / val 772 (19.3%) / test 386 (9.7%) |
| Curvature sign | exactly symmetric: 403 samples at each of κ=±1/3, 599 at each of κ=±1/5 (ADR-10) |

## 2. Label schema (`labels.csv`)

| Column | Meaning |
|---|---|
| `filename` | image file in `images/` |
| `source_filename` | for a mirrored row, the base row it was flipped from; empty for base rows |
| `x`, `y`, `heading` | vehicle pose used to render the frame (world frame) — a mirror row carries its **source's** pose, since there is no mirrored world to place a vehicle in; see ADR-10 |
| `s` | arc length of the projection point on the centerline |
| `lateral_error`, `heading_error`, `curvature` | the `/lane_state` ground truth at that pose — see the contract for sign conventions. Negated (not recomputed) for mirror rows |
| `confidence`, `valid` | confidence-head training target; `valid = confidence >= 0.5` |
| `split` | `train` / `val` / `test` — identical between a mirror row and its source, by construction |
| `mirrored` | `True` for the horizontally-flipped augmented rows, `False` for the rendered originals |

## 3. Distribution

![label distributions](dataset/label_distributions.png)

Curvature takes five signed values on this track (κ ∈ {-1/3, -1/5, 0, 1/5, 1/3}), and every one is present in every split by construction (ADR-9 for the per-split stratification, ADR-10 for the sign symmetry) — verified by `test_every_curvature_bin_present_in_every_split` and `test_dataset_doubles_and_covers_both_curvature_signs`. `lateral_error` and `heading_error` are close to uniform across their declared envelopes for every curvature bin (no bias toward straight-line samples). Negative samples average 6.10 m off centerline (min 2.03 m, max 10.0 m) — deliberately easy negatives; see ADR-8 for why.

**Read before M2:** `REFERENCE_TRACK` itself only ever turns left (`track_definitions.py`) — the negative-curvature half of the dataset exists entirely because of mirror augmentation, not because the vehicle ever drove a right turn. If the track geometry changes, re-check this bar chart before assuming right-turn coverage still holds.

Regenerate this figure and its console breakdown (per-split, per-curvature-bin counts) with:

```bash
python3 perception/dataset/plot_label_distributions.py
```

## 4. Normalization statistics

Per-channel mean/std, computed over the **train split only** (val/test stay unseen at the pixel-statistics level, not just at the label level), written to [`perception/dataset/normalization_stats.json`](../perception/dataset/normalization_stats.json):

```json
{
  "mean": [0.4298, 0.4899, 0.5642],
  "std":  [0.3188, 0.3696, 0.4291]
}
```

(RGB, `[0, 1]` scale.) Both training (M2) and `perception_node`'s inference-time preprocessing must load these from the same file rather than recomputing or hard-coding them — that consistency is NFR-8. Numerically identical to the pre-mirror-augmentation values: a horizontal flip permutes pixel positions, not pixel values, so per-channel mean/std over the (now doubled) train split is unaffected — a useful sanity check in itself.

Regenerate with:

```bash
python3 perception/dataset/compute_normalization_stats.py
```

## 5. Reproducing the dataset exactly

Generation is seeded (`SEED = 42`) and deterministic — see `test_generation_is_deterministic`. From the repo root, with the sim venv active:

```bash
source venv/bin/activate
python3 perception/dataset/generate_dataset.py       # labels.csv: 2000 rendered + 2000 mirrored rows (headless, ~instant)
python3 perception/dataset/render_dataset_images.py  # images/: renders the 2000 base poses (MuJoCo, Mac-only, ~25s), then mirrors them (~10s)
python3 perception/dataset/plot_label_distributions.py
python3 perception/dataset/compute_normalization_stats.py
```

`generate_dataset.py` has no MuJoCo dependency and is fully covered by `tests/test_generate_dataset.py`; `render_dataset_images.py` only turns already-decided poses into pixels (or, for a mirror row, flips an already-rendered one) and carries no sampling logic of its own, so a rendering bug can never be confused with a labeling bug (see the module docstrings).

## 6. Validation performed

- `python3 -m pytest tests/ -q` — 34/34 passing, including label envelope, camera-visibility ground truth (`lane_is_visible`/`any_lane_visible` against every generated pose, not a resample), split-purity (`zone_for_s`), curvature-coverage-per-split, and mirror-augmentation regression tests (label negation, split preservation, left/right balance).
- Every base row's stored label recomputed independently from `(x, y, heading)` and diffed against the CSV (agrees to 1e-9).
- Every mirror row's label checked against its source row's negation directly from the CSV; a sample of mirror images checked **pixel-for-pixel** against a manual horizontal flip of their source image on disk (exact match, not just label agreement).
- Every image file checked for correct dimensions and non-degenerate pixel content; CSV rows and image files verified 1:1.
- No cross-split leakage: split is a pure function of `zone_for_s(s)`; nearest cross-split neighbor in arc length checked directly (closest pair ~6 mm in `s`, but differs by ≥0.1 m in `lateral_error` and/or ≥0.1 rad in `heading_error` — not a near-duplicate).
- Left/right curvature balance checked per split, not just globally (403/403 at κ=±1/3, 599/599 at κ=±1/5, in every one of train/val/test).
