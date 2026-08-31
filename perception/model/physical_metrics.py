"""
Test-set error in physical units, with the two things a bare MAE table
hides: tail behaviour (a controller cares about p95/max, not the mean) and
per-curvature-bin breakdown (an aggregate can average away a specific
geometry the model is bad at).

kappa is deliberately absent from this module's output. Its loss weight is
0 (docs/decisions.md ADR-12) -- the head is untrained, so any statistic
computed on it would describe initialization drift, not model performance,
and would look like a result if printed next to e_y/e_psi in the same
table.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from perception.model.targets import E_Y_SCALE, E_PSI_SCALE, KAPPA_SCALE

CURVATURE_BINS = [("straight", 0.0), ("R=5m arc", 1 / 5), ("R=3m arc", 1 / 3)]
CURVATURE_BIN_TOL = 1e-3


def _bin_label(kappa_true_physical: float):
    mag = abs(kappa_true_physical)
    for label, m in CURVATURE_BINS:
        if abs(mag - m) < CURVATURE_BIN_TOL:
            return label
    return None  # shouldn't happen on v0 -- see docs/decisions.md ADR-11 finding 4


def _stats(errs) -> dict:
    a = np.asarray(errs, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mae": float(a.mean()),
        "rmse": float(np.sqrt((a ** 2).mean())),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


@torch.no_grad()
def physical_metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                      confidence_threshold: float = 0.5) -> dict:
    """Runs inference over `loader` once. Returns:
      e_y, e_psi:        {mae, rmse, p50, p95, max, n} in physical units (m, rad),
                          valid samples only (an invalid sample's lateral_error/
                          heading_error describe a pose never meant to be
                          tracked -- same masking as the training loss).
      e_y_by_bin,
      e_psi_by_bin:       the same stats, split by the TRUE curvature bin
                          (from the label, not the untrained kappa head).
      confidence:         {accuracy, valid_recall, invalid_recall, n_valid,
                          n_invalid} -- accuracy alone hides class imbalance
                          (90/10 valid/invalid on v0), so both per-class
                          recalls are reported alongside it.
    """
    model.eval()
    e_y_errs, e_psi_errs = [], []
    bin_e_y = {label: [] for label, _ in CURVATURE_BINS}
    bin_e_psi = {label: [] for label, _ in CURVATURE_BINS}
    conf_true_all, conf_pred_all = [], []

    for img, target, valid in loader:
        img, target, valid = img.to(device), target.to(device), valid.to(device)
        pred, _ = model(img)

        e_y_pred = (pred[:, 0] * E_Y_SCALE).cpu().numpy()
        e_psi_pred = (pred[:, 1] * E_PSI_SCALE).cpu().numpy()
        e_y_true = (target[:, 0] * E_Y_SCALE).cpu().numpy()
        e_psi_true = (target[:, 1] * E_PSI_SCALE).cpu().numpy()
        kappa_true = (target[:, 2] * KAPPA_SCALE).cpu().numpy()
        v = valid.cpu().numpy().astype(bool)

        conf_pred = (torch.sigmoid(pred[:, 3]) >= confidence_threshold).cpu().numpy()
        conf_true_all.extend(v.tolist())
        conf_pred_all.extend(conf_pred.tolist())

        ey_err = np.abs(e_y_pred - e_y_true)
        epsi_err = np.abs(e_psi_pred - e_psi_true)
        for i in range(len(v)):
            if not v[i]:
                continue
            e_y_errs.append(ey_err[i])
            e_psi_errs.append(epsi_err[i])
            label = _bin_label(kappa_true[i])
            if label is not None:
                bin_e_y[label].append(ey_err[i])
                bin_e_psi[label].append(epsi_err[i])

    conf_true_arr = np.array(conf_true_all, dtype=bool)
    conf_pred_arr = np.array(conf_pred_all, dtype=bool)
    n_valid, n_invalid = int(conf_true_arr.sum()), int((~conf_true_arr).sum())
    accuracy = float((conf_true_arr == conf_pred_arr).mean()) if len(conf_true_arr) else float("nan")
    valid_recall = float(conf_pred_arr[conf_true_arr].mean()) if n_valid else float("nan")
    invalid_recall = float((~conf_pred_arr[~conf_true_arr]).mean()) if n_invalid else float("nan")

    return {
        "e_y": _stats(e_y_errs),
        "e_psi": _stats(e_psi_errs),
        "e_y_by_bin": {k: _stats(v) for k, v in bin_e_y.items()},
        "e_psi_by_bin": {k: _stats(v) for k, v in bin_e_psi.items()},
        "confidence": {
            "accuracy": accuracy, "valid_recall": valid_recall, "invalid_recall": invalid_recall,
            "n_valid": n_valid, "n_invalid": n_invalid,
        },
    }


def format_physical_report(metrics: dict, title: str, in_distribution: bool,
                            lane_half_width: float, heading_envelope: float) -> str:
    """Markdown. `in_distribution` controls the framing sentence -- True for
    the usual test split (per docs/decisions.md ADR-11 finding 5, this is
    interpolation on track geometry the model has effectively memorised,
    not generalisation), False for the mirror-generalisation probe (the
    only v0 evaluation that isn't)."""
    e_y, e_psi, conf = metrics["e_y"], metrics["e_psi"], metrics["confidence"]
    lines = [f"### {title}\n"]

    if in_distribution:
        lines.append(
            "**In-distribution error, not a generalisation measurement.** Per ADR-11 finding 5, "
            "no partition of v0 measures generalisation -- every arc is geometrically identical to "
            "its twins under identical lighting and texture. These are upper bounds on a track the "
            "model has effectively memorised, reported because they're still the honest description "
            "of what was measured, not because they say whether the model can drive on a track it "
            "hasn't seen.\n"
        )
    else:
        lines.append(
            "**The one v0 number that is not pure interpolation.** Trained on source (non-mirrored) "
            "renders only, evaluated on their never-seen mirror twins (right turns from a track that "
            "only physically contains left turns, ADR-10) -- geometrically distinct enough from "
            "training that this measures something closer to generalisation than the in-distribution "
            "table above does.\n"
        )

    lines.append(
        f"| Output | Units | MAE | RMSE | p50 | p95 | max | n |\n"
        f"|---|---|---|---|---|---|---|---|\n"
        f"| e_y | m | {e_y['mae']:.4f} | {e_y['rmse']:.4f} | {e_y['p50']:.4f} | "
        f"{e_y['p95']:.4f} | {e_y['max']:.4f} | {e_y['n']} |\n"
        f"| e_psi | rad | {e_psi['mae']:.4f} | {e_psi['rmse']:.4f} | {e_psi['p50']:.4f} | "
        f"{e_psi['p95']:.4f} | {e_psi['max']:.4f} | {e_psi['n']} |\n"
        f"| e_psi | deg | {np.degrees(e_psi['mae']):.2f} | {np.degrees(e_psi['rmse']):.2f} | "
        f"{np.degrees(e_psi['p50']):.2f} | {np.degrees(e_psi['p95']):.2f} | "
        f"{np.degrees(e_psi['max']):.2f} | {e_psi['n']} |\n"
    )
    lines.append(
        f"\ne_y MAE is {e_y['mae']/lane_half_width*100:.1f}% of the lane half-width "
        f"({lane_half_width} m); p95 is {e_y['p95']/lane_half_width*100:.1f}%. "
        f"e_psi MAE is {np.degrees(e_psi['mae'])/np.degrees(heading_envelope)*100:.1f}% of the "
        f"sampling envelope (+/-{heading_envelope} rad = +/-{np.degrees(heading_envelope):.1f} deg); "
        f"p95 is {np.degrees(e_psi['p95'])/np.degrees(heading_envelope)*100:.1f}%.\n"
    )

    lines.append(
        f"\nConfidence: accuracy {conf['accuracy']:.3f} over n_valid={conf['n_valid']}, "
        f"n_invalid={conf['n_invalid']} (imbalanced ~90/10 -- accuracy alone hides class "
        f"performance). Per-class: valid recall {conf['valid_recall']:.3f}, "
        f"invalid recall {conf['invalid_recall']:.3f}.\n"
    )

    lines.append("\n**kappa: not reported.** Loss weight is 0 (ADR-12) -- the head is untrained, "
                  "so any number here would describe initialization drift, not model performance.\n")

    lines.append("\n| Curvature bin | e_y MAE (m) | e_y p95 (m) | e_psi MAE (deg) | e_psi p95 (deg) | n |\n"
                  "|---|---|---|---|---|---|\n")
    for label, _ in CURVATURE_BINS:
        eyb, epb = metrics["e_y_by_bin"][label], metrics["e_psi_by_bin"][label]
        if eyb["n"] == 0:
            continue
        lines.append(f"| {label} | {eyb['mae']:.4f} | {eyb['p95']:.4f} | "
                      f"{np.degrees(epb['mae']):.2f} | {np.degrees(epb['p95']):.2f} | {eyb['n']} |\n")

    return "".join(lines)
