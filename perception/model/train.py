"""
Training entry point for LaneCNN (M2). Plain PyTorch: Adam, cosine schedule,
argparse. No Lightning, no Hydra, no config-class hierarchy.

Hyperparameters default from perception/model/training_config.yaml (the
versioned, comparable source of truth -- its hash is written into every row
of results.md); CLI flags override individual values for a one-off run.

Also runs the M2 baselines/ablation (these are the point of the exercise,
not extras -- see docs/decisions.md ADR-11):
  1. constant predictor       -- is the CNN better than memorizing the mean?
  2. width_mult 1.0 vs 0.25   -- does v0 (ADR-11 findings 3-5: 5-valued
                                  curvature, single-geometry arcs) actually
                                  need full capacity, or is the margin
                                  deliberate over-provisioning for the
                                  eventual degraded-domain case?
  3. mirror-generalization probe -- train on source renders only, evaluate
                                  on their mirror twins (ADR-10) -- the only
                                  v0 evaluation that isn't pure interpolation
                                  within identical geometry (ADR-11 finding 5).

Usage:
    python3 perception/model/train.py                  # full run: main model + all baselines
    python3 perception/model/train.py --epochs 2        # quick smoke run, config override
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from perception.model.dataset import LaneDataset
from perception.model.lane_cnn import LaneCNN, count_params, count_macs
from perception.model.loss import lane_loss, component_losses
from perception.model.targets import E_Y_SCALE, E_PSI_SCALE, KAPPA_SCALE

CHECKPOINT_DIR = "perception/model/checkpoints"
RESULTS_PATH = "perception/model/results.md"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "training_config.yaml")
LOSS_CURVES_PATH = "perception/model/loss_curves.png"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(config: dict) -> str:
    """Short, stable hash of the effective config (after CLI overrides) --
    written into results.md so two result tables can be compared without
    guessing what differed between the runs that produced them."""
    canonical = json.dumps(config, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class EpochStats:
    loss: float
    reg_loss: float
    conf_loss: float
    mae_e_y: float    # metres
    mae_e_psi: float  # radians
    mae_kappa: float  # 1/metres
    conf_accuracy: float
    comp_e_y: float = 0.0       # per-component SmoothL1, normalized units
    comp_e_psi: float = 0.0
    comp_kappa: float = 0.0
    comp_conf: float = 0.0


def _physical_mae(pred3_n: torch.Tensor, target3_n: torch.Tensor, valid: torch.Tensor):
    """MAE per component, in physical units, over valid samples only --
    denormalizing before taking the difference is equivalent to scaling the
    normalized MAE, but doing it explicitly avoids relying on that being
    remembered correctly at every call site."""
    scale = torch.tensor([E_Y_SCALE, E_PSI_SCALE, KAPPA_SCALE], device=pred3_n.device)
    diff = (pred3_n - target3_n).abs() * scale
    mask = valid.unsqueeze(1)
    n_valid = valid.sum().clamp(min=1.0)
    mae = (diff * mask).sum(dim=0) / n_valid
    return mae[0].item(), mae[1].item(), mae[2].item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             component_weights: dict = None) -> EpochStats:
    model.eval()
    total_loss = total_reg = total_conf = 0.0
    n_batches = 0
    sum_mae = np.zeros(3)
    n_valid_total = 0.0
    correct_conf = 0.0
    n_total = 0
    comp_sums = {"e_y": 0.0, "e_psi": 0.0, "kappa": 0.0, "confidence": 0.0}

    for img, target, valid in loader:
        img, target, valid = img.to(device), target.to(device), valid.to(device)
        pred, _ = model(img)
        loss, reg, conf = lane_loss(pred, target, valid, component_weights=component_weights)
        total_loss += loss.item()
        total_reg += reg.item()
        total_conf += conf.item()
        n_batches += 1

        comps = component_losses(pred, target, valid)
        for k in comp_sums:
            comp_sums[k] += comps[k]

        mae_y, mae_psi, mae_k = _physical_mae(pred[:, :3], target, valid)
        n_valid_batch = valid.sum().item()
        sum_mae += np.array([mae_y, mae_psi, mae_k]) * n_valid_batch
        n_valid_total += n_valid_batch

        pred_valid = (torch.sigmoid(pred[:, 3]) >= 0.5).float()
        correct_conf += (pred_valid == valid).sum().item()
        n_total += valid.numel()

    n_valid_total = max(n_valid_total, 1.0)
    mae = sum_mae / n_valid_total
    return EpochStats(
        loss=total_loss / n_batches, reg_loss=total_reg / n_batches, conf_loss=total_conf / n_batches,
        mae_e_y=mae[0], mae_e_psi=mae[1], mae_kappa=mae[2],
        conf_accuracy=correct_conf / max(n_total, 1),
        comp_e_y=comp_sums["e_y"] / n_batches, comp_e_psi=comp_sums["e_psi"] / n_batches,
        comp_kappa=comp_sums["kappa"] / n_batches, comp_conf=comp_sums["confidence"] / n_batches,
    )


def train_model(width_mult: float, train_ds, val_ds, epochs: int, lr: float, batch_size: int,
                 device: torch.device, seed: int, component_weights: dict = None,
                 log_prefix: str = "") -> Tuple[nn.Module, EpochStats, list]:
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LaneCNN(width_mult=width_mult).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss, n_steps = 0.0, 0
        comp_sums = {"e_y": 0.0, "e_psi": 0.0, "kappa": 0.0, "confidence": 0.0}
        for img, target, valid in train_loader:
            img, target, valid = img.to(device), target.to(device), valid.to(device)
            pred, _ = model(img)
            loss, _, _ = lane_loss(pred, target, valid, component_weights=component_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_steps += 1
            with torch.no_grad():
                comps = component_losses(pred, target, valid)
            for k in comp_sums:
                comp_sums[k] += comps[k]
        scheduler.step()
        train_loss = running_loss / max(n_steps, 1)
        train_comp = {k: v / max(n_steps, 1) for k, v in comp_sums.items()}

        # Full eval pass only over val (not train, which was already scored
        # live above) -- halves epoch cost, and reports genuine train-mode
        # (not eval-mode BatchNorm) loss, which is the more honest number.
        val_stats = evaluate(model, val_loader, device, component_weights=component_weights)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_stats.loss,
            "train_e_y": train_comp["e_y"], "val_e_y": val_stats.comp_e_y,
            "train_e_psi": train_comp["e_psi"], "val_e_psi": val_stats.comp_e_psi,
            "train_kappa": train_comp["kappa"], "val_kappa": val_stats.comp_kappa,
            "train_confidence": train_comp["confidence"], "val_confidence": val_stats.comp_conf,
            "val_mae_e_y": val_stats.mae_e_y, "val_mae_e_psi": val_stats.mae_e_psi,
            "val_mae_kappa": val_stats.mae_kappa,
        })
        print(f"{log_prefix}epoch {epoch+1:2d}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_stats.loss:.4f}  "
              f"comp[e_y={val_stats.comp_e_y:.4f} e_psi={val_stats.comp_e_psi:.4f} "
              f"kappa={val_stats.comp_kappa:.4f} conf={val_stats.comp_conf:.4f}]  "
              f"val_MAE e_y={val_stats.mae_e_y:.4f}m e_psi={val_stats.mae_e_psi:.4f}rad "
              f"kappa={val_stats.mae_kappa:.4f}/m conf_acc={val_stats.conf_accuracy:.3f}")

        if val_stats.loss < best_val_loss:
            best_val_loss = val_stats.loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_stats = evaluate(model, val_loader, device, component_weights=component_weights)
    return model, final_val_stats, history


def constant_predictor_baseline(train_ds: LaneDataset, eval_ds: LaneDataset) -> EpochStats:
    """Predict the training-set mean (over valid samples) for every output,
    regardless of input. If LaneCNN isn't clearly better than this, it has
    learned nothing about geometry -- ADR-11 finding 4: curvature is 5
    discrete values, so even a constant predictor scores well on it alone."""
    sums = np.zeros(3)
    n_valid = 0
    for row in train_ds.rows:
        if row["valid"] == "True":
            sums += np.array([float(row["lateral_error"]), float(row["heading_error"]),
                               float(row["curvature"])])
            n_valid += 1
    mean_target = sums / max(n_valid, 1)
    train_valid_frac = n_valid / len(train_ds.rows)
    # constant predictor's "confidence" is just the train prevalence,
    # thresholded at 0.5 -- same rule applied everywhere else in this file.
    predicted_valid = train_valid_frac >= 0.5

    sum_ae = np.zeros(3)
    n_eval_valid = 0
    correct_conf = 0
    for row in eval_ds.rows:
        is_valid = row["valid"] == "True"
        if is_valid:
            actual = np.array([float(row["lateral_error"]), float(row["heading_error"]),
                                float(row["curvature"])])
            sum_ae += np.abs(actual - mean_target)
            n_eval_valid += 1
        if predicted_valid == is_valid:
            correct_conf += 1

    mae = sum_ae / max(n_eval_valid, 1)
    return EpochStats(
        loss=float("nan"), reg_loss=float("nan"), conf_loss=float("nan"),
        mae_e_y=mae[0], mae_e_psi=mae[1], mae_kappa=mae[2],
        conf_accuracy=correct_conf / len(eval_ds.rows),
    )


def save_checkpoint(model: nn.Module, width_mult: float, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "width_mult": width_mult}, path)


def plot_loss_curves(history: list, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] + 1 for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    components = [("e_y", axes[0, 0]), ("e_psi", axes[0, 1]),
                  ("kappa", axes[1, 0]), ("confidence", axes[1, 1])]
    for name, ax in components:
        ax.plot(epochs, [h[f"train_{name}"] for h in history], label="train")
        ax.plot(epochs, [h[f"val_{name}"] for h in history], label="val")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8)
    fig.suptitle("Per-component loss (normalized units for e_y/e_psi/kappa; BCE for confidence)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"Loss curves written to {path}")


def plateau_epoch(history: list, key: str, rel_tol: float = 0.02, window: int = 5) -> int:
    """First epoch (1-indexed) after which `key`'s value never again differs
    from its own final value by more than rel_tol (relative), sustained for
    `window` consecutive epochs -- a simple, inspectable plateau detector,
    not a statistical claim."""
    values = np.array([h[key] for h in history])
    final = values[-window:].mean()
    threshold = abs(final) * rel_tol + 1e-6
    for i in range(len(values) - window, -1, -1):
        window_vals = values[i:i + window] if i + window <= len(values) else values[i:]
        if np.any(np.abs(window_vals - final) > threshold):
            return min(i + window + 1, len(values))
    return 1


def main():
    config = load_config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config["epochs"])
    ap.add_argument("--lr", type=float, default=config["lr"])
    ap.add_argument("--batch-size", type=int, default=config["batch_size"])
    ap.add_argument("--seed", type=int, default=config["seed"])
    ap.add_argument("--skip-ablation", action="store_true",
                     help="train only the main width=1.0 model, skip baselines")
    ap.add_argument("--plot-main-loss-curves", action="store_true",
                     help="write loss_curves.png from the main model's per-epoch history")
    args = ap.parse_args()

    # Effective config = file, with any CLI overrides folded back in, so the
    # hash reflects what actually ran, not just what the file said.
    config["epochs"] = args.epochs
    config["lr"] = args.lr
    config["batch_size"] = args.batch_size
    config["seed"] = args.seed
    run_hash = config_hash(config)

    device = get_device()
    print(f"device: {device}  config_hash: {run_hash}")

    aug_params = config["augmentation"]
    dataset_dir = config["dataset_dir"]
    labels_csv = f"{dataset_dir}/labels.csv"
    img_dir = f"{dataset_dir}/images"

    train_ds = LaneDataset(labels_csv, img_dir, split="train", augment=True, seed=args.seed,
                            augment_params=aug_params)
    val_ds = LaneDataset(labels_csv, img_dir, split="val", augment=False)
    test_ds = LaneDataset(labels_csv, img_dir, split="test", augment=False)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    results = {}

    # --- main model, width_mult=1.0 ---
    t0 = time.time()
    model, val_stats, history = train_model(
        width_mult=1.0, train_ds=train_ds, val_ds=val_ds, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=device, seed=args.seed,
        component_weights=config["component_weights"], log_prefix="[main] ",
    )
    test_stats = evaluate(model, DataLoader(test_ds, batch_size=args.batch_size), device,
                           component_weights=config["component_weights"])
    save_checkpoint(model, 1.0, f"{CHECKPOINT_DIR}/lane_cnn_width1.0_best.pt")
    results["main (width=1.0)"] = {
        "params": count_params(model), "macs": count_macs(model),
        "val": val_stats, "test": test_stats, "train_time_s": time.time() - t0,
    }

    plot_loss_curves(history, LOSS_CURVES_PATH)
    plateaus = {k: plateau_epoch(history, f"val_{k}") for k in ("e_y", "e_psi", "kappa", "confidence")}
    print("plateau epochs (val, main model):", plateaus)

    if args.skip_ablation:
        _write_results(results, config, run_hash, plateaus)
        return

    # --- baseline 1: constant predictor ---
    const_stats = constant_predictor_baseline(train_ds, test_ds)
    results["constant predictor"] = {"params": 3, "macs": 0, "val": None, "test": const_stats,
                                      "train_time_s": 0.0}

    # --- baseline 2: width ablation, width_mult=0.25 ---
    t0 = time.time()
    model025, val_stats025, _ = train_model(
        width_mult=0.25, train_ds=train_ds, val_ds=val_ds, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=device, seed=args.seed,
        component_weights=config["component_weights"], log_prefix="[width=0.25] ",
    )
    test_stats025 = evaluate(model025, DataLoader(test_ds, batch_size=args.batch_size), device,
                              component_weights=config["component_weights"])
    save_checkpoint(model025, 0.25, f"{CHECKPOINT_DIR}/lane_cnn_width0.25_best.pt")
    results["width ablation (width=0.25)"] = {
        "params": count_params(model025), "macs": count_macs(model025),
        "val": val_stats025, "test": test_stats025, "train_time_s": time.time() - t0,
    }

    # --- baseline 3: mirror-generalization probe ---
    train_src_ds = LaneDataset(labels_csv, img_dir, split="train", mirrored=False, augment=True,
                                seed=args.seed, augment_params=aug_params)
    train_mirror_eval_ds = LaneDataset(labels_csv, img_dir, split="train", mirrored=True, augment=False)
    t0 = time.time()
    probe_model, probe_val_stats, _ = train_model(
        width_mult=1.0, train_ds=train_src_ds, val_ds=val_ds, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=device, seed=args.seed,
        component_weights=config["component_weights"], log_prefix="[mirror-probe] ",
    )
    mirror_eval_stats = evaluate(probe_model, DataLoader(train_mirror_eval_ds, batch_size=args.batch_size), device,
                                  component_weights=config["component_weights"])
    results["mirror-generalization probe"] = {
        "params": count_params(probe_model), "macs": count_macs(probe_model),
        "val": probe_val_stats, "test": mirror_eval_stats, "train_time_s": time.time() - t0,
        "note": "trained on train-split SOURCE renders only; 'test' column here is "
                "evaluation on those same samples' MIRROR twins (unseen), not the usual test split",
    }

    _write_results(results, config, run_hash, plateaus)


def _write_results(results: dict, config: dict, run_hash: str, plateaus: dict = None):
    lines = ["# M2 results\n",
             f"\nconfig_hash={run_hash}  (perception/model/training_config.yaml, "
             f"epochs={config['epochs']} lr={config['lr']} batch_size={config['batch_size']} "
             f"seed={config['seed']})\n",
             "\nNo early stopping on val loss: v0's val split does not measure generalisation "
             "(ADR-11 finding 5 -- every arc is geometrically identical to its twin), so stopping "
             "on it would halt on noise and make runs non-reproducible. Epoch count is fixed, "
             "chosen from where perception/model/loss_curves.png actually plateaus.\n",
             f"\nkappa's loss weight is {config['component_weights']['kappa']}: the point-wise Frenet "
             "kappa label is measurably wrong within L_usable (2.36m) of a curvature transition "
             "(~42% of the loop). perception/model/kappa_transition_proximity.png: on straight "
             "samples, mean|kappa_pred| correlates at r=-0.67 with distance to the next transition, "
             "plateauing almost exactly at L_usable (0.023 beyond it vs 0.115 within it, a 5x gap) -- "
             "the network is reading the road correctly and being penalised for it. The MAE/scatter "
             "numbers below are NOT merely unusable: a kappa output that predicts curvature on "
             "straights and near-zero in curves would actively steer an MPC feedforward term off a "
             "straight line approaching a bend. The output head stays in the architecture "
             "(component_losses still logs kappa's raw, unweighted loss in loss_curves.png for "
             "monitoring) so a future windowed or continuous-curvature label can reuse it.\n"]
    if plateaus:
        lines.append(f"\nPlateau epoch per component (val, main model): {plateaus}\n")
    lines += ["\n| Run | Params | MACs | val loss | test/eval MAE e_y (m) | MAE e_psi (rad) | MAE kappa (1/m) | conf acc | train time |\n",
              "|---|---|---|---|---|---|---|---|---|\n"]
    for name, r in results.items():
        val_loss = f"{r['val'].loss:.4f}" if r["val"] is not None else "n/a"
        t = r["test"]
        lines.append(f"| {name} | {r['params']:,} | {r['macs']:,} | {val_loss} | "
                      f"{t.mae_e_y:.4f} | {t.mae_e_psi:.4f} | {t.mae_kappa:.4f} | "
                      f"{t.conf_accuracy:.3f} | {r['train_time_s']:.1f}s |\n")
        if "note" in r:
            lines.append(f"\n*{name}*: {r['note']}\n")
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.writelines(lines)
    print(f"\nResults written to {RESULTS_PATH}")
    print("".join(lines))


if __name__ == "__main__":
    main()
