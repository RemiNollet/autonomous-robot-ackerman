"""
Training entry point for LaneCNN (M2). Plain PyTorch: Adam, cosine schedule,
argparse. No Lightning, no Hydra, no config-class hierarchy.

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
    python3 perception/model/train.py --epochs 2        # quick smoke run
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from perception.model.dataset import LaneDataset
from perception.model.lane_cnn import LaneCNN, count_params, count_macs
from perception.model.loss import lane_loss, LAMBDA_CONF
from perception.model.targets import E_Y_SCALE, E_PSI_SCALE, KAPPA_SCALE

CHECKPOINT_DIR = "perception/model/checkpoints"
RESULTS_PATH = "perception/model/results.md"


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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EpochStats:
    model.eval()
    total_loss = total_reg = total_conf = 0.0
    n_batches = 0
    sum_mae = np.zeros(3)
    n_valid_total = 0.0
    correct_conf = 0.0
    n_total = 0

    for img, target, valid in loader:
        img, target, valid = img.to(device), target.to(device), valid.to(device)
        pred, _ = model(img)
        loss, reg, conf = lane_loss(pred, target, valid)
        total_loss += loss.item()
        total_reg += reg.item()
        total_conf += conf.item()
        n_batches += 1

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
    )


def train_model(width_mult: float, train_ds, val_ds, epochs: int, lr: float, batch_size: int,
                 device: torch.device, seed: int,
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
        for img, target, valid in train_loader:
            img, target, valid = img.to(device), target.to(device), valid.to(device)
            pred, _ = model(img)
            loss, _, _ = lane_loss(pred, target, valid)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_steps += 1
        scheduler.step()
        train_loss = running_loss / max(n_steps, 1)

        # Full eval pass only over val (not train, which was already scored
        # live above) -- halves epoch cost, and reports genuine train-mode
        # (not eval-mode BatchNorm) loss, which is the more honest number.
        val_stats = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_stats.loss,
                         "val_mae_e_y": val_stats.mae_e_y, "val_mae_e_psi": val_stats.mae_e_psi,
                         "val_mae_kappa": val_stats.mae_kappa})
        print(f"{log_prefix}epoch {epoch+1:2d}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_stats.loss:.4f}  val_MAE e_y={val_stats.mae_e_y:.4f}m "
              f"e_psi={val_stats.mae_e_psi:.4f}rad kappa={val_stats.mae_kappa:.4f}/m "
              f"conf_acc={val_stats.conf_accuracy:.3f}")

        if val_stats.loss < best_val_loss:
            best_val_loss = val_stats.loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_stats = evaluate(model, val_loader, device)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-ablation", action="store_true",
                     help="train only the main width=1.0 model, skip baselines")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")

    train_ds = LaneDataset(split="train", augment=True, seed=args.seed)
    val_ds = LaneDataset(split="val", augment=False)
    test_ds = LaneDataset(split="test", augment=False)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    results = {}

    # --- main model, width_mult=1.0 ---
    t0 = time.time()
    model, val_stats, history = train_model(
        width_mult=1.0, train_ds=train_ds, val_ds=val_ds, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=device, seed=args.seed, log_prefix="[main] ",
    )
    test_stats = evaluate(model, DataLoader(test_ds, batch_size=args.batch_size), device)
    save_checkpoint(model, 1.0, f"{CHECKPOINT_DIR}/lane_cnn_width1.0_best.pt")
    results["main (width=1.0)"] = {
        "params": count_params(model), "macs": count_macs(model),
        "val": val_stats, "test": test_stats, "train_time_s": time.time() - t0,
    }

    if args.skip_ablation:
        _write_results(results, args)
        return

    # --- baseline 1: constant predictor ---
    const_stats = constant_predictor_baseline(train_ds, test_ds)
    results["constant predictor"] = {"params": 3, "macs": 0, "val": None, "test": const_stats,
                                      "train_time_s": 0.0}

    # --- baseline 2: width ablation, width_mult=0.25 ---
    t0 = time.time()
    model025, val_stats025, _ = train_model(
        width_mult=0.25, train_ds=train_ds, val_ds=val_ds, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=device, seed=args.seed, log_prefix="[width=0.25] ",
    )
    test_stats025 = evaluate(model025, DataLoader(test_ds, batch_size=args.batch_size), device)
    save_checkpoint(model025, 0.25, f"{CHECKPOINT_DIR}/lane_cnn_width0.25_best.pt")
    results["width ablation (width=0.25)"] = {
        "params": count_params(model025), "macs": count_macs(model025),
        "val": val_stats025, "test": test_stats025, "train_time_s": time.time() - t0,
    }

    # --- baseline 3: mirror-generalization probe ---
    train_src_ds = LaneDataset(split="train", mirrored=False, augment=True, seed=args.seed)
    train_mirror_eval_ds = LaneDataset(split="train", mirrored=True, augment=False)
    t0 = time.time()
    probe_model, probe_val_stats, _ = train_model(
        width_mult=1.0, train_ds=train_src_ds, val_ds=val_ds, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=device, seed=args.seed, log_prefix="[mirror-probe] ",
    )
    mirror_eval_stats = evaluate(probe_model, DataLoader(train_mirror_eval_ds, batch_size=args.batch_size), device)
    results["mirror-generalization probe"] = {
        "params": count_params(probe_model), "macs": count_macs(probe_model),
        "val": probe_val_stats, "test": mirror_eval_stats, "train_time_s": time.time() - t0,
        "note": "trained on train-split SOURCE renders only; 'test' column here is "
                "evaluation on those same samples' MIRROR twins (unseen), not the usual test split",
    }

    _write_results(results, args)


def _write_results(results: dict, args):
    lines = ["# M2 results\n", f"\nseed={args.seed} epochs={args.epochs} lr={args.lr} batch_size={args.batch_size}\n",
              "\n| Run | Params | MACs | val loss | test/eval MAE e_y (m) | MAE e_psi (rad) | MAE kappa (1/m) | conf acc | train time |\n",
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
