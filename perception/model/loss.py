"""
L = weighted-mean_i( w_i * SmoothL1(pred[:,i], target[:,i]) ) * valid_mask,
    mean over valid samples only, i in {e_y, e_psi, kappa}
  + lambda_conf * BCEWithLogits(pred[3], valid)

The regression term must be exactly zero-weighted on invalid samples, not
merely down-weighted: an invalid sample's lateral_error/heading_error/
curvature describe a pose the vehicle was never meant to track (see
docs/lane-state-contract.md section 4), so there is no sense in which the
network should be pulled toward matching them. Confidence is supervised on
every sample -- that's the only label an invalid sample carries meaning for.

kappa's default weight is 0 (COMPONENT_WEIGHTS below, overridable via
training_config.yaml's `component_weights`), not 1: the point-wise Frenet
label is measurably wrong within L_usable (2.36 m) of a curvature transition
(docs/decisions.md ADR-11 finding 3, confirmed by the M2 diagnostics --
mean|kappa_pred| for straight samples correlates at r=-0.67 with distance to
the next transition, plateauing exactly at L_usable, not noise). Feeding
that gradient into the network taught it to predict curvature that
contradicts the label near every one of the track's 8 transitions --
covering ~42% of the loop -- which is actively harmful, not merely
unhelpful: an MPC feedforward term built on it would steer off a straight
line approaching a curve. The output head is kept in the architecture
(unweighted diagnostics still computed by component_losses below) so a
future windowed or continuous-curvature label can reuse it without an
architecture change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LAMBDA_CONF = 0.1
COMPONENT_WEIGHTS = {"e_y": 1.0, "e_psi": 1.0, "kappa": 0.0}


def lane_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
              lambda_conf: float = LAMBDA_CONF, component_weights: dict = None):
    """
    pred:   (B, 4) -- (e_y, e_psi, kappa, confidence_logit), normalized units
    target: (B, 3) -- (e_y, e_psi, kappa), normalized units
    valid:  (B,) float or bool -- 1/True for a valid (in-lane) sample
    component_weights: {"e_y":.., "e_psi":.., "kappa":..} -- defaults to
        COMPONENT_WEIGHTS (kappa=0). Weighted MEAN, not weighted sum: with
        kappa's weight at 0, this is exactly the mean of e_y's and e_psi's
        loss, not that mean diluted by an always-zero third term.

    Returns (total_loss, regression_loss, confidence_loss) so callers can
    log the components separately.
    """
    if component_weights is None:
        component_weights = COMPONENT_WEIGHTS
    weights = torch.tensor(
        [component_weights["e_y"], component_weights["e_psi"], component_weights["kappa"]],
        dtype=pred.dtype, device=pred.device,
    )
    weight_sum = weights.sum().clamp(min=1e-8)

    valid = valid.to(pred.dtype)
    n_valid = valid.sum()

    per_sample_per_target = F.smooth_l1_loss(pred[:, :3], target, reduction="none")  # (B, 3)
    per_sample_reg = (per_sample_per_target * weights).sum(dim=1) / weight_sum
    masked_reg = per_sample_reg * valid
    regression_loss = masked_reg.sum() / n_valid.clamp(min=1.0)
    # If a batch happens to contain zero valid samples, there is nothing to
    # regress toward -- contribute exactly zero, not a division artifact.
    regression_loss = torch.where(n_valid > 0, regression_loss, torch.zeros_like(regression_loss))

    confidence_loss = F.binary_cross_entropy_with_logits(pred[:, 3], valid)

    total = regression_loss + lambda_conf * confidence_loss
    return total, regression_loss, confidence_loss


def component_losses(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    """Per-target breakdown of the regression term (e_y, e_psi, kappa each
    scored separately, not averaged together) plus confidence -- diagnostic
    only, not used to weight training. The combined `lane_loss` above can
    fall while one component stays flat; this is what makes that visible
    (docs/decisions.md ADR-11-adjacent M2 diagnostics)."""
    valid_f = valid.to(pred.dtype)
    n_valid = valid_f.sum().clamp(min=1.0)

    per_sample_per_target = F.smooth_l1_loss(pred[:, :3], target, reduction="none")  # (B, 3)
    masked = per_sample_per_target * valid_f.unsqueeze(1)
    per_target = masked.sum(dim=0) / n_valid  # (3,)

    confidence_loss = F.binary_cross_entropy_with_logits(pred[:, 3], valid_f)

    return {
        "e_y": per_target[0].item(),
        "e_psi": per_target[1].item(),
        "kappa": per_target[2].item(),
        "confidence": confidence_loss.item(),
    }
