"""
L = SmoothL1(pred[:3], target[:3]) * valid_mask, mean over valid samples only
  + lambda_conf * BCEWithLogits(pred[3], valid)

The regression term must be exactly zero-weighted on invalid samples, not
merely down-weighted: an invalid sample's lateral_error/heading_error/
curvature describe a pose the vehicle was never meant to track (see
docs/lane-state-contract.md section 4), so there is no sense in which the
network should be pulled toward matching them. Confidence is supervised on
every sample -- that's the only label an invalid sample carries meaning for.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LAMBDA_CONF = 0.1


def lane_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
              lambda_conf: float = LAMBDA_CONF):
    """
    pred:   (B, 4) -- (e_y, e_psi, kappa, confidence_logit), normalized units
    target: (B, 3) -- (e_y, e_psi, kappa), normalized units
    valid:  (B,) float or bool -- 1/True for a valid (in-lane) sample

    Returns (total_loss, regression_loss, confidence_loss) so callers can
    log the components separately.
    """
    valid = valid.to(pred.dtype)
    n_valid = valid.sum()

    per_sample_reg = F.smooth_l1_loss(pred[:, :3], target, reduction="none").mean(dim=1)
    masked_reg = per_sample_reg * valid
    regression_loss = masked_reg.sum() / n_valid.clamp(min=1.0)
    # If a batch happens to contain zero valid samples, there is nothing to
    # regress toward -- contribute exactly zero, not a division artifact.
    regression_loss = torch.where(n_valid > 0, regression_loss, torch.zeros_like(regression_loss))

    confidence_loss = F.binary_cross_entropy_with_logits(pred[:, 3], valid)

    total = regression_loss + lambda_conf * confidence_loss
    return total, regression_loss, confidence_loss
