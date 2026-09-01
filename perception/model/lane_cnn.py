"""
LaneCNN: the M2 perception network (FR-8, FR-9).

Custom, compact architecture dimensioned for embedded deployment (FR-9) --
not a fine-tuned generic detector. Input 3x80x160 (see preprocess.py for how
a 320x240 render becomes that). Output is the four raw /lane_state scalars
in NORMALIZED units (see targets.py): (e_y, e_psi, kappa, confidence_logit).
The caller denormalizes and applies sigmoid to the confidence logit -- this
module has no opinion on physical units or the valid/invalid threshold.

Every conv is followed by BatchNorm2d + ReLU and carries no bias (the BN
shift makes a conv bias redundant). `width_mult` scales every conv channel
count (not the FC layers) so capacity can be swept -- see the M2 width
ablation in train.py, which checks whether v0's 3x80x160/five-curvature-value
task (docs/decisions.md ADR-11, findings 4-5) actually needs full capacity.

`forward` returns (output, b3_features): the raw b3 feature map (before the
1x1 channel reduction) is exposed so a different head -- e.g. a per-column
head -- can be attached later without retraining the encoder. ADR-11 finding
3 is why v0 doesn't use a per-column head itself: the labels are a single
Frenet-frame evaluation at the projection point, not per-column ground
truth, so there is nothing for a per-column head to be supervised against
on this dataset.
"""

from typing import Tuple

import torch
import torch.nn as nn

IN_CHANNELS, IN_HEIGHT, IN_WIDTH = 3, 80, 160
N_OUTPUTS = 4  # e_y, e_psi, kappa, confidence_logit


def _round_channels(base: int, width_mult: float) -> int:
    return max(1, int(round(base * width_mult)))


def _conv_bn_relu(in_ch: int, out_ch: int, k: int, s: int, p: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class LaneCNN(nn.Module):
    def __init__(self, width_mult: float = 1.0):
        super().__init__()
        self.width_mult = width_mult
        c16 = _round_channels(16, width_mult)
        c32 = _round_channels(32, width_mult)
        c64 = _round_channels(64, width_mult)
        c96 = _round_channels(96, width_mult)
        cred = _round_channels(32, width_mult)

        # 3x80x160 -> c16 x 40x80
        self.stem = _conv_bn_relu(IN_CHANNELS, c16, k=5, s=2, p=2)

        # c16x40x80 -> c32x20x40
        self.b1a = _conv_bn_relu(c16, c32, k=3, s=2, p=1)
        self.b1b = _conv_bn_relu(c32, c32, k=3, s=1, p=1)

        # c32x20x40 -> c64x10x20
        self.b2a = _conv_bn_relu(c32, c64, k=3, s=2, p=1)
        self.b2b = _conv_bn_relu(c64, c64, k=3, s=1, p=1)

        # c64x10x20 -> c96x5x10
        self.b3a = _conv_bn_relu(c64, c96, k=3, s=2, p=1)
        self.b3b = _conv_bn_relu(c96, c96, k=3, s=1, p=1)

        # c96x5x10 -> credx5x10
        self.red = _conv_bn_relu(c96, cred, k=1, s=1, p=0)

        flat_dim = cred * 5 * 10
        self.fc1 = nn.Linear(flat_dim, 128)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, N_OUTPUTS)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.b1a(x)
        x = self.b1b(x)
        x = self.b2a(x)
        x = self.b2b(x)
        x = self.b3a(x)
        b3_features = self.b3b(x)
        x = self.red(b3_features)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out, b3_features


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, input_shape: Tuple[int, int, int] = (IN_CHANNELS, IN_HEIGHT, IN_WIDTH)) -> int:
    """Multiply-accumulate count, computed from the actual graph via forward
    hooks on Conv2d/Linear -- not a hand-derived constant, so it tracks the
    architecture (and width_mult) automatically if either changes.

    Conv2d: MACs = out_H * out_W * out_ch * (in_ch / groups) * kH * kW.
    Linear: MACs = in_features * out_features (batched over any leading dims).
    BatchNorm/ReLU/Dropout are elementwise -- O(1) MACs each, omitted, as is
    conventional for this kind of count.
    """
    macs = 0

    def conv_hook(module, inputs, output):
        nonlocal macs
        out_h, out_w = output.shape[-2], output.shape[-1]
        in_ch_per_group = module.in_channels // module.groups
        kh, kw = module.kernel_size
        macs += out_h * out_w * module.out_channels * in_ch_per_group * kh * kw

    def linear_hook(module, inputs, output):
        nonlocal macs
        batch = 1
        for d in output.shape[:-1]:
            batch *= d
        macs += batch * module.in_features * module.out_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape, device=device)
            model(dummy)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    return macs
