"""Muon (with AdamW for non-matrix params) and learning-rate schedules.

Muon orthogonalizes the momentum of each 2-D weight matrix with a Newton-Schulz iteration before
applying it. It runs on hidden matrices: attention projections, dense and shared FFNs, and routed
experts (stacked [E, in, out]; each expert's matrix is orthogonalized on its own, batched).
Embeddings, norms, the router gate and anything 1-D stay on AdamW.

Updates are scaled by 0.2 * sqrt(max(rows, cols)) (Moonlight, Liu et al. 2025) so their RMS
matches AdamW's; one learning rate and one weight decay then serve both parameter kinds.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately orthogonalize the last two dims of G (batched over leading dims)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transpose = X.shape[-2] > X.shape[-1]
    if transpose:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if transpose:
        X = X.mT
    return X.to(G.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """One optimizer, two update rules: param groups with ``use_muon=True`` get Muon, the rest AdamW.

    Weight decay is decoupled (p *= 1 - lr * wd) for both.
    """

    def __init__(self, param_groups, lr: float = 2e-3, weight_decay: float = 0.1, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, betas: tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, betas=betas, eps=eps, use_muon=False)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd:
                    p.mul_(1 - lr * wd)
                if group["use_muon"]:
                    self._muon(p, group)
                else:
                    self._adamw(p, group)
        return loss

    def _muon(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
        buf, beta = state["momentum_buffer"], group["momentum"]
        buf.lerp_(p.grad, 1 - beta)
        update = p.grad.lerp(buf, beta) if group["nesterov"] else buf
        update = newton_schulz(update, group["ns_steps"])
        scale = 0.2 * math.sqrt(max(p.shape[-2], p.shape[-1]))
        p.add_(update, alpha=-group["lr"] * scale)

    def _adamw(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        state["step"] += 1
        b1, b2 = group["betas"]
        m, v = state["exp_avg"], state["exp_avg_sq"]
        m.lerp_(p.grad, 1 - b1)
        v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
        bc1, bc2 = 1 - b1 ** state["step"], 1 - b2 ** state["step"]
        denom = (v / bc2).sqrt_().add_(group["eps"])
        p.addcdiv_(m, denom, value=-group["lr"] / bc1)


def is_muon_param(name: str, p: nn.Parameter) -> bool:
    """Hidden weight matrices: 2-D+ and not the embedding/head or the router."""
    return p.ndim >= 2 and not name.startswith(("embed.", "head.")) and not name.endswith("gate.weight")


def build_optimizer(model: nn.Module, kind: str, lr: float, weight_decay: float,
                    betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.Optimizer:
    """kind: 'muon' (Muon for hidden matrices + AdamW for the rest) or 'adamw' (AdamW everywhere).

    Weight decay applies to every 2-D+ parameter (not to norms and other 1-D params).
    """
    named = list(model.named_parameters())
    no_decay = [p for _, p in named if p.ndim < 2]
    if kind == "adamw":
        decay = [p for _, p in named if p.ndim >= 2]
        return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=betas)
    if kind != "muon":
        raise ValueError(f"unknown optimizer {kind!r}")
    muon = [p for n, p in named if is_muon_param(n, p)]
    adam_decay = [p for n, p in named if p.ndim >= 2 and not is_muon_param(n, p)]
    return MuonAdamW([
        {"params": muon, "use_muon": True, "weight_decay": weight_decay},
        {"params": adam_decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, weight_decay=weight_decay, betas=betas)


def lr_at(step: int, total_steps: int, peak: float, warmup: int, schedule: str = "wsd",
          decay_frac: float = 0.2, min_lr: float = 0.0) -> float:
    """Learning rate for a step.

    wsd:    linear warmup, constant at peak, then linear decay to min_lr over the last decay_frac of
            training. Until the decay starts, the LR doesn't depend on total_steps, so a run can be
            resumed with a larger --steps to train longer without disturbing the schedule.
    cosine: linear warmup, then cosine from peak to min_lr.
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    if schedule == "wsd":
        decay_start = max(warmup, int(total_steps * (1 - decay_frac)))
        if step < decay_start:
            return peak
        progress = (step - decay_start + 1) / max(1, total_steps - decay_start)
        return peak + (min_lr - peak) * min(progress, 1.0)
    if schedule == "cosine":
        progress = (step - warmup) / max(1, total_steps - warmup)
        return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))
    raise ValueError(f"unknown schedule {schedule!r}")
