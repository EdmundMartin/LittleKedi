"""DeepSeekMoE with auxiliary-loss-free load balancing, built for batched matmuls and torch.compile.

Routing (unchanged from v1):
  * Fine-grained experts: many narrow experts, several activated per token.
  * Shared experts: always-on experts that absorb common knowledge.
  * Sigmoid affinities, normalized over the selected top-k.
  * Group-limited routing: each token may only pick from its best ``n_limited_groups`` groups.
  * Aux-loss-free balancing: a per-expert bias is added to affinities *only for top-k selection*
    and nudged after each optimizer step from the observed load.
  * A tiny complementary sequence-wise balance loss.

Execution (new in v2): routed expert weights are stacked into [E, D, H] / [E, H, D] tensors, so all
experts run as a few large kernels instead of ~3 small ones per expert:

  * ``capacity``: GShard/Switch-style fixed-capacity dispatch. Every shape depends only on
    (tokens, E, top-k, capacity factor), there are no host syncs, and the layer compiles to one
    static graph. Assignments past an expert's capacity are dropped (top-1 choices are placed first).
  * ``grouped_mm``: ``torch._grouped_mm`` over expert-sorted tokens. Dropless and sync-free.
  * ``sorted``: per-expert loop over a sorted batch. Exact reference; one host sync per layer.

The forward is pure: the load statistics and aux loss are returned, not stored on the module, so a
compiled graph has no side effects. ``update_bias`` runs outside any compiled region.
"""

from __future__ import annotations



import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, capacity
from .runtime import compute_dtype, full_precision, grouped_mm_available


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, dim, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _GroupedMM(torch.autograd.Function):
    """torch._grouped_mm with an explicit backward.

    x: [M, K] rows sorted by group, w: [G, K, N], offs: [G] int32 cumulative row ends.
    The built-in backward fails on the ROCm 10 / gfx1150 build; its two halves work when called directly.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, w, offs)
        return torch._grouped_mm(x, w, offs=offs)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w, offs = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch._grouped_mm(dy, w.transpose(-2, -1).contiguous(), offs=offs)
        dw = torch._grouped_mm(x.t(), dy, offs=offs)
        return dx, dw, None


class Experts(nn.Module):
    """E routed SwiGLU experts with stacked weights.

    w_gate, w_up: [E, D, H]    w_down: [E, H, D]    (y = (silu(x @ w_gate) * (x @ w_up)) @ w_down)
    """

    NAMES = ("w_gate", "w_up", "w_down")

    def __init__(self, n_experts: int, dim: int, hidden: int):
        super().__init__()
        self.n_experts, self.dim, self.hidden = n_experts, dim, hidden
        self.w_gate = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.w_up = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.w_down = nn.Parameter(torch.empty(n_experts, hidden, dim))

    def params_per_expert(self) -> int:
        return 3 * self.dim * self.hidden

    def _weights(self, dt: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.w_gate.to(dt), self.w_up.to(dt), self.w_down.to(dt)

    def forward_padded(self, x: torch.Tensor) -> torch.Tensor:
        """x: [E, C, D] -> [E, C, D], every expert on its own slice, as 3 batched matmuls."""
        dt = compute_dtype(x)
        wg, wu, wd = self._weights(dt)
        x = x.to(dt)
        return torch.bmm(F.silu(torch.bmm(x, wg)) * torch.bmm(x, wu), wd)

    def forward_grouped(self, x: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        """x: [M, D] rows sorted by expert, offs: [E] int32 cumulative row counts -> [M, D]."""
        dt = compute_dtype(x)
        wg, wu, wd = self._weights(dt)
        x = x.to(dt)
        h = F.silu(_GroupedMM.apply(x, wg, offs)) * _GroupedMM.apply(x, wu, offs)
        return _GroupedMM.apply(h, wd, offs)

    def forward_one(self, e: int, x: torch.Tensor) -> torch.Tensor:
        """Expert e on x: [n, D] (the reference loop path)."""
        dt = compute_dtype(x)
        wg, wu, wd = self._weights(dt)
        x = x.to(dt)
        return (F.silu(x @ wg[e]) * (x @ wu[e])) @ wd[e]


class Gate(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.n_activated_experts
        self.n_groups = cfg.n_expert_groups
        self.topk_groups = cfg.n_limited_groups
        self.route_scale = cfg.route_scale
        # The gate stays in high precision, as in V3.
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.dim))
        # Balancing bias: not trained by gradient, updated by MoE.update_bias().
        self.register_buffer("bias", torch.zeros(cfg.n_routed_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [N, dim] -> (weights [N, k], indices [N, k], scores [N, E])."""
        with full_precision(x):  # router stays fp32 even under bf16 autocast
            scores = torch.sigmoid(F.linear(x.float(), self.weight.float()))
        biased = scores + self.bias

        if self.n_groups > 1:
            # Rank groups by the sum of their top-2 biased scores; keep the best few.
            grouped = biased.view(-1, self.n_groups, self.n_experts // self.n_groups)
            k_in_group = min(2, grouped.shape[-1])
            group_scores = grouped.topk(k_in_group, dim=-1).values.sum(-1)
            top_groups = group_scores.topk(self.topk_groups, dim=-1).indices
            keep = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, top_groups, True)
            biased = grouped.masked_fill(~keep.unsqueeze(-1), float("-inf")).flatten(1)

        indices = biased.topk(self.top_k, dim=-1).indices
        weights = scores.gather(1, indices)  # unbiased affinities
        weights = weights / weights.sum(-1, keepdim=True) * self.route_scale
        return weights, indices, scores


def _expert_counts(flat_idx: torch.Tensor, n_experts: int, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Assignments per expert, as a fixed-size tensor (bincount's output size is data-dependent)."""
    src = torch.ones_like(flat_idx, dtype=torch.float32) if weight is None else weight
    return torch.zeros(n_experts, dtype=src.dtype, device=flat_idx.device).scatter_add_(0, flat_idx, src)


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gate = Gate(cfg)
        self.experts = Experts(cfg.n_routed_experts, cfg.dim, cfg.moe_hidden_dim)
        self.shared = (
            SwiGLU(cfg.dim, cfg.n_shared_experts * cfg.moe_hidden_dim)
            if cfg.n_shared_experts > 0
            else None
        )

    def backend(self) -> str:
        """Training uses the configured backend; evaluation and decoding are always dropless.

        Dropless means grouped_mm where torch has it, else the per-expert loop (which faults the
        ROCm iGPU with 100+ experts, so it's only the fallback).
        """
        if self.training:
            return self.cfg.moe_backend
        return "grouped_mm" if grouped_mm_available() else "sorted"

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """x: [B, T, D]; token_mask: optional [B, T] bool of real (non-padding) tokens.

        Returns (output [B, T, D], stats) where stats holds tensors only:
          load [E]  real-token assignments per expert (for the balancing bias and usage histograms)
          aux       sequence-wise balance loss (training only)
          dropped   fraction of assignments dropped by capacity dispatch (capacity backend only)
        """
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        weights, indices, scores = self.gate(flat)
        K, E = indices.shape[1], self.cfg.n_routed_experts

        mask = token_mask[:, :T] if token_mask is not None else None
        real = None if mask is None else mask.reshape(-1, 1).expand(-1, K).reshape(-1).float()
        stats = {"load": _expert_counts(indices.reshape(-1), E, real)}
        if self.training:
            stats["aux"] = self._sequence_aux_loss(indices.view(B, T, K), scores.view(B, T, E), mask)

        backend = self.backend()
        if backend == "capacity":
            out, stats["dropped"] = self._dispatch_capacity(flat, weights, indices)
        else:
            out = self._dispatch_sorted(flat, weights, indices, grouped=backend == "grouped_mm")

        if self.shared is not None:
            out = out + self.shared(flat)
        return out.view(B, T, D), stats

    def _dispatch_capacity(self, flat, weights, indices) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed-capacity dispatch: scatter into [E, C, D], 3 batched matmuls, gather back."""
        N, D = flat.shape
        K, E = indices.shape[1], self.cfg.n_routed_experts
        C = capacity(N, E, K, self.cfg.capacity_factor)

        # Assignments in k-major order, so every token's first choice is placed before any second
        # choice; within a choice rank, earlier tokens first. Over-capacity assignments are dropped.
        expert = indices.t().reshape(-1)  # [K*N]
        token = torch.arange(N, device=flat.device).repeat(K)
        order = expert.argsort(stable=True)
        counts = _expert_counts(expert, E).long()
        starts = counts.cumsum(0) - counts
        rank_sorted = torch.arange(K * N, device=flat.device) - starts[expert[order]]
        rank = torch.empty_like(rank_sorted).scatter_(0, order, rank_sorted)  # position within its expert
        keep = rank < C
        slot = torch.where(keep, expert * C + rank, E * C)  # dropped assignments -> a spare zero row

        # slot -> source token; unfilled slots read the zero row at index N.
        src = torch.full((E * C + 1,), N, dtype=torch.long, device=flat.device).scatter_(0, slot, token)
        padded = torch.cat([flat, flat.new_zeros(1, D)])
        x_e = padded.index_select(0, src[: E * C]).view(E, C, D)
        y_e = self.experts.forward_padded(x_e).reshape(E * C, D)

        y = torch.cat([y_e, y_e.new_zeros(1, D)]).index_select(0, slot)  # [K*N, D]; dropped -> 0
        out = (y * weights.t().reshape(-1, 1).to(y.dtype)).view(K, N, D).sum(0)
        dropped = 1.0 - keep.float().mean()
        return out.to(flat.dtype), dropped

    def _dispatch_sorted(self, flat, weights, indices, grouped: bool = False) -> torch.Tensor:
        """Sort token-expert pairs by expert, then run each expert on its contiguous slice. Dropless."""
        K, E = indices.shape[1], self.cfg.n_routed_experts
        flat_idx = indices.reshape(-1)
        order = flat_idx.argsort(stable=True)
        tok = order // K
        xs = flat.index_select(0, tok)
        counts = _expert_counts(flat_idx, E).long()
        if grouped:
            ys = self.experts.forward_grouped(xs, counts.cumsum(0).to(torch.int32))
        else:
            ys = torch.cat([self.experts.forward_one(e, chunk)
                            for e, chunk in enumerate(xs.split(counts.tolist())) if chunk.shape[0]])
        y = ys * weights.reshape(-1)[order, None].to(ys.dtype)
        return torch.zeros(flat.shape, dtype=y.dtype, device=flat.device).index_add_(0, tok, y).to(flat.dtype)

    def _sequence_aux_loss(self, indices: torch.Tensor, scores: torch.Tensor,
                           mask: torch.Tensor | None = None) -> torch.Tensor:
        """L_bal = alpha * sum_i f_i * P_i, computed per sequence over real tokens and averaged."""
        B, T, K = indices.shape
        E = self.cfg.n_routed_experts
        w = torch.ones(B, T, device=indices.device) if mask is None else mask.to(scores.dtype)
        n = w.sum(1).clamp(min=1)  # real tokens per sequence
        one_hot = torch.zeros(B, T, E, device=indices.device).scatter_(2, indices, 1.0) * w[..., None]
        f = one_hot.sum(1) * (E / K) / n[:, None]  # [B, E]
        p = ((scores / scores.sum(-1, keepdim=True)) * w[..., None]).sum(1) / n[:, None]  # [B, E]
        return self.cfg.seq_aux_loss_alpha * (f * p).sum(-1).mean()

    @torch.no_grad()
    def update_bias(self, load: torch.Tensor) -> None:
        """Nudge each expert's bias toward balanced load. All tensor ops: no host sync."""
        step = self.cfg.bias_update_speed * torch.sign(load.mean() - load)
        self.gate.bias += torch.where(load.sum() > 0, step, torch.zeros_like(step))


def init_experts(experts: Experts, std: float) -> None:
    for name in Experts.NAMES:
        nn.init.normal_(getattr(experts, name), std=std)



