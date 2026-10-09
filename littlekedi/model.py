"""The full model: MLA attention + DeepSeekMoE blocks, tied embeddings, next-token loss only.

v2 drops Multi-Token Prediction: at these sizes its module cost a large share of parameters and
compute per step for no measured gain. The training forward is pure (no ``.item()``, no module
state written), so it compiles to one graph; MoE statistics come back on the output instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .mla import MLA, RMSNorm, precompute_rope
from .moe import Experts, Gate, MoE, SwiGLU, init_experts


def sample_next(
    logits: torch.Tensor,
    context: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Pick the next token from last-position logits [B, V]; returns [B, 1].

    Order: repetition penalty -> temperature -> top-k -> top-p (nucleus) -> sample.
    temperature <= 0 means greedy (after the repetition penalty).
    """
    logits = logits.float()
    if repetition_penalty != 1.0:
        # CTRL-style penalty on every token already in the context: shrink positive
        # logits and push negative ones further down, so repeats get less likely.
        prev = logits.gather(1, context)
        prev = torch.where(prev > 0, prev / repetition_penalty, prev * repetition_penalty)
        logits = logits.scatter(1, context, prev)
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        v, _ = logits.topk(min(top_k, logits.shape[-1]))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(-1)
        # Drop tokens once the probability mass *before* them already exceeds top_p
        # (so the single most likely token is always kept).
        drop = probs.cumsum(-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, order, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over targets that aren't -100 (SFT prompt tokens / padding).

    Branch-free (compiles without graph breaks); 0 with gradient when every target is masked.
    """
    total = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=-100, reduction="sum")
    return total / (targets != -100).sum().clamp(min=1)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, use_moe: bool):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = MLA(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = MoE(cfg) if use_moe else SwiGLU(cfg.dim, cfg.dense_hidden_dim)

    def forward(self, x, cos, sin, token_mask=None, cache=None) -> tuple[torch.Tensor, dict | None]:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        if isinstance(self.ffn, MoE):
            y, stats = self.ffn(self.ffn_norm(x), token_mask)
            return x + y, stats
        return x + self.ffn(self.ffn_norm(x)), None


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    # Detached scalar tensors (lm_loss, aux_loss, drop_rate). Call .item() outside the forward,
    # once per log step, so the training step has no host syncs.
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    # [n_moe_layers, n_routed_experts] real-token assignments per expert in this forward.
    expert_load: torch.Tensor | None = None


class LittleKediModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg, use_moe=i >= cfg.n_dense_layers) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        # Kept in high precision. With tied embeddings the head reuses embed.weight.
        self.head = None if cfg.tie_embeddings else nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        cos, sin = precompute_rope(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(m, (nn.Linear, nn.Embedding, Gate)):
            nn.init.normal_(m.weight, std=std)
        elif isinstance(m, Experts):
            init_experts(m, std)

    def output(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm(h)
        return F.linear(h, self.embed.weight) if self.head is None else self.head(h)

    # ------------------------------------------------------------------ forward

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                token_mask: torch.Tensor | None = None) -> ModelOutput:
        """Training / evaluation forward. idx, targets: [B, T] (targets = idx shifted by 1).

        token_mask: optional [B, T] bool of real (non-padding) positions; padding is excluded
        from MoE load statistics and the balance loss (SFT batches are right-padded).
        """
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"sequence length {T} > max_seq_len {self.cfg.max_seq_len}"
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]

        h = self.embed(idx)
        stats = []
        for layer in self.layers:
            h, s = layer(h, cos, sin, token_mask)
            if s is not None:
                stats.append(s)
        logits = self.output(h)
        load = torch.stack([s["load"] for s in stats]) if stats else None
        if targets is None:
            return ModelOutput(logits, expert_load=load)

        lm_loss = masked_cross_entropy(logits, targets)
        loss = lm_loss
        metrics = {"lm_loss": lm_loss.detach()}
        aux = [s["aux"] for s in stats if "aux" in s]
        if aux:
            aux_loss = torch.stack(aux).sum()
            loss = loss + aux_loss
            metrics["aux_loss"] = aux_loss.detach()
        dropped = [s["dropped"] for s in stats if "dropped" in s]
        if dropped:
            metrics["drop_rate"] = torch.stack(dropped).mean().detach()
        return ModelOutput(logits, loss, metrics, load)

    def forward_cached(self, idx: torch.Tensor, caches: list[dict], start_pos: int) -> torch.Tensor:
        """Incremental forward for generation; returns logits for the last position."""
        T = idx.shape[1]
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        h = self.embed(idx)
        for layer, cache in zip(self.layers, caches):
            h, _ = layer(h, cos, sin, cache=cache)
        return self.output(h[:, -1])

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_ids: set[int] | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Sample using the MLA latent KV cache (the cache holds only c_KV and k_pe)."""
        self.eval()
        window = self.cfg.max_seq_len
        caches: list[dict] = []
        pos = 0
        for _ in range(max_new_tokens):
            if not caches or pos >= window:
                # (Re)prefill. When the window is full, restart from the most recent half.
                ctx = idx[:, -(window // 2 if pos >= window else window) :]
                caches = [{} for _ in self.layers]
                logits = self.forward_cached(ctx, caches, 0)
                pos = ctx.shape[1]
            else:
                logits = self.forward_cached(idx[:, -1:], caches, pos)
                pos += 1
            nxt = sample_next(logits, idx, temperature, top_k, top_p, repetition_penalty)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_ids and idx.shape[0] == 1 and nxt.item() in stop_ids:
                break
        return idx

    # ------------------------------------------------------------------ helpers

    def moe_layers(self) -> list[MoE]:
        return [m for m in self.modules() if isinstance(m, MoE)]

    @torch.no_grad()
    def update_moe_biases(self, load: torch.Tensor | None) -> None:
        """Aux-loss-free balancing step from a step's summed expert_load [n_moe_layers, E].

        Call once after each optimizer step, outside any compiled region. No host sync.
        """
        if load is None:
            return
        for m, layer_load in zip(self.moe_layers(), load):
            m.update_bias(layer_load)

    @staticmethod
    def load_stats(load: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Balance and usage stats from expert_load [n_moe_layers, E] (tensors; .item() when logging).

        max/mean_load_ratio: busiest expert / mean load per layer (ceiling n_routed / n_activated).
        dead_experts: experts with no assignments, summed over layers; needs a large sample (eval).
        min_load_ratio: least-used expert / mean load, worst layer; near 0 means starved experts.
        """
        if load is None:
            return {}
        mean = load.mean(-1).clamp(min=1e-9)
        ratio_max = load.max(-1).values / mean
        return {
            "max_load_ratio": ratio_max.max(),
            "mean_load_ratio": ratio_max.mean(),
            "min_load_ratio": (load.min(-1).values / mean).min(),
            "dead_experts": (load == 0).sum().float(),
        }

    @torch.no_grad()
    def resize_vocab(self, new_size: int) -> None:
        """Grow the embedding (and untied head). New rows start at the mean of the existing rows."""
        old = self.cfg.vocab_size
        if new_size <= old:
            return
        emb = nn.Embedding(new_size, self.cfg.dim).to(self.embed.weight)
        emb.weight[:old] = self.embed.weight
        emb.weight[old:] = self.embed.weight.mean(0)
        self.embed = emb
        if self.head is not None:
            head = nn.Linear(self.cfg.dim, new_size, bias=False).to(self.head.weight)
            head.weight[:old] = self.head.weight
            head.weight[old:] = self.head.weight.mean(0)
            self.head = head
        self.cfg.vocab_size = new_size

    def param_counts(self) -> dict[str, int | float]:
        """Total, activated per token, embedding, and activated non-embedding parameters."""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embed.weight.numel() + (self.head.weight.numel() if self.head is not None else 0)
        moe = self.moe_layers()
        per_expert = moe[0].experts.params_per_expert() if moe else 0
        inactive = len(moe) * (self.cfg.n_routed_experts - self.cfg.n_activated_experts) * per_expert
        activated = total - inactive
        return {
            "total": total,
            "activated": activated,
            "embedding": embedding,
            "activated_non_embedding": activated - embedding,
            "fraction_activated": activated / total,
        }
