"""Model configuration and presets.

Every size comes in three variants that differ in sparsity. As currently set, they have about the same
activated parameters and increasing totals; they can be scaled independently:

    <size>              dense-ish: a few wide routed experts         (~55-60% of params activated)
    <size>-sparse       5x the experts at half width, 2x top-k       (~30% activated)
    <size>-wide-sparse  the sparse twin with 2x the experts again    (~15-20% activated)

``ModelConfig.from_preset("small-sparse", dim=320)`` starts from a preset and overrides fields.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

MOE_BACKENDS = ("capacity", "grouped_mm", "sorted")


@dataclass
class ModelConfig:
    # --- core transformer ---
    vocab_size: int = 8192  # trained byte-level BPE; each preset sets its own size (see SIZES)
    max_seq_len: int = 1024
    dim: int = 256
    n_layers: int = 4
    n_dense_layers: int = 1  # first k layers use a dense FFN (V3 uses 3 of 61)
    dense_hidden_dim: int = 704
    norm_eps: float = 1e-6
    init_std: float = 0.02
    tie_embeddings: bool = True  # one table for input embedding and output head

    # --- Multi-head Latent Attention (MLA) ---
    n_heads: int = 4
    q_lora_rank: int = 0  # 0 disables query compression (V2-Lite style)
    kv_lora_rank: int = 64  # size of the cached latent c_KV
    qk_nope_head_dim: int = 32  # per-head dims that carry content
    qk_rope_head_dim: int = 16  # per-head dims that carry position (decoupled RoPE)
    v_head_dim: int = 32
    rope_theta: float = 10000.0

    # --- DeepSeekMoE ---
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2  # top-k routed experts per token
    moe_hidden_dim: int = 128  # fine-grained experts are narrow
    n_expert_groups: int = 1  # group-limited (node-limited) routing
    n_limited_groups: int = 1  # groups each token may route into
    route_scale: float = 1.0
    bias_update_speed: float = 1e-3  # gamma for auxiliary-loss-free balancing
    seq_aux_loss_alpha: float = 1e-4  # small complementary sequence-wise loss

    # --- MoE execution (no effect on the maths, except capacity drops) ---
    # capacity:   fixed-size [E, C, D] buffers + 3 batched matmuls; static shapes, compiles. Drops overflow.
    # grouped_mm: torch._grouped_mm over expert-sorted tokens; dropless, no host sync.
    # sorted:     per-expert loop over a sorted batch; exact reference, one host sync per layer.
    # Evaluation and generation always run dropless (grouped_mm when torch has it, else sorted).
    moe_backend: str = "capacity"
    capacity_factor: float = 2.0  # slots per expert = ceil(cf * tokens * top_k / n_experts)

    def __post_init__(self) -> None:
        assert 0 <= self.n_dense_layers <= self.n_layers
        assert self.qk_rope_head_dim % 2 == 0, "RoPE needs an even head dim"
        assert self.n_activated_experts <= self.n_routed_experts
        assert self.n_routed_experts % self.n_expert_groups == 0
        assert self.n_limited_groups <= self.n_expert_groups
        experts_per_group = self.n_routed_experts // self.n_expert_groups
        assert self.n_limited_groups * experts_per_group >= self.n_activated_experts, (
            "group-limited routing must leave at least top-k experts selectable"
        )
        assert self.moe_backend in MOE_BACKENDS, f"moe_backend must be one of {MOE_BACKENDS}"
        assert self.capacity_factor > 0

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def n_moe_layers(self) -> int:
        return self.n_layers - self.n_dense_layers

    def tokens_per_expert(self, train_tokens: float) -> float:
        """Tokens each routed expert sees per MoE layer over a run (assuming balanced routing)."""
        return train_tokens * self.n_activated_experts / self.n_routed_experts

    def param_estimate(self) -> dict[str, int | float]:
        """Exact parameter counts computed from the config, without building the model.

        Matches LittleKediModel.param_counts(). Compare models on ``activated_non_embedding``:
        at small sizes the embedding table is a large share of the activated count but costs
        almost no compute per token.
        """
        d, H = self.dim, self.n_heads
        if self.q_lora_rank > 0:
            q = d * self.q_lora_rank + self.q_lora_rank + self.q_lora_rank * H * self.qk_head_dim
        else:
            q = d * H * self.qk_head_dim
        kv = d * (self.kv_lora_rank + self.qk_rope_head_dim) + self.kv_lora_rank
        kv += self.kv_lora_rank * H * (self.qk_nope_head_dim + self.v_head_dim)
        attn = q + kv + H * self.v_head_dim * d
        expert = 3 * d * self.moe_hidden_dim
        dense_block = 2 * d + attn + 3 * d * self.dense_hidden_dim
        moe_block = (2 * d + attn + self.n_routed_experts * d + self.n_routed_experts * expert
                     + self.n_shared_experts * expert)
        embedding = (1 if self.tie_embeddings else 2) * self.vocab_size * d
        total = embedding + d + self.n_dense_layers * dense_block + self.n_moe_layers * moe_block
        inactive = self.n_moe_layers * (self.n_routed_experts - self.n_activated_experts) * expert
        activated = total - inactive
        return {
            "total": total,
            "activated": activated,
            "embedding": embedding,
            "activated_non_embedding": activated - embedding,
            "fraction_activated": activated / total,
        }

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "ModelConfig":
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Dense-ish presets: a few wide routed experts (v1 shapes, so results stay comparable) ----
# vocab_size is the BPE vocab trained for that size: 8K up to small; 16K from moderate up (better
# compression, at the cost of a bigger embedding share); 32K for base. Variants share their size's vocab.
SIZES: dict[str, dict] = {
    "tiny": dict(
        vocab_size=8192,
        dim=128, n_layers=4, n_dense_layers=1, dense_hidden_dim=352,
        n_heads=4, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        n_routed_experts=8, n_shared_experts=1, n_activated_experts=2, moe_hidden_dim=64,
        max_seq_len=1024,
    ),
    # Deep-and-thin (8 layers at dim 192), tuned for loss over speed.
    "mini": dict(
        vocab_size=8192,
        dim=192, n_layers=8, n_dense_layers=1, dense_hidden_dim=512,
        n_heads=6, kv_lora_rank=48, qk_nope_head_dim=24, qk_rope_head_dim=16, v_head_dim=24,
        n_routed_experts=12, n_shared_experts=1, n_activated_experts=3, moe_hidden_dim=128,
        n_expert_groups=3, n_limited_groups=2, max_seq_len=1024,
    ),
    "small": dict(
        vocab_size=8192,
        dim=256, n_layers=6, n_dense_layers=1, dense_hidden_dim=704,
        n_heads=8, q_lora_rank=128, kv_lora_rank=64,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
        n_routed_experts=16, n_shared_experts=1, n_activated_experts=4, moe_hidden_dim=128,
        n_expert_groups=4, n_limited_groups=2, max_seq_len=1024,
    ),
    # Between small and medium; the sparse twin is the one to match against Wisp-15M at equal compute.
    "moderate": dict(
        vocab_size=16384,
        dim=384, n_layers=8, n_dense_layers=1, dense_hidden_dim=1056,
        n_heads=8, q_lora_rank=192, kv_lora_rank=96,
        qk_nope_head_dim=48, qk_rope_head_dim=16, v_head_dim=48,
        n_routed_experts=24, n_shared_experts=1, n_activated_experts=4, moe_hidden_dim=192,
        n_expert_groups=4, n_limited_groups=2, max_seq_len=1024,
    ),
    "medium": dict(
        vocab_size=16384,
        dim=512, n_layers=8, n_dense_layers=1, dense_hidden_dim=1408,
        n_heads=8, q_lora_rank=256, kv_lora_rank=128,
        qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
        n_routed_experts=24, n_shared_experts=1, n_activated_experts=4, moe_hidden_dim=256,
        n_expert_groups=4, n_limited_groups=2, max_seq_len=2048,
    ),
    "base": dict(
        vocab_size=32768,
        dim=768, n_layers=12, n_dense_layers=2, dense_hidden_dim=2048,
        n_heads=12, q_lora_rank=384, kv_lora_rank=256,
        qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
        n_routed_experts=32, n_shared_experts=2, n_activated_experts=6, moe_hidden_dim=256,
        n_expert_groups=4, n_limited_groups=2, max_seq_len=4096, init_std=0.006,
    ),
}

# ---- Sparse: as set now, about the same activated parameters and ~2x the total ----
# One rule, following fine-grained MoE practice (DeepSeekMoE, Qwen3-Next): 5x the routed experts at
# half the width, 2x as many active per token (so active expert FLOPs match), 2x shared experts at
# half width (so the always-on path keeps its size), and V3-style 8-group routing.
SPARSE_OVERRIDES: dict[str, dict] = {
    "tiny":     dict(n_routed_experts=32,  moe_hidden_dim=32,  n_activated_experts=4,  n_shared_experts=2,
                     n_expert_groups=4, n_limited_groups=2),
    "mini":     dict(n_routed_experts=64,  moe_hidden_dim=64,  n_activated_experts=6,  n_shared_experts=2,
                     n_expert_groups=8, n_limited_groups=4),
    "small":    dict(n_routed_experts=80,  moe_hidden_dim=64,  n_activated_experts=8,  n_shared_experts=2,
                     n_expert_groups=8, n_limited_groups=4),
    "moderate": dict(n_routed_experts=120, moe_hidden_dim=96,  n_activated_experts=8,  n_shared_experts=2,
                     n_expert_groups=8, n_limited_groups=4),
    "medium":   dict(n_routed_experts=120, moe_hidden_dim=128, n_activated_experts=8,  n_shared_experts=2,
                     n_expert_groups=8, n_limited_groups=4),
    "base":     dict(n_routed_experts=160, moe_hidden_dim=128, n_activated_experts=12, n_shared_experts=4,
                     n_expert_groups=8, n_limited_groups=4),
}

# ---- Wide-sparse: the sparse twin with 2x the routed experts (same width and top-k) ----
# Total parameters grow while activated stay fixed. Only worth it with a batched MoE backend
# (capacity / grouped_mm): with a per-expert loop, 2x the experts is ~2x the MoE overhead.
# Watch tokens per expert (ModelConfig.tokens_per_expert): below ~25-50M per layer, experts are
# likely undertrained, so wide-sparse wants a bigger data budget than its sparse twin.
WIDE_SPARSE_EXPERT_MULTIPLIER = 2

VARIANTS = ("", "-sparse", "-wide-sparse")

PRESETS: dict[str, dict] = {}
for _name, _base in SIZES.items():
    _sparse = {**_base, **SPARSE_OVERRIDES[_name]}
    PRESETS[_name] = dict(_base)
    PRESETS[f"{_name}-sparse"] = _sparse
    PRESETS[f"{_name}-wide-sparse"] = {
        **_sparse, "n_routed_experts": _sparse["n_routed_experts"] * WIDE_SPARSE_EXPERT_MULTIPLIER,
    }


def size_of(preset: str) -> str:
    """'small-wide-sparse' -> 'small'."""
    for suffix in ("-wide-sparse", "-sparse"):
        if preset.endswith(suffix):
            return preset[: -len(suffix)]
    return preset


def capacity(n_tokens: int, n_experts: int, top_k: int, capacity_factor: float) -> int:
    """Slots per expert for fixed-capacity dispatch; an expert can never receive more than n_tokens."""
    return max(1, min(n_tokens, math.ceil(capacity_factor * n_tokens * top_k / n_experts)))
