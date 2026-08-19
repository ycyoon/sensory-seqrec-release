"""Small, dependency-free PyTorch sequential backbones for ASER experiments.

All four recommenders expose the same interface:

``encode(item_ids, ...) -> [batch, length, hidden]``
``predict_next(item_ids, ...) -> [batch, num_items]``

Item id ``0`` is reserved for padding and catalog items use ids
``1..num_items``.  The output scorer is tied to the ID embedding table; ASER
changes the input token, as described in the manuscript, rather than silently
changing the candidate scorer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .fusion import SensoryFusion


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(diagonal=1)


def _safe_padding_mask(mask: Tensor) -> Tensor:
    """Avoid all-masked attention rows, whose softmax is undefined."""

    safe = mask.clone()
    all_padding = safe.all(dim=-1)
    if safe.shape[-1] and all_padding.any():
        safe[all_padding, 0] = False
    return safe


class CausalAttentionBlock(nn.Module):
    """Post-norm causal attention and feed-forward block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        *,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def attention_only(
        self,
        hidden_states: Tensor,
        *,
        causal_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        attended, _ = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=causal_mask,
            key_padding_mask=_safe_padding_mask(padding_mask),
            need_weights=False,
        )
        return self.attention_norm(hidden_states + self.attention_dropout(attended))

    def feed_forward_only(self, hidden_states: Tensor) -> Tensor:
        return self.feed_forward_norm(hidden_states + self.feed_forward(hidden_states))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        causal_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        hidden_states = self.attention_only(
            hidden_states,
            causal_mask=causal_mask,
            padding_mask=padding_mask,
        )
        return self.feed_forward_only(hidden_states)


class SequentialRecommender(nn.Module):
    """Shared item embedding, frozen sensory-bank lookup, and scoring API."""

    def __init__(
        self,
        num_items: int,
        *,
        max_seq_len: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        use_sensory: bool = True,
        sensory_bank: Optional[Any] = None,
        bank_dim: int = 768,
        num_facets: int = 5,
        sensory_dropout: Optional[float] = None,
        initial_sensory_weight: float = 0.02,
        sensory_min_coverage: float = 0.0,
        symmetric_fusion: bool = False,
        relational_fusion: bool = False,
    ) -> None:
        super().__init__()
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if max_seq_len <= 0 or hidden_size <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("sequence and model dimensions must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.bank_dim = bank_dim
        self.num_facets = num_facets
        # Symmetric fusion also adds the sensory vector to the *candidate* item
        # embedding at scoring time, so a target's own sensory content directly
        # shifts its score (input-only fusion only reshapes the history query).
        self.symmetric_fusion = bool(symmetric_fusion)

        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.input_dropout = nn.Dropout(dropout)

        if use_sensory:
            self.sensory_fusion: Optional[SensoryFusion] = SensoryFusion(
                hidden_size,
                bank_dim=bank_dim,
                num_facets=num_facets,
                initial_weight=initial_sensory_weight,
                dropout=dropout if sensory_dropout is None else sensory_dropout,
                min_total_coverage=sensory_min_coverage,
            )
        else:
            self.sensory_fusion = None

        # Learned relational fusion: the model scores each candidate by its
        # sensory compatibility with the user's per-facet preference prototype
        # (cos(mu_u^f, z_i^f)) and adds a learned facet-weighted residual to the
        # logits. Unlike additive fusion (unary, input-side) this is a learned
        # *relational* term optimized on the ranking loss. Input stays pure ID.
        self.relational_fusion = bool(relational_fusion)
        if self.relational_fusion:
            # Free signed per-facet weights, small nonzero init so gradients flow
            # from the first step; the model can drive a facet's weight to zero.
            self.relational_facet_weights = nn.Parameter(torch.full((num_facets,), 0.05))

        # The frozen bank is a separately reproducible artifact and is omitted
        # from checkpoints. Reattach it with set_sensory_bank after loading.
        self.register_buffer("_bank_canonical", torch.empty(0), persistent=False)
        self.register_buffer("_bank_confidence", torch.empty(0), persistent=False)
        self.register_buffer("_bank_coverage", torch.empty(0), persistent=False)
        self.register_buffer("_bank_znorm", torch.empty(0), persistent=False)
        self.register_buffer("_bank_quality", torch.empty(0), persistent=False)
        if sensory_bank is not None:
            self.set_sensory_bank_from_mapping(sensory_bank)

        # Training-time sensory regularizer (option A): a frozen sensory kNN
        # graph over items. It shapes the ID embedding during training only --
        # sensorially similar items are pulled together so undertrained (tail)
        # items borrow geometry from well-trained neighbors. It is NOT used at
        # inference: scoring stays pure ID. Attach with set_sensory_regularizer.
        self.register_buffer("_reg_neighbors", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_reg_neighbor_weights", torch.empty(0), persistent=False)
        self._sensory_reg_lambda = 0.0

    def set_sensory_regularizer(
        self,
        neighbors: Tensor,
        neighbor_weights: Tensor,
        *,
        reg_lambda: float,
    ) -> None:
        """Attach a frozen sensory kNN graph for training-time regularization.

        ``neighbors`` is ``[num_items + 1, k]`` of 1-based item ids (row 0 is
        padding and ignored); ``neighbor_weights`` matches and should sum to one
        per row over supported neighbors.
        """

        if reg_lambda < 0:
            raise ValueError("reg_lambda cannot be negative")
        if neighbors.shape != neighbor_weights.shape:
            raise ValueError("neighbors and neighbor_weights must have equal shape")
        if neighbors.shape[0] != self.num_items + 1:
            raise ValueError("neighbor rows must equal num_items + 1")
        self._reg_neighbors = neighbors.to(dtype=torch.long, device=self.item_embedding.weight.device)
        self._reg_neighbor_weights = neighbor_weights.to(
            dtype=self.item_embedding.weight.dtype, device=self.item_embedding.weight.device
        )
        self._sensory_reg_lambda = float(reg_lambda)

    @property
    def has_sensory_regularizer(self) -> bool:
        return self._sensory_reg_lambda > 0.0 and self._reg_neighbors.numel() > 0

    def sensory_smoothness_loss(self, sample_size: int = 1024, generator: Optional[torch.Generator] = None) -> Tensor:
        """Mean squared distance of sampled item embeddings from the (detached)
        centroid of their sensory neighbors. Detaching the neighbor side keeps
        the pull one-directional and prevents mutual embedding collapse."""

        if not self.has_sensory_regularizer:
            return self.item_embedding.weight.sum() * 0.0
        device = self.item_embedding.weight.device
        num_real = self.num_items
        n = min(sample_size, num_real)
        offsets = torch.randint(0, num_real, (n,), generator=generator, device=device)
        items = offsets + 1  # skip padding row 0
        neighbors = self._reg_neighbors[items]  # [n, k]
        weights = self._reg_neighbor_weights[items]  # [n, k]
        valid = neighbors > 0
        # Rows with no supported neighbor contribute nothing.
        has_neighbor = valid.any(dim=1)
        if not bool(has_neighbor.any()):
            return self.item_embedding.weight.sum() * 0.0
        emb = self.item_embedding(items)  # [n, d]
        neighbor_emb = self.item_embedding(neighbors.clamp(min=0))  # [n, k, d]
        w = (weights * valid).unsqueeze(-1)
        centroid = (w * neighbor_emb).sum(dim=1)  # [n, d]
        centroid = centroid.detach()
        diff = (emb - centroid).pow(2).sum(dim=-1)
        diff = diff[has_neighbor]
        return diff.mean()

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def reset_parameters(self) -> None:
        self.apply(self._init_module)

    @property
    def has_sensory_bank(self) -> bool:
        return (
            (self.sensory_fusion is not None or self.relational_fusion)
            and self._bank_canonical.numel() > 0
        )

    @property
    def sensory_blend_weight(self) -> Optional[Tensor]:
        if self.sensory_fusion is None:
            return None
        return self.sensory_fusion.blend_weight

    def _with_padding_row(
        self,
        tensor: Tensor,
        *,
        trailing_shape: tuple[int, ...],
        name: str,
    ) -> Tensor:
        if tensor.ndim != len(trailing_shape) + 1 or tuple(tensor.shape[1:]) != trailing_shape:
            dimensions = ", ".join(map(str, trailing_shape))
            raise ValueError(f"{name} must have shape [items, {dimensions}]")
        if tensor.shape[0] == self.num_items:
            padding = torch.zeros(
                (1,) + trailing_shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = torch.cat((padding, tensor), dim=0)
        elif tensor.shape[0] != self.num_items + 1:
            raise ValueError(
                f"{name} must have {self.num_items} rows (without padding) or "
                f"{self.num_items + 1} rows (with padding)"
            )
        else:
            tensor = tensor.clone()
            tensor[0].zero_()
        return tensor

    def set_sensory_bank(
        self,
        canonical: Tensor,
        confidence: Tensor,
        coverage: Tensor,
    ) -> None:
        """Attach a frozen bank indexed by the model's one-based item ids."""

        if self.sensory_fusion is None and not self.relational_fusion:
            raise RuntimeError("this model was constructed with use_sensory=False")
        canonical = self._with_padding_row(
            canonical.detach(),
            trailing_shape=(self.num_facets, self.bank_dim),
            name="canonical",
        )
        confidence = self._with_padding_row(
            confidence.detach(),
            trailing_shape=(self.num_facets,),
            name="confidence",
        )
        coverage = self._with_padding_row(
            coverage.detach(),
            trailing_shape=(self.num_facets,),
            name="coverage",
        )
        target_device = self.item_embedding.weight.device
        self._bank_canonical = canonical.to(device=target_device)
        self._bank_confidence = confidence.to(device=target_device)
        self._bank_coverage = coverage.to(device=target_device)
        # Precompute per-facet unit vectors and quality weights for the learned
        # relational residual (frozen; only the facet weights are trainable).
        if self.relational_fusion:
            self._bank_znorm = F.normalize(self._bank_canonical, dim=-1)
            self._bank_quality = torch.where(
                self._bank_coverage > 0,
                self._bank_confidence.clamp(0, 1) * torch.log1p(self._bank_coverage),
                torch.zeros_like(self._bank_confidence),
            )

    def set_sensory_bank_from_mapping(self, bank: Any) -> None:
        """Attach a mapping or an ``ItemFacetBank``-style artifact."""

        def get(name: str) -> Optional[Tensor]:
            if isinstance(bank, Mapping):
                value = bank.get(name)
            else:
                value = getattr(bank, name, None)
            return value if isinstance(value, Tensor) else None

        canonical = get("canonical")
        if canonical is None:
            canonical = get("canonical_embeddings")
        if canonical is None:
            canonical = get("facet_embeddings")
        if canonical is None:
            canonical = get("embeddings")
        confidence = get("confidence")
        coverage = get("coverage")
        if canonical is None or confidence is None or coverage is None:
            raise KeyError(
                "bank must provide canonical/canonical_embeddings, confidence, and coverage"
            )
        self.set_sensory_bank(canonical, confidence, coverage)

    def clear_sensory_bank(self) -> None:
        device = self.item_embedding.weight.device
        self._bank_canonical = torch.empty(0, device=device)
        self._bank_confidence = torch.empty(0, device=device)
        self._bank_coverage = torch.empty(0, device=device)

    def _validate_item_ids(self, item_ids: Tensor) -> None:
        if item_ids.ndim != 2:
            raise ValueError("item_ids must have shape [batch, length]")
        if item_ids.shape[1] == 0:
            raise ValueError("item_ids must contain at least one sequence position")
        if item_ids.shape[1] > self.max_seq_len:
            raise ValueError(
                f"sequence length {item_ids.shape[1]} exceeds max_seq_len={self.max_seq_len}"
            )
        if item_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("item_ids must be an integer tensor")
        if item_ids.numel() and (item_ids.min() < 0 or item_ids.max() > self.num_items):
            raise ValueError(f"item ids must lie in [0, {self.num_items}]")

    def _lookup_sensory(self, item_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self._bank_canonical[item_ids],
            self._bank_confidence[item_ids],
            self._bank_coverage[item_ids],
        )

    def _valid_positions(
        self,
        item_ids: Tensor,
        mask_positions: Optional[Tensor] = None,
    ) -> Tensor:
        valid = item_ids.ne(0)
        if mask_positions is not None:
            if mask_positions.shape != item_ids.shape or mask_positions.dtype != torch.bool:
                raise ValueError("mask_positions must be a boolean tensor shaped like item_ids")
            valid = valid | mask_positions
        return valid

    def _build_input_embeddings(
        self,
        item_ids: Tensor,
        *,
        mask_positions: Optional[Tensor] = None,
        mask_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        self._validate_item_ids(item_ids)
        valid = self._valid_positions(item_ids, mask_positions)

        hidden_states = self.item_embedding(item_ids)
        if self.has_sensory_bank and self.sensory_fusion is not None:
            canonical, confidence, coverage = self._lookup_sensory(item_ids)
            hidden_states = self.sensory_fusion(
                hidden_states,
                canonical,
                confidence,
                coverage,
            )

        # Crucially, replacement happens after ID and sensory fusion. A masked
        # item therefore cannot reveal itself through its frozen bank entry.
        if mask_positions is not None:
            if mask_embedding is None:
                raise ValueError("a mask embedding is required when mask_positions is supplied")
            replacement = mask_embedding.view(1, 1, -1).to(dtype=hidden_states.dtype)
            hidden_states = torch.where(
                mask_positions.unsqueeze(-1),
                replacement,
                hidden_states,
            )

        positions = valid.long().cumsum(dim=-1).sub(1).clamp_min(0)
        hidden_states = hidden_states + self.position_embedding(positions)
        hidden_states = self.input_dropout(self.input_norm(hidden_states))
        return hidden_states.masked_fill(~valid.unsqueeze(-1), 0.0)

    def build_input_embeddings(
        self,
        item_ids: Tensor,
        *,
        mask_positions: Optional[Tensor] = None,
    ) -> Tensor:
        if mask_positions is not None and mask_positions.any():
            raise ValueError(f"{type(self).__name__} does not support masked item tokens")
        return self._build_input_embeddings(item_ids)

    def encode(
        self,
        item_ids: Tensor,
        *,
        mask_positions: Optional[Tensor] = None,
        categorical_side_info: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError

    def forward(self, item_ids: Tensor, **kwargs: Any) -> Tensor:
        return self.encode(item_ids, **kwargs)

    def _output_item_table(self) -> Tensor:
        """Item embedding table used for scoring.

        With symmetric fusion the candidate embeddings carry the sensory vector
        too, so a target's own sensory content affects its own score. Otherwise
        the raw ID table is returned (input-only fusion, the default).
        """

        weight = self.item_embedding.weight
        if not (self.symmetric_fusion and self.has_sensory_bank):
            return weight
        assert self.sensory_fusion is not None
        return self.sensory_fusion(
            weight, self._bank_canonical, self._bank_confidence, self._bank_coverage
        )

    def score_items(
        self,
        hidden_states: Tensor,
        candidate_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Score all real items, or an explicitly supplied candidate set."""

        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states has the wrong final dimension")
        table = self._output_item_table()
        if candidate_ids is None:
            return hidden_states @ table[1:].transpose(0, 1)

        candidates = table[candidate_ids]
        if candidate_ids.ndim == 1:
            return hidden_states @ candidates.transpose(0, 1)
        if hidden_states.ndim == 2 and candidate_ids.ndim == 2:
            return torch.einsum("bd,bcd->bc", hidden_states, candidates)
        raise ValueError("batched candidate_ids require [batch, hidden] hidden states")

    def _last_query_index(
        self,
        item_ids: Tensor,
        mask_positions: Optional[Tensor] = None,
    ) -> Tensor:
        valid = self._valid_positions(item_ids, mask_positions)
        indices = torch.arange(item_ids.shape[1], device=item_ids.device)
        return (valid.long() * indices.unsqueeze(0)).max(dim=-1).values

    def _relational_residual(
        self,
        item_ids: Tensor,
        candidate_ids: Optional[Tensor],
    ) -> Tensor:
        """Learned relational residual added to candidate logits.

        Builds the user's per-facet preference prototype ``mu_u^f`` from the
        history's facet vectors, then scores each candidate by
        ``sum_f softplus(w_f) * cos(mu_u^f, z_i^f)`` with learned facet weights.
        """

        znorm = self._bank_znorm            # [N+1, F, D]
        quality = self._bank_quality        # [N+1, F]
        history = item_ids.clamp(min=0)     # [b, L]
        hz = znorm[history]                 # [b, L, F, D]
        hq = quality[history].unsqueeze(-1) * (history > 0).unsqueeze(-1).unsqueeze(-1)
        mu = (hq * hz).sum(dim=1)           # [b, F, D]
        mu = F.normalize(mu, dim=-1)        # unit per facet
        weights = self.relational_facet_weights  # [F], free signed

        if candidate_ids is None:
            cand_z = znorm[1:]              # [N, F, D]
            cand_q = quality[1:]           # [N, F]
            # a[b, i, f] = cos(mu[b,f], z[i,f]) ; weight by candidate facet quality
            a = torch.einsum("bfd,nfd->bnf", mu, cand_z)     # [b, N, F]
            residual = (a * cand_q.unsqueeze(0) * weights).sum(-1)  # [b, N]
        else:
            cand_z = znorm[candidate_ids]  # [b, C, F, D] or [C, F, D]
            cand_q = quality[candidate_ids]
            if candidate_ids.ndim == 1:
                a = torch.einsum("bfd,cfd->bcf", mu, cand_z)
                residual = (a * cand_q.unsqueeze(0) * weights).sum(-1)
            else:
                a = torch.einsum("bfd,bcfd->bcf", mu, cand_z)
                residual = (a * cand_q * weights).sum(-1)
        return residual

    def predict_next(
        self,
        item_ids: Tensor,
        *,
        candidate_ids: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        sequence = self.encode(item_ids, **kwargs)
        query_index = self._last_query_index(item_ids, kwargs.get("mask_positions"))
        batch_index = torch.arange(item_ids.shape[0], device=item_ids.device)
        query = sequence[batch_index, query_index]
        logits = self.score_items(query, candidate_ids)
        if self.relational_fusion and self.has_sensory_bank:
            logits = logits + self._relational_residual(item_ids, candidate_ids)
        return logits

    def next_item_loss(self, item_ids: Tensor, targets: Tensor, **kwargs: Any) -> Tensor:
        if targets.ndim != 1 or targets.shape[0] != item_ids.shape[0]:
            raise ValueError("targets must have shape [batch]")
        if targets.dtype not in (torch.int32, torch.int64):
            raise TypeError("targets must be an integer tensor")
        if targets.numel() and (targets.min() < 1 or targets.max() > self.num_items):
            raise ValueError(f"targets must lie in [1, {self.num_items}]")
        logits = self.predict_next(item_ids, **kwargs)
        return F.cross_entropy(logits, targets.long() - 1)


def _transformer_encoder(
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_size,
        nhead=num_heads,
        dim_feedforward=4 * hidden_size,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)


def _run_causal_transformer(
    encoder: nn.TransformerEncoder,
    hidden_states: Tensor,
    *,
    causal_mask: Tensor,
    padding_mask: Tensor,
) -> Tensor:
    """Run causal layers while preventing left-padding NaN propagation.

    A left-padding query has no unmasked causal key and can therefore become
    NaN inside attention. Valid queries remain finite in that layer, so
    resetting padding rows before the next layer preserves the intended
    key-padding semantics and prevents those NaNs from entering later keys and
    values.
    """

    safe_padding_mask = _safe_padding_mask(padding_mask)
    for layer in encoder.layers:
        hidden_states = layer(
            hidden_states,
            src_mask=causal_mask,
            src_key_padding_mask=safe_padding_mask,
        )
        hidden_states = hidden_states.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )
    if encoder.norm is not None:
        hidden_states = encoder.norm(hidden_states)
    return hidden_states


class SASRec(SequentialRecommender):
    """Causal self-attentive sequential recommendation backbone."""

    def __init__(
        self,
        num_items: int,
        *,
        max_seq_len: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_items,
            max_seq_len=max_seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )
        self.encoder = _transformer_encoder(hidden_size, num_heads, num_layers, dropout)
        self.output_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.reset_parameters()

    def encode(
        self,
        item_ids: Tensor,
        *,
        mask_positions: Optional[Tensor] = None,
        categorical_side_info: Optional[Tensor] = None,
    ) -> Tensor:
        if categorical_side_info is not None:
            raise ValueError("SASRec does not consume categorical_side_info")
        hidden_states = self.build_input_embeddings(item_ids, mask_positions=mask_positions)
        padding_mask = item_ids.eq(0)
        encoded = _run_causal_transformer(
            self.encoder,
            hidden_states,
            causal_mask=_causal_mask(item_ids.shape[1], item_ids.device),
            padding_mask=padding_mask,
        )
        encoded = self.output_norm(encoded)
        return encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class BERT4Rec(SequentialRecommender):
    """Bidirectional masked-item backbone with full-token leakage prevention."""

    def __init__(
        self,
        num_items: int,
        *,
        max_seq_len: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_items,
            max_seq_len=max_seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )
        self.mask_embedding = nn.Parameter(torch.empty(hidden_size))
        self.encoder = _transformer_encoder(hidden_size, num_heads, num_layers, dropout)
        self.output_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.reset_parameters()
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)

    def build_input_embeddings(
        self,
        item_ids: Tensor,
        *,
        mask_positions: Optional[Tensor] = None,
    ) -> Tensor:
        if mask_positions is None:
            mask_positions = torch.zeros_like(item_ids, dtype=torch.bool)
        return self._build_input_embeddings(
            item_ids,
            mask_positions=mask_positions,
            mask_embedding=self.mask_embedding,
        )

    def encode(
        self,
        item_ids: Tensor,
        *,
        mask_positions: Optional[Tensor] = None,
        categorical_side_info: Optional[Tensor] = None,
    ) -> Tensor:
        if categorical_side_info is not None:
            raise ValueError("BERT4Rec does not consume categorical_side_info")
        if mask_positions is None:
            mask_positions = torch.zeros_like(item_ids, dtype=torch.bool)
        hidden_states = self.build_input_embeddings(
            item_ids,
            mask_positions=mask_positions,
        )
        padding_mask = item_ids.eq(0) & ~mask_positions
        encoded = self.encoder(
            hidden_states,
            src_key_padding_mask=_safe_padding_mask(padding_mask),
        )
        encoded = self.output_norm(encoded)
        return encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def masked_item_loss(
        self,
        item_ids: Tensor,
        targets: Tensor,
        mask_positions: Tensor,
    ) -> Tensor:
        """Cross-entropy over every selected masked-item position."""

        if targets.shape != item_ids.shape:
            raise ValueError("targets must have the same shape as item_ids")
        if mask_positions.shape != item_ids.shape or mask_positions.dtype != torch.bool:
            raise ValueError("mask_positions must be a boolean tensor shaped like item_ids")
        if not mask_positions.any():
            raise ValueError("at least one masked position is required")

        selected_targets = targets[mask_positions]
        if (
            selected_targets.dtype not in (torch.int32, torch.int64)
            or selected_targets.min() < 1
            or selected_targets.max() > self.num_items
        ):
            raise ValueError(f"masked targets must be integer ids in [1, {self.num_items}]")

        encoded = self.encode(item_ids, mask_positions=mask_positions)
        logits = self.score_items(encoded[mask_positions])
        return F.cross_entropy(logits, selected_targets.long() - 1)


class BSARecFrequencyLayer(nn.Module):
    """Official BSARec low/high-frequency residual layer.

    The reference implementation converts the public hyperparameter ``c`` to
    ``c // 2 + 1`` retained rFFT bins and scales the high-frequency residual
    with a learned non-negative ``sqrt_beta ** 2`` coefficient.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        cutoff: int = 5,
        dropout: float = 0.2,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        self.cutoff = cutoff
        self.retained_bins = cutoff // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        spectrum = torch.fft.rfft(hidden_states, dim=1, norm="ortho")
        keep = min(self.retained_bins, spectrum.shape[1])
        low_spectrum = torch.zeros_like(spectrum)
        low_spectrum[:, :keep, :] = spectrum[:, :keep, :]
        low_pass = torch.fft.irfft(
            low_spectrum,
            n=hidden_states.shape[1],
            dim=1,
            norm="ortho",
        )
        high_pass = hidden_states - low_pass
        filtered = low_pass + self.sqrt_beta.square() * high_pass
        return self.norm(hidden_states + self.dropout(filtered))


class BSARecBlock(nn.Module):
    """Parallel frequency and self-attention paths from official BSARec."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        alpha: float,
        frequency_cutoff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.alpha = alpha
        self.frequency = BSARecFrequencyLayer(
            hidden_size,
            cutoff=frequency_cutoff,
            dropout=dropout,
        )
        self.attention = CausalAttentionBlock(hidden_size, num_heads, dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        causal_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        frequency_states = self.frequency(hidden_states)
        attention_states = self.attention.attention_only(
            hidden_states,
            causal_mask=causal_mask,
            padding_mask=padding_mask,
        )
        mixed = self.alpha * frequency_states + (1.0 - self.alpha) * attention_states
        return self.attention.feed_forward_only(mixed)


class BSARec(SequentialRecommender):
    """BSARec with its parallel Fourier and causal-attention inductive bias."""

    def __init__(
        self,
        num_items: int,
        *,
        max_seq_len: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        alpha: float = 0.7,
        frequency_cutoff: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_items,
            max_seq_len=max_seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )
        self.blocks = nn.ModuleList(
            [
                BSARecBlock(
                    hidden_size,
                    num_heads,
                    alpha=alpha,
                    frequency_cutoff=frequency_cutoff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.reset_parameters()

    def encode(
        self,
        item_ids: Tensor,
        *,
        mask_positions: Optional[Tensor] = None,
        categorical_side_info: Optional[Tensor] = None,
    ) -> Tensor:
        if categorical_side_info is not None:
            raise ValueError("BSARec does not consume categorical_side_info")
        hidden_states = self.build_input_embeddings(item_ids, mask_positions=mask_positions)
        padding_mask = item_ids.eq(0)
        causal_mask = _causal_mask(item_ids.shape[1], item_ids.device)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                causal_mask=causal_mask,
                padding_mask=padding_mask,
            )
            hidden_states = hidden_states.masked_fill(
                padding_mask.unsqueeze(-1),
                0.0,
            )
        hidden_states = self.output_norm(hidden_states)
        return hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class DIFF(SequentialRecommender):
    """Practical DIFF adapter retaining categorical side information.

    DIFF's original non-sensory categorical attributes remain optional inputs
    through ``categorical_side_info[batch, length, fields]``. ASER sensory
    fusion is applied to the item token *before* either DIFF path, i.e. as
    input-level early fusion.

    Reproducibility caveat:
        The ASER manuscript does not report which categorical fields were
        supplied to DIFF, their vocabularies, the original DIFF fusion
        function, its frequency split, alignment-loss coefficient, or several
        optimization settings. Consequently this compact implementation keeps
        the defining frequency filtering and dual ID/attribute-enriched causal
        paths, but exposes the omitted choices instead of claiming a
        bit-identical reconstruction of the authors' private DIFF adaptation.
    """

    def __init__(
        self,
        num_items: int,
        *,
        max_seq_len: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        side_vocab_sizes: Optional[Sequence[int]] = None,
        frequency_cutoff: int = 3,
        id_path_weight: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_items,
            max_seq_len=max_seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )
        if not 0.0 <= id_path_weight <= 1.0:
            raise ValueError("id_path_weight must lie in [0, 1]")
        self.id_path_weight = id_path_weight
        self.side_vocab_sizes = tuple(side_vocab_sizes or ())
        if any(size <= 0 for size in self.side_vocab_sizes):
            raise ValueError("all side vocabularies must be positive")
        self.side_embeddings = nn.ModuleList(
            [nn.Embedding(size + 1, hidden_size, padding_idx=0) for size in self.side_vocab_sizes]
        )
        self.side_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.early_fusion_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.intermediate_gate = nn.Linear(2 * hidden_size, hidden_size)

        self.id_filters = nn.ModuleList(
            [
                BSARecFrequencyLayer(
                    hidden_size,
                    cutoff=frequency_cutoff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.enriched_filters = nn.ModuleList(
            [
                BSARecFrequencyLayer(
                    hidden_size,
                    cutoff=frequency_cutoff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.id_blocks = nn.ModuleList(
            [CausalAttentionBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.enriched_blocks = nn.ModuleList(
            [CausalAttentionBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.reset_parameters()

    def _side_features(
        self,
        item_ids: Tensor,
        categorical_side_info: Optional[Tensor],
    ) -> Tensor:
        if categorical_side_info is None:
            return torch.zeros(
                *item_ids.shape,
                self.hidden_size,
                dtype=self.item_embedding.weight.dtype,
                device=item_ids.device,
            )
        if not self.side_embeddings:
            raise ValueError("side_vocab_sizes must be configured to use categorical_side_info")
        expected = item_ids.shape + (len(self.side_embeddings),)
        if categorical_side_info.shape != expected:
            raise ValueError(
                f"categorical_side_info must have shape {tuple(expected)}, "
                f"got {tuple(categorical_side_info.shape)}"
            )
        if categorical_side_info.dtype not in (torch.int32, torch.int64):
            raise TypeError("categorical_side_info must be an integer tensor")

        fields = [
            embedding(categorical_side_info[..., index])
            for index, embedding in enumerate(self.side_embeddings)
        ]
        combined = torch.stack(fields, dim=-2).mean(dim=-2)
        combined = self.side_norm(combined)
        return combined.masked_fill(item_ids.eq(0).unsqueeze(-1), 0.0)

    def encode(
        self,
        item_ids: Tensor,
        *,
        mask_positions: Optional[Tensor] = None,
        categorical_side_info: Optional[Tensor] = None,
    ) -> Tensor:
        item_states = self.build_input_embeddings(item_ids, mask_positions=mask_positions)
        side_states = self._side_features(item_ids, categorical_side_info)
        padding_mask = item_ids.eq(0)

        # Attribute-enriched early fusion path.
        enriched_states = self.early_fusion_norm(item_states + side_states)
        # A gated side residual supplies the practical ID-centric/intermediate
        # path while keeping ID states primary.
        gate = torch.sigmoid(self.intermediate_gate(torch.cat((item_states, side_states), dim=-1)))
        id_states = item_states + gate * side_states
        id_states = id_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        enriched_states = enriched_states.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        causal_mask = _causal_mask(item_ids.shape[1], item_ids.device)
        for id_filter, enriched_filter, id_block, enriched_block in zip(
            self.id_filters,
            self.enriched_filters,
            self.id_blocks,
            self.enriched_blocks,
        ):
            id_states = id_block(
                id_filter(id_states),
                causal_mask=causal_mask,
                padding_mask=padding_mask,
            )
            enriched_states = enriched_block(
                enriched_filter(enriched_states),
                causal_mask=causal_mask,
                padding_mask=padding_mask,
            )
            id_states = id_states.masked_fill(
                padding_mask.unsqueeze(-1),
                0.0,
            )
            enriched_states = enriched_states.masked_fill(
                padding_mask.unsqueeze(-1),
                0.0,
            )

        hidden_states = (
            self.id_path_weight * id_states
            + (1.0 - self.id_path_weight) * enriched_states
        )
        hidden_states = self.output_norm(hidden_states)
        return hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)


# Explicit aliases make configuration files readable without imposing a
# separate factory dependency.
SASRecModel = SASRec
BERT4RecModel = BERT4Rec
BSARecModel = BSARec
DIFFModel = DIFF
