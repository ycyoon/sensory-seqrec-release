"""Runtime projection and early fusion for the frozen ASER facet bank.

The implementation follows equations (18)--(22) in the manuscript:

* one shared ``bank_dim -> hidden_size`` projection is used for every facet;
* a learned facet-type offset is added before per-facet layer normalization;
* confidence and coverage define ``confidence * log1p(coverage)`` weights;
* unsupported items produce an *exact* zero sensory vector; and
* the sensory vector is added to the ID embedding through a learnable
  sigmoid-constrained scalar initialized to 0.02.

Leading dimensions are unrestricted.  For example, ``canonical`` may be an
item table ``[num_items, facets, bank_dim]`` or an already looked-up sequence
``[batch, length, facets, bank_dim]``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SensoryBankProjector(nn.Module):
    """Convert the frozen multi-facet bank entry into one sensory vector."""

    def __init__(
        self,
        hidden_size: int,
        *,
        bank_dim: int = 768,
        num_facets: int = 5,
        eps: float = 1e-8,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or bank_dim <= 0 or num_facets <= 0:
            raise ValueError("hidden_size, bank_dim, and num_facets must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.hidden_size = hidden_size
        self.bank_dim = bank_dim
        self.num_facets = num_facets
        self.eps = eps

        # This projection is deliberately shared by all five facets.
        self.projection = nn.Linear(bank_dim, hidden_size)
        self.facet_offsets = nn.Parameter(torch.zeros(num_facets, hidden_size))
        self.facet_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.output_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _validate(
        self,
        canonical: Tensor,
        confidence: Tensor,
        coverage: Tensor,
    ) -> None:
        if canonical.ndim < 2:
            raise ValueError("canonical must have shape [..., facets, bank_dim]")
        if canonical.shape[-2:] != (self.num_facets, self.bank_dim):
            raise ValueError(
                "canonical trailing shape must be "
                f"({self.num_facets}, {self.bank_dim}), got {tuple(canonical.shape[-2:])}"
            )
        expected_scalars = canonical.shape[:-1]
        if confidence.shape != expected_scalars:
            raise ValueError(
                f"confidence must have shape {tuple(expected_scalars)}, "
                f"got {tuple(confidence.shape)}"
            )
        if coverage.shape != expected_scalars:
            raise ValueError(
                f"coverage must have shape {tuple(expected_scalars)}, "
                f"got {tuple(coverage.shape)}"
            )

    def quality_weights(self, confidence: Tensor, coverage: Tensor) -> tuple[Tensor, Tensor]:
        """Return normalized facet weights and an item-level support mask.

        Confidence is clamped to its documented range.  Non-positive coverage
        denotes a missing facet regardless of the stored embedding or
        confidence value.
        """

        if confidence.shape != coverage.shape:
            raise ValueError("confidence and coverage must have identical shapes")
        if confidence.shape[-1] != self.num_facets:
            raise ValueError(f"the last dimension must contain {self.num_facets} facets")

        dtype = self.projection.weight.dtype
        confidence_float = confidence.to(dtype=dtype).clamp(0.0, 1.0)
        coverage_float = coverage.to(dtype=dtype)
        present = coverage_float > 0
        safe_coverage = coverage_float.clamp_min(0)
        quality = torch.where(
            present,
            confidence_float * torch.log1p(safe_coverage),
            torch.zeros_like(confidence_float),
        )
        total = quality.sum(dim=-1, keepdim=True)
        supported = total > self.eps
        weights = torch.where(
            supported,
            quality / (total + self.eps),
            torch.zeros_like(quality),
        )
        return weights, supported.squeeze(-1)

    def forward(self, canonical: Tensor, confidence: Tensor, coverage: Tensor) -> Tensor:
        self._validate(canonical, confidence, coverage)

        # Frozen banks are commonly stored in float16 while the lightweight
        # recommender remains float32.  Match the learnable projection dtype.
        canonical = canonical.to(dtype=self.projection.weight.dtype)
        projected = self.projection(canonical)
        offset_shape = (1,) * (projected.ndim - 2) + self.facet_offsets.shape
        projected = self.facet_norm(projected + self.facet_offsets.view(offset_shape))

        weights, supported = self.quality_weights(confidence, coverage)
        mixed = (weights.unsqueeze(-1) * projected).sum(dim=-2)
        sensory = self.output_norm(mixed)

        # LayerNorm and projection biases must never turn a completely missing
        # bank entry into a learned pseudo-feature.
        return torch.where(supported.unsqueeze(-1), sensory, torch.zeros_like(sensory))


class LearnableSensoryBlend(nn.Module):
    """Add a sensory vector using a sigmoid-constrained learnable scalar."""

    def __init__(self, *, initial_weight: float = 0.02, dropout: float = 0.0) -> None:
        super().__init__()
        if not 0.0 < initial_weight < 1.0:
            raise ValueError("initial_weight must lie strictly between zero and one")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        initial_logit = math.log(initial_weight / (1.0 - initial_weight))
        self.logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    @property
    def weight(self) -> Tensor:
        """Current blend strength in the open interval ``(0, 1)``."""

        return torch.sigmoid(self.logit)

    def forward(self, id_embedding: Tensor, sensory: Tensor) -> Tensor:
        if id_embedding.shape != sensory.shape:
            raise ValueError(
                "id_embedding and sensory must have identical shapes, got "
                f"{tuple(id_embedding.shape)} and {tuple(sensory.shape)}"
            )
        return id_embedding + self.weight.to(dtype=id_embedding.dtype) * self.dropout(sensory)


class SensoryFusion(nn.Module):
    """Convenience module combining bank projection and scalar early fusion."""

    def __init__(
        self,
        hidden_size: int,
        *,
        bank_dim: int = 768,
        num_facets: int = 5,
        initial_weight: float = 0.02,
        dropout: float = 0.0,
        min_total_coverage: float = 0.0,
        eps: float = 1e-8,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if min_total_coverage < 0:
            raise ValueError("min_total_coverage cannot be negative")
        self.projector = SensoryBankProjector(
            hidden_size,
            bank_dim=bank_dim,
            num_facets=num_facets,
            eps=eps,
            layer_norm_eps=layer_norm_eps,
        )
        self.blend = LearnableSensoryBlend(
            initial_weight=initial_weight,
            dropout=dropout,
        )
        # Items whose total facet coverage (number of supporting texts summed
        # over facets) is below this threshold receive no sensory contribution.
        # The per-popularity diagnosis showed low-coverage items inject noise
        # rather than signal, so gating them out lets the head-item gain survive
        # instead of being averaged away by a single global blend scalar.
        # 0.0 disables the gate and exactly reproduces the ungated fusion.
        self.min_total_coverage = float(min_total_coverage)

    @property
    def blend_weight(self) -> Tensor:
        return self.blend.weight

    def sensory_vector(
        self,
        canonical: Tensor,
        confidence: Tensor,
        coverage: Tensor,
    ) -> Tensor:
        return self.projector(canonical, confidence, coverage)

    def forward(
        self,
        id_embedding: Tensor,
        canonical: Tensor,
        confidence: Tensor,
        coverage: Tensor,
    ) -> Tensor:
        sensory = self.sensory_vector(canonical, confidence, coverage)
        if self.min_total_coverage > 0.0:
            total_coverage = coverage.to(dtype=sensory.dtype).sum(dim=-1)
            reliable = total_coverage >= self.min_total_coverage
            sensory = torch.where(
                reliable.unsqueeze(-1), sensory, torch.zeros_like(sensory)
            )
        return self.blend(id_embedding, sensory)


# Descriptive aliases retained for callers that use the paper's terminology.
FacetBankProjector = SensoryBankProjector
ASERSensoryFusion = SensoryFusion
