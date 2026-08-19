"""Exact chunked full-ranking evaluation.

Catalogue IDs are assumed to be contiguous in ``[1, num_items]``; ID 0 is
padding and is never ranked.  Items present in the input history are excluded,
except that the held-out target is always restored to the candidate set.

Ranks use a deterministic total order: higher score first, then lower item ID
first for exactly equal scores.  This item-ID tie break avoids optimistic rank
estimates and, unlike chunk-order tie breaks, is invariant to catalogue chunk
size.  Score callbacks must be deterministic and should put models in eval
mode before this evaluator is called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Collection, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

DEFAULT_KS = (5, 10, 20)
ScoreFunction = Callable[[Any, np.ndarray], Any]


@dataclass(frozen=True)
class RankingCase:
    """One next-item query for full-catalogue evaluation."""

    query: Any
    target_item: int
    seen_items: Collection[int]


@dataclass(frozen=True)
class FullRankingResult:
    """Per-query exact ranks and their aggregate HR/NDCG metrics."""

    ranks: Tuple[int, ...]
    metrics: Mapping[str, float]

    def to_dict(self) -> Dict[str, float]:
        return dict(self.metrics)


def _score_array(
    score_fn: ScoreFunction,
    query: Any,
    item_ids: np.ndarray,
) -> np.ndarray:
    raw_scores = score_fn(query, item_ids)
    if hasattr(raw_scores, "detach"):
        raw_scores = raw_scores.detach().cpu().numpy()
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    scores = scores.reshape(-1)
    if scores.size != item_ids.size:
        raise ValueError(
            "score_fn must return one score per item "
            f"(got {scores.size} scores for {item_ids.size} items)"
        )
    if np.isnan(scores).any():
        raise ValueError("score_fn returned NaN; exact ranking is undefined")
    return scores


def exact_full_catalogue_rank(
    case: RankingCase,
    score_fn: ScoreFunction,
    *,
    num_items: int,
    catalogue_chunk_size: int = 4096,
) -> int:
    """Compute one exact rank without materializing a full score vector.

    ``score_fn(query, item_ids)`` receives a one-dimensional ``int64`` NumPy
    array and must return one numeric score per ID.  Torch tensors are accepted
    as outputs and are detached to CPU automatically.
    """

    if num_items < 1:
        raise ValueError("num_items must be positive")
    if catalogue_chunk_size < 1:
        raise ValueError("catalogue_chunk_size must be positive")
    target_item = int(case.target_item)
    if not 1 <= target_item <= num_items:
        raise ValueError("target_item must lie in [1, num_items]")

    excluded = np.zeros(num_items + 1, dtype=np.bool_)
    for seen_item in case.seen_items:
        item_id = int(seen_item)
        if item_id == 0:
            continue
        if not 1 <= item_id <= num_items:
            raise ValueError(f"seen item {item_id} lies outside [1, num_items]")
        excluded[item_id] = True
    excluded[target_item] = False

    target_ids = np.asarray([target_item], dtype=np.int64)
    target_score = float(_score_array(score_fn, case.query, target_ids)[0])

    preceding = 0
    for start in range(1, num_items + 1, catalogue_chunk_size):
        stop = min(num_items + 1, start + catalogue_chunk_size)
        candidate_ids = np.arange(start, stop, dtype=np.int64)
        active = ~excluded[candidate_ids]
        active &= candidate_ids != target_item
        candidate_ids = candidate_ids[active]
        if candidate_ids.size == 0:
            continue
        scores = _score_array(score_fn, case.query, candidate_ids)
        better = scores > target_score
        tied_before_target = (scores == target_score) & (candidate_ids < target_item)
        preceding += int(np.count_nonzero(better | tied_before_target))
    return preceding + 1


def ranking_metrics(
    ranks: Sequence[int],
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, float]:
    """Compute single-relevant-item HR@K and NDCG@K from exact ranks."""

    if len(ranks) == 0:
        raise ValueError("at least one rank is required")
    normalized_ranks = tuple(int(rank) for rank in ranks)
    if any(rank < 1 for rank in normalized_ranks):
        raise ValueError("ranks must be positive")
    normalized_ks = tuple(int(k) for k in ks)
    if len(normalized_ks) == 0 or any(k < 1 for k in normalized_ks):
        raise ValueError("ks must contain positive cutoffs")
    if len(set(normalized_ks)) != len(normalized_ks):
        raise ValueError("ks may not contain duplicates")

    total = float(len(normalized_ranks))
    metrics: Dict[str, float] = {}
    for cutoff in normalized_ks:
        hits = [rank <= cutoff for rank in normalized_ranks]
        metrics[f"HR@{cutoff}"] = sum(hits) / total
        metrics[f"NDCG@{cutoff}"] = (
            sum(
                1.0 / math.log2(rank + 1)
                for rank, hit in zip(normalized_ranks, hits)
                if hit
            )
            / total
        )
    return metrics


def evaluate_full_ranking(
    cases: Iterable[RankingCase],
    score_fn: ScoreFunction,
    *,
    num_items: int,
    catalogue_chunk_size: int = 4096,
    ks: Sequence[int] = DEFAULT_KS,
) -> FullRankingResult:
    """Evaluate exact full-catalogue ranks and aggregate HR/NDCG."""

    ranks = tuple(
        exact_full_catalogue_rank(
            case,
            score_fn,
            num_items=num_items,
            catalogue_chunk_size=catalogue_chunk_size,
        )
        for case in cases
    )
    metrics = ranking_metrics(ranks, ks=ks)
    return FullRankingResult(ranks=ranks, metrics=MappingProxyType(metrics))
