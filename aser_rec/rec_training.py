"""Training and evaluation glue for the sequential recommenders.

The paper does not report several optimization and masking hyperparameters.
They are therefore explicit arguments in :class:`RecommendationTrainingConfig`
instead of hidden constants.  The implementation follows these conventions:

* causal models train on every prefix/next-item pair from the leave-one-out
  training sequence;
* batches are left padded with item ID 0 and histories are left truncated;
* BERT4Rec keeps the original item ID plus a separate boolean mask, allowing
  the model to replace the *entire fused token* after sensory lookup;
* optional sampled negatives are deterministic per seed, epoch, and example
  and exclude every known positive for that user;
* validation and test use exact, chunked full-catalogue ranking;
* Base and Sens models receive the same seed, data order, and shared-parameter
  initialization before being trained independently.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from .data import (
    LeaveOneOut,
    PrefixExample,
    SequenceDataset,
    build_prefix_training_examples,
    leave_one_out_split,
)
from .evaluation import (
    DEFAULT_KS,
    FullRankingResult,
    RankingCase,
    evaluate_full_ranking,
)


@dataclass(frozen=True)
class CausalTrainingExample:
    user_id: int
    prefix: Tuple[int, ...]
    target: int
    negative_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class BERTMaskingExample:
    user_id: int
    item_ids: Tuple[int, ...]
    targets: Tuple[int, ...]
    mask_positions: Tuple[bool, ...]


@dataclass(frozen=True)
class RecommenderQuery:
    """A model-ready history stored inside an evaluation ``RankingCase``."""

    user_id: int
    item_ids: Tuple[int, ...]
    mask_positions: Optional[Tuple[bool, ...]] = None
    categorical_side_info: Optional[Tuple[Tuple[int, ...], ...]] = None


def _coerce_splits(
    source: Union[SequenceDataset, Mapping[int, LeaveOneOut]],
) -> Dict[int, LeaveOneOut]:
    if isinstance(source, SequenceDataset):
        return leave_one_out_split(source.sequences)
    return {int(user_id): split for user_id, split in source.items()}


def _infer_num_items(splits: Mapping[int, LeaveOneOut]) -> int:
    items = [
        item
        for split in splits.values()
        for item in (*split.train, split.validation, split.test)
    ]
    if not items:
        raise ValueError("cannot infer num_items from empty splits")
    return max(items)


def _validate_item_side_info(
    item_side_info: Optional[Tensor],
    *,
    num_items: Optional[int] = None,
) -> Optional[Tensor]:
    """Validate and normalize a one-based categorical item-feature table."""

    if item_side_info is None:
        return None
    if not isinstance(item_side_info, Tensor):
        raise TypeError("item_side_info must be a torch.Tensor or None")
    if item_side_info.ndim != 2 or item_side_info.shape[1] < 1:
        raise ValueError("item_side_info must have shape [num_items + 1, fields]")
    if num_items is not None and item_side_info.shape[0] != int(num_items) + 1:
        raise ValueError(
            "item_side_info must have exactly num_items + 1 rows "
            f"({int(num_items) + 1}), got {item_side_info.shape[0]}"
        )
    if item_side_info.dtype not in (torch.int32, torch.int64):
        raise TypeError("item_side_info must be an integer tensor")
    if torch.count_nonzero(item_side_info[0]).item():
        raise ValueError("item_side_info row 0 must contain only zeros")
    if torch.any(item_side_info < 0).item():
        raise ValueError("item_side_info values must be non-negative")
    return item_side_info.detach().to(device="cpu", dtype=torch.long).contiguous()


def deterministic_negative_sample(
    *,
    num_items: int,
    excluded_items: Iterable[int],
    count: int,
    seed: int,
    sample_key: int = 0,
    epoch: int = 0,
    replacement: bool = False,
) -> Tuple[int, ...]:
    """Sample one-based catalogue IDs reproducibly without false negatives.

    Sparse requests use bounded rejection sampling, avoiding construction of
    the full catalogue complement for every training example.  Dense
    exclusions and large unique samples fall back to materializing the
    available IDs, which guarantees termination without rejection blow-ups.
    """

    if num_items < 1:
        raise ValueError("num_items must be positive")
    if count < 0:
        raise ValueError("count cannot be negative")
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    if count == 0:
        return ()
    excluded = {int(item) for item in excluded_items}
    excluded.discard(0)
    if any(item < 1 or item > num_items for item in excluded):
        raise ValueError("excluded item lies outside the catalogue")
    available_count = num_items - len(excluded)
    if available_count == 0:
        raise ValueError("no catalogue items remain for negative sampling")
    if not replacement and count > available_count:
        raise ValueError(
            f"requested {count} unique negatives, but only {available_count} "
            "are available"
        )
    seed_sequence = np.random.SeedSequence(
        [
            int(seed) & 0xFFFFFFFF,
            int(epoch) & 0xFFFFFFFF,
            int(sample_key) & 0xFFFFFFFF,
        ]
    )
    rng = np.random.default_rng(seed_sequence)

    def materialize_available(
        additionally_excluded: Iterable[int] = (),
    ) -> np.ndarray:
        blocked = excluded.union(int(item) for item in additionally_excluded)
        remaining = num_items - len(blocked)
        return np.fromiter(
            (
                item
                for item in range(1, num_items + 1)
                if item not in blocked
            ),
            dtype=np.int64,
            count=remaining,
        )

    # Rejection is efficient while at least a quarter of the catalogue is
    # available.  For unique samples, switch before consuming more than half
    # of that complement because duplicate draws then become progressively
    # expensive.
    use_rejection = (
        available_count * 4 >= num_items
        and (replacement or count * 2 <= available_count)
    )
    if not use_rejection:
        available = materialize_available()
        sampled = rng.choice(available, size=count, replace=replacement)
        return tuple(int(item) for item in np.asarray(sampled).reshape(-1))

    selected: List[int] = []
    selected_set: set[int] = set()
    attempts = 0
    max_attempts = max(64, count * 16)
    while len(selected) < count and attempts < max_attempts:
        needed = count - len(selected)
        batch_size = min(
            max_attempts - attempts,
            max(16, needed * 2),
        )
        candidates = rng.integers(
            1,
            num_items + 1,
            size=batch_size,
            dtype=np.int64,
        )
        attempts += batch_size
        for raw_item in candidates:
            item = int(raw_item)
            if item in excluded:
                continue
            if not replacement and item in selected_set:
                continue
            selected.append(item)
            if not replacement:
                selected_set.add(item)
            if len(selected) == count:
                break

    if len(selected) < count:
        # The bound above makes this exceptional for the sparse path, while
        # still guaranteeing completion for an unlucky deterministic stream.
        additionally_excluded = selected_set if not replacement else ()
        available = materialize_available(additionally_excluded)
        remaining = count - len(selected)
        sampled = rng.choice(available, size=remaining, replace=replacement)
        selected.extend(
            int(item) for item in np.asarray(sampled).reshape(-1)
        )
    return tuple(selected)


class CausalPrefixDataset(Dataset[CausalTrainingExample]):
    """Every causal prefix from the leave-one-out training histories."""

    def __init__(
        self,
        source: Union[SequenceDataset, Mapping[int, LeaveOneOut]],
        *,
        num_items: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        num_negative_samples: int = 0,
        negative_seed: int = 0,
        negative_replacement: bool = False,
    ) -> None:
        self.splits = _coerce_splits(source)
        if isinstance(source, SequenceDataset):
            inferred_items = source.num_items
        else:
            inferred_items = _infer_num_items(self.splits)
        self.num_items = inferred_items if num_items is None else int(num_items)
        if self.num_items < inferred_items:
            raise ValueError("num_items is smaller than an item present in the splits")
        if max_seq_len is not None and max_seq_len < 1:
            raise ValueError("max_seq_len must be positive or None")
        if num_negative_samples < 0:
            raise ValueError("num_negative_samples cannot be negative")
        self.max_seq_len = max_seq_len
        self.num_negative_samples = int(num_negative_samples)
        self.negative_seed = int(negative_seed)
        self.negative_replacement = bool(negative_replacement)
        self.epoch = 0
        self.examples = tuple(
            build_prefix_training_examples(
                self.splits,
                max_prefix_length=max_seq_len,
            )
        )
        self._known_positives = {
            user_id: frozenset(
                (*split.train, split.validation, split.test)
            )
            for user_id, split in self.splits.items()
        }

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> CausalTrainingExample:
        example: PrefixExample = self.examples[index]
        negatives = deterministic_negative_sample(
            num_items=self.num_items,
            excluded_items=self._known_positives[example.user_id],
            count=self.num_negative_samples,
            seed=self.negative_seed,
            sample_key=index,
            epoch=self.epoch,
            replacement=self.negative_replacement,
        )
        return CausalTrainingExample(
            user_id=example.user_id,
            prefix=example.prefix,
            target=example.target,
            negative_ids=negatives,
        )


class CausalLeftPadCollator:
    """Left pad causal histories to an explicit model sequence length."""

    def __init__(
        self,
        max_seq_len: int,
        *,
        item_side_info: Optional[Tensor] = None,
    ) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = int(max_seq_len)
        self.item_side_info = _validate_item_side_info(item_side_info)

    def __call__(self, examples: Sequence[CausalTrainingExample]) -> Dict[str, Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        negative_count = len(examples[0].negative_ids)
        if any(len(example.negative_ids) != negative_count for example in examples):
            raise ValueError("every example must contain the same number of negatives")
        item_ids = torch.zeros(
            len(examples),
            self.max_seq_len,
            dtype=torch.long,
        )
        for row, example in enumerate(examples):
            prefix = example.prefix[-self.max_seq_len :]
            if not prefix:
                raise ValueError("causal prefixes must be non-empty")
            item_ids[row, -len(prefix) :] = torch.as_tensor(prefix, dtype=torch.long)
        batch = {
            "user_ids": torch.tensor(
                [example.user_id for example in examples],
                dtype=torch.long,
            ),
            "item_ids": item_ids,
            "targets": torch.tensor(
                [example.target for example in examples],
                dtype=torch.long,
            ),
            "negative_ids": torch.tensor(
                [example.negative_ids for example in examples],
                dtype=torch.long,
            ).reshape(len(examples), negative_count),
        }
        if self.item_side_info is not None:
            if item_ids.max().item() >= self.item_side_info.shape[0]:
                raise ValueError("an item ID lies outside item_side_info")
            batch["categorical_side_info"] = self.item_side_info[item_ids]
        return batch


class BERT4RecMaskingDataset(Dataset[BERTMaskingExample]):
    """Deterministic masked-item examples from leave-one-out train histories.

    Selected positions retain their original item IDs.  ``mask_positions`` is
    passed separately to ``BERT4Rec.masked_item_loss``, whose model-side input
    builder replaces ID plus sensory fusion with one learned mask embedding.
    """

    def __init__(
        self,
        source: Union[SequenceDataset, Mapping[int, LeaveOneOut]],
        *,
        max_seq_len: int,
        mask_probability: float,
        seed: int = 0,
        ensure_at_least_one_mask: bool = True,
    ) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if not 0.0 <= mask_probability <= 1.0:
            raise ValueError("mask_probability must lie in [0, 1]")
        self.splits = _coerce_splits(source)
        self.max_seq_len = int(max_seq_len)
        self.mask_probability = float(mask_probability)
        self.seed = int(seed)
        self.ensure_at_least_one_mask = bool(ensure_at_least_one_mask)
        self.epoch = 0
        self._users = tuple(
            user_id
            for user_id in sorted(self.splits)
            if len(self.splits[user_id].train) > 0
        )

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._users)

    def __getitem__(self, index: int) -> BERTMaskingExample:
        user_id = self._users[index]
        item_ids = self.splits[user_id].train[-self.max_seq_len :]
        seed_sequence = np.random.SeedSequence(
            [
                self.seed & 0xFFFFFFFF,
                self.epoch & 0xFFFFFFFF,
                int(index) & 0xFFFFFFFF,
            ]
        )
        rng = np.random.default_rng(seed_sequence)
        mask = rng.random(len(item_ids)) < self.mask_probability
        if self.ensure_at_least_one_mask and len(item_ids) and not mask.any():
            mask[int(rng.integers(0, len(item_ids)))] = True
        targets = tuple(
            item if selected else 0
            for item, selected in zip(item_ids, mask)
        )
        return BERTMaskingExample(
            user_id=user_id,
            item_ids=tuple(item_ids),
            targets=targets,
            mask_positions=tuple(bool(value) for value in mask),
        )


class BERT4RecLeftPadCollator:
    """Left pad IDs, targets, and boolean mask positions consistently."""

    def __init__(self, max_seq_len: int) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = int(max_seq_len)

    def __call__(self, examples: Sequence[BERTMaskingExample]) -> Dict[str, Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        shape = (len(examples), self.max_seq_len)
        item_ids = torch.zeros(shape, dtype=torch.long)
        targets = torch.zeros(shape, dtype=torch.long)
        mask_positions = torch.zeros(shape, dtype=torch.bool)
        for row, example in enumerate(examples):
            length = min(len(example.item_ids), self.max_seq_len)
            if length == 0:
                raise ValueError("BERT4Rec training histories must be non-empty")
            item_ids[row, -length:] = torch.as_tensor(
                example.item_ids[-length:],
                dtype=torch.long,
            )
            targets[row, -length:] = torch.as_tensor(
                example.targets[-length:],
                dtype=torch.long,
            )
            mask_positions[row, -length:] = torch.as_tensor(
                example.mask_positions[-length:],
                dtype=torch.bool,
            )
        if not mask_positions.any():
            raise ValueError("a BERT4Rec batch must contain at least one masked position")
        return {
            "user_ids": torch.tensor(
                [example.user_id for example in examples],
                dtype=torch.long,
            ),
            "item_ids": item_ids,
            "targets": targets,
            "mask_positions": mask_positions,
        }


def build_ranking_cases(
    source: Union[SequenceDataset, Mapping[int, LeaveOneOut]],
    *,
    partition: str,
    objective: str,
    max_seq_len: int,
    num_items: Optional[int] = None,
    item_side_info: Optional[Tensor] = None,
) -> List[RankingCase]:
    """Create validation/test queries under the paper's leave-one-out split."""

    if partition not in {"validation", "test"}:
        raise ValueError("partition must be 'validation' or 'test'")
    if objective not in {"causal", "bert4rec"}:
        raise ValueError("objective must be 'causal' or 'bert4rec'")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    splits = _coerce_splits(source)
    if item_side_info is not None and objective == "bert4rec":
        raise ValueError("item_side_info is only supported for causal objectives")
    prepared_side_info = None
    if item_side_info is not None:
        inferred_num_items = (
            source.num_items
            if isinstance(source, SequenceDataset)
            else _infer_num_items(splits)
        )
        catalogue_size = (
            inferred_num_items
            if num_items is None
            else int(num_items)
        )
        if catalogue_size < inferred_num_items:
            raise ValueError("num_items is smaller than an item present in the splits")
        prepared_side_info = _validate_item_side_info(
            item_side_info,
            num_items=catalogue_size,
        )

    cases: List[RankingCase] = []
    for user_id in sorted(splits):
        split = splits[user_id]
        if partition == "validation":
            full_history = split.train
            target = split.validation
        else:
            full_history = (*split.train, split.validation)
            target = split.test

        if objective == "causal":
            retained_history = tuple(full_history[-max_seq_len:])
            if not retained_history:
                raise ValueError("causal evaluation requires a non-empty history")
            model_items = (0,) * (max_seq_len - len(retained_history)) + retained_history
            mask_positions = None
        else:
            history_limit = max_seq_len - 1
            retained_history = (
                tuple(full_history[-history_limit:])
                if history_limit > 0
                else ()
            )
            left_padding = max_seq_len - len(retained_history) - 1
            model_items = (0,) * left_padding + (*retained_history, 0)
            mask_positions = (
                (False,) * left_padding
                + (False,) * len(retained_history)
                + (True,)
            )

        categorical_side_info = None
        if prepared_side_info is not None:
            categorical_side_info = tuple(
                tuple(int(value) for value in row)
                for row in prepared_side_info[list(model_items)].tolist()
            )
        query = RecommenderQuery(
            user_id=user_id,
            item_ids=model_items,
            mask_positions=mask_positions,
            categorical_side_info=categorical_side_info,
        )
        # Seen filtering uses the complete available history, even if the model
        # consumes a left-truncated suffix. The evaluator restores a repeated
        # target if it also appeared earlier.
        cases.append(
            RankingCase(
                query=query,
                target_item=target,
                seen_items=frozenset(full_history),
            )
        )
    return cases


class ModelChunkScoreAdapter:
    """Adapt ``encode``/``score_items`` to evaluation's NumPy chunk callback."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model = model
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        self._cached_query: Optional[RecommenderQuery] = None
        self._cached_hidden: Optional[Tensor] = None

    def clear_cache(self) -> None:
        self._cached_query = None
        self._cached_hidden = None

    def _encode_query(self, query: RecommenderQuery) -> Tensor:
        if query == self._cached_query and self._cached_hidden is not None:
            return self._cached_hidden

        item_ids = torch.tensor(
            [query.item_ids],
            dtype=torch.long,
            device=self.device,
        )
        kwargs: Dict[str, Tensor] = {}
        if query.mask_positions is not None:
            kwargs["mask_positions"] = torch.tensor(
                [query.mask_positions],
                dtype=torch.bool,
                device=self.device,
            )
        if query.categorical_side_info is not None:
            kwargs["categorical_side_info"] = torch.tensor(
                [query.categorical_side_info],
                dtype=torch.long,
                device=self.device,
            )

        self.model.eval()
        with torch.no_grad():
            encoded = self.model.encode(item_ids, **kwargs)
            valid = item_ids.ne(0)
            if "mask_positions" in kwargs:
                valid |= kwargs["mask_positions"]
            positions = torch.arange(item_ids.shape[1], device=self.device)
            last_index = int(
                (valid.long() * positions.unsqueeze(0)).max(dim=-1).values.item()
            )
            hidden = encoded[0, last_index].detach()
        self._cached_query = query
        self._cached_hidden = hidden
        return hidden

    def __call__(self, query: RecommenderQuery, item_ids: np.ndarray) -> Tensor:
        if not isinstance(query, RecommenderQuery):
            raise TypeError("query must be a RecommenderQuery")
        hidden = self._encode_query(query)
        candidates = torch.as_tensor(
            item_ids,
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            return self.model.score_items(hidden, candidates)


def evaluate_recommender(
    model: nn.Module,
    source: Union[SequenceDataset, Mapping[int, LeaveOneOut]],
    *,
    partition: str,
    objective: str,
    max_seq_len: int,
    num_items: Optional[int] = None,
    catalogue_chunk_size: int = 4096,
    ks: Sequence[int] = DEFAULT_KS,
    device: Optional[Union[str, torch.device]] = None,
    item_side_info: Optional[Tensor] = None,
) -> FullRankingResult:
    """Connect a common recommender API to exact full-ranking evaluation."""

    if num_items is None:
        if isinstance(source, SequenceDataset):
            num_items = source.num_items
        else:
            model_num_items = getattr(model, "num_items", None)
            num_items = (
                _infer_num_items(source)
                if model_num_items is None
                else int(model_num_items)
            )
    cases = build_ranking_cases(
        source,
        partition=partition,
        objective=objective,
        max_seq_len=max_seq_len,
        num_items=num_items,
        item_side_info=item_side_info,
    )
    adapter = ModelChunkScoreAdapter(model, device=device)
    return evaluate_full_ranking(
        cases,
        adapter,
        num_items=num_items,
        catalogue_chunk_size=catalogue_chunk_size,
        ks=ks,
    )


@dataclass(frozen=True)
class RecommendationTrainingConfig:
    """All reported and unreported downstream optimization choices."""

    max_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    adam_amsgrad: bool = False
    max_seq_len: int = 50
    mask_probability: float = 0.2
    ensure_at_least_one_mask: bool = True
    num_negative_samples: int = 0
    negative_replacement: bool = False
    gradient_clip_norm: Optional[float] = None
    patience: Optional[int] = 10
    min_delta: float = 0.0
    catalogue_chunk_size: int = 4096
    evaluation_ks: Tuple[int, ...] = DEFAULT_KS
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 0
    device: str = "cpu"
    deterministic_algorithms: bool = False
    sensory_reg_sample_size: int = 1024

    def validate(self) -> None:
        if self.max_epochs < 1 or self.batch_size < 1 or self.max_seq_len < 1:
            raise ValueError("epochs, batch size, and sequence length must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if (
            len(self.adam_betas) != 2
            or not 0 <= self.adam_betas[0] < 1
            or not 0 <= self.adam_betas[1] < 1
            or self.adam_eps <= 0
        ):
            raise ValueError("invalid Adam beta/epsilon configuration")
        if not 0 <= self.mask_probability <= 1:
            raise ValueError("mask_probability must lie in [0, 1]")
        if self.num_negative_samples < 0:
            raise ValueError("num_negative_samples cannot be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")
        if self.patience is not None and self.patience < 1:
            raise ValueError("patience must be positive or None")
        if self.min_delta < 0 or self.catalogue_chunk_size < 1:
            raise ValueError("min_delta must be non-negative and chunk size positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if 10 not in self.evaluation_ks:
            raise ValueError("evaluation_ks must include 10 for NDCG@10 early stopping")


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    training_loss: float
    validation_ndcg_at_10: float
    validation_metrics: Mapping[str, float]


@dataclass
class TrainingResult:
    model: nn.Module
    best_epoch: int
    best_validation: FullRankingResult
    history: Tuple[EpochRecord, ...]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch from one experiment seed."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def synchronize_shared_parameters(source: nn.Module, target: nn.Module) -> Tuple[str, ...]:
    """Copy same-name/same-shape state so Base/Sens share initialization."""

    source_state = source.state_dict()
    target_state = target.state_dict()
    shared = {
        name: tensor.detach().clone()
        for name, tensor in source_state.items()
        if name in target_state and tensor.shape == target_state[name].shape
    }
    target.load_state_dict(shared, strict=False)
    return tuple(sorted(shared))


def _snapshot_state(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _training_loader(
    source: SequenceDataset,
    splits: Mapping[int, LeaveOneOut],
    *,
    objective: str,
    config: RecommendationTrainingConfig,
    item_side_info: Optional[Tensor] = None,
) -> Tuple[Dataset[Any], DataLoader[Any]]:
    if objective == "causal":
        prepared_side_info = _validate_item_side_info(
            item_side_info,
            num_items=source.num_items,
        )
        training_dataset: Dataset[Any] = CausalPrefixDataset(
            splits,
            num_items=source.num_items,
            max_seq_len=config.max_seq_len,
            num_negative_samples=config.num_negative_samples,
            negative_seed=config.seed,
            negative_replacement=config.negative_replacement,
        )
        collator: Callable[[Sequence[Any]], Dict[str, Tensor]] = CausalLeftPadCollator(
            config.max_seq_len,
            item_side_info=prepared_side_info,
        )
    elif objective == "bert4rec":
        if item_side_info is not None:
            raise ValueError("item_side_info is only supported for causal objectives")
        if config.num_negative_samples:
            raise ValueError("sampled negatives are only implemented for causal objectives")
        training_dataset = BERT4RecMaskingDataset(
            splits,
            max_seq_len=config.max_seq_len,
            mask_probability=config.mask_probability,
            seed=config.seed,
            ensure_at_least_one_mask=config.ensure_at_least_one_mask,
        )
        collator = BERT4RecLeftPadCollator(config.max_seq_len)
    else:
        raise ValueError("objective must be 'causal' or 'bert4rec'")
    if len(training_dataset) == 0:
        raise ValueError("no training examples were produced")
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        worker_init_fn=_seed_worker if config.num_workers else None,
        generator=generator,
    )
    return training_dataset, loader


def _batch_loss(
    model: nn.Module,
    batch: Mapping[str, Tensor],
    *,
    objective: str,
    device: torch.device,
) -> Tuple[Tensor, int]:
    item_ids = batch["item_ids"].to(device)
    targets = batch["targets"].to(device)
    categorical_side_info = batch.get("categorical_side_info")
    if objective == "bert4rec":
        if categorical_side_info is not None:
            raise ValueError("item_side_info is only supported for causal objectives")
        mask_positions = batch["mask_positions"].to(device)
        loss = model.masked_item_loss(item_ids, targets, mask_positions)
        weight = int(mask_positions.sum().item())
        return loss, weight

    model_kwargs: Dict[str, Tensor] = {}
    if categorical_side_info is not None:
        model_kwargs["categorical_side_info"] = categorical_side_info.to(device)
    negatives = batch["negative_ids"].to(device)
    if negatives.shape[1] == 0:
        return (
            model.next_item_loss(item_ids, targets, **model_kwargs),
            item_ids.shape[0],
        )
    candidate_ids = torch.cat((targets.unsqueeze(1), negatives), dim=1)
    logits = model.predict_next(
        item_ids,
        candidate_ids=candidate_ids,
        **model_kwargs,
    )
    labels = torch.zeros(item_ids.shape[0], dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels), item_ids.shape[0]


def train_recommender(
    model: nn.Module,
    source: SequenceDataset,
    *,
    objective: str,
    config: RecommendationTrainingConfig,
    splits: Optional[Mapping[int, LeaveOneOut]] = None,
    item_side_info: Optional[Tensor] = None,
) -> TrainingResult:
    """Train one recommender and restore the best validation-NDCG@10 state."""

    config.validate()
    if int(getattr(model, "num_items", source.num_items)) != source.num_items:
        raise ValueError("model and SequenceDataset disagree on num_items")
    if config.max_seq_len > int(getattr(model, "max_seq_len", config.max_seq_len)):
        raise ValueError("config.max_seq_len exceeds the model's max_seq_len")
    if item_side_info is not None and objective == "bert4rec":
        raise ValueError("item_side_info is only supported for causal objectives")
    prepared_side_info = _validate_item_side_info(
        item_side_info,
        num_items=source.num_items,
    )
    seed_everything(config.seed)
    device = torch.device(config.device)
    model.to(device)
    prepared_splits = (
        leave_one_out_split(source.sequences)
        if splits is None
        else {int(user): split for user, split in splits.items()}
    )
    training_dataset, loader = _training_loader(
        source,
        prepared_splits,
        objective=objective,
        config=config,
        item_side_info=prepared_side_info,
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
        amsgrad=config.adam_amsgrad,
    )

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    # Deterministic sampler for the sensory regularizer, on the model's device
    # so it can drive torch.randint without host/device transfers.
    reg_generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    history: List[EpochRecord] = []
    best_score = -math.inf
    best_epoch = 0
    best_state: Optional[Dict[str, Tensor]] = None
    best_validation: Optional[FullRankingResult] = None
    stale_epochs = 0
    try:
        for epoch in range(1, config.max_epochs + 1):
            if isinstance(
                training_dataset,
                (CausalPrefixDataset, BERT4RecMaskingDataset),
            ):
                training_dataset.set_epoch(epoch - 1)
            model.train()
            weighted_loss = 0.0
            example_weight = 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss, weight = _batch_loss(
                    model,
                    batch,
                    objective=objective,
                    device=device,
                )
                if getattr(model, "has_sensory_regularizer", False):
                    loss = loss + model._sensory_reg_lambda * model.sensory_smoothness_loss(
                        sample_size=config.sensory_reg_sample_size,
                        generator=reg_generator,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("training loss became non-finite")
                loss.backward()
                if config.gradient_clip_norm is not None:
                    clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                weighted_loss += float(loss.detach()) * weight
                example_weight += weight

            validation = evaluate_recommender(
                model,
                prepared_splits,
                partition="validation",
                objective=objective,
                max_seq_len=config.max_seq_len,
                num_items=source.num_items,
                catalogue_chunk_size=config.catalogue_chunk_size,
                ks=config.evaluation_ks,
                device=device,
                item_side_info=prepared_side_info,
            )
            score = validation.metrics["NDCG@10"]
            history.append(
                EpochRecord(
                    epoch=epoch,
                    training_loss=weighted_loss / example_weight,
                    validation_ndcg_at_10=score,
                    validation_metrics=MappingProxyType(dict(validation.metrics)),
                )
            )
            if score > best_score + config.min_delta:
                best_score = score
                best_epoch = epoch
                best_state = _snapshot_state(model)
                best_validation = validation
                stale_epochs = 0
            else:
                stale_epochs += 1
                if config.patience is not None and stale_epochs >= config.patience:
                    break
    finally:
        if config.deterministic_algorithms != previous_deterministic:
            torch.use_deterministic_algorithms(previous_deterministic)

    if best_state is None or best_validation is None:
        raise RuntimeError("training finished without a validation checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation=best_validation,
        history=tuple(history),
    )


@dataclass(frozen=True)
class BaseSensRunResult:
    seed: int
    base: TrainingResult
    sensory: TrainingResult
    base_test: FullRankingResult
    sensory_test: FullRankingResult
    synchronized_parameters: Tuple[str, ...]


def train_base_sens_pair(
    model_factory: Callable[[bool], nn.Module],
    source: SequenceDataset,
    *,
    objective: str,
    config: RecommendationTrainingConfig,
    item_side_info: Optional[Tensor] = None,
) -> BaseSensRunResult:
    """Train matched Base/Sens models under one shared experiment seed.

    ``model_factory(False)`` must build Base and ``model_factory(True)`` Sens.
    Shared same-name parameters are copied from Base to Sens before training so
    extra sensory modules cannot shift backbone initialization via RNG order.
    """

    config.validate()
    seed_everything(config.seed)
    base_model = model_factory(False)
    seed_everything(config.seed)
    sensory_model = model_factory(True)
    synchronized = synchronize_shared_parameters(base_model, sensory_model)
    splits = leave_one_out_split(source.sequences)

    base = train_recommender(
        base_model,
        source,
        objective=objective,
        config=config,
        splits=splits,
        item_side_info=item_side_info,
    )
    sensory = train_recommender(
        sensory_model,
        source,
        objective=objective,
        config=config,
        splits=splits,
        item_side_info=item_side_info,
    )
    base_test = evaluate_recommender(
        base.model,
        splits,
        partition="test",
        objective=objective,
        max_seq_len=config.max_seq_len,
        num_items=source.num_items,
        catalogue_chunk_size=config.catalogue_chunk_size,
        ks=config.evaluation_ks,
        device=config.device,
        item_side_info=item_side_info,
    )
    sensory_test = evaluate_recommender(
        sensory.model,
        splits,
        partition="test",
        objective=objective,
        max_seq_len=config.max_seq_len,
        num_items=source.num_items,
        catalogue_chunk_size=config.catalogue_chunk_size,
        ks=config.evaluation_ks,
        device=config.device,
        item_side_info=item_side_info,
    )
    return BaseSensRunResult(
        seed=config.seed,
        base=base,
        sensory=sensory,
        base_test=base_test,
        sensory_test=sensory_test,
        synchronized_parameters=synchronized,
    )


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 200
    tolerance = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        doubled = 2 * iteration
        coefficient = (
            iteration
            * (b - iteration)
            * x
            / ((qam + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c

        coefficient = (
            -(a + iteration)
            * (qab + iteration)
            * x
            / ((a + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= tolerance:
            return result
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    factor = math.exp(log_term)
    if x < (a + 1.0) / (a + b + 2.0):
        value = factor * _beta_continued_fraction(a, b, x) / a
    else:
        value = 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class PairedTTestResult:
    mean_difference: float
    t_statistic: float
    degrees_of_freedom: int
    p_value: float


def paired_t_test(
    base_values: Sequence[float],
    sensory_values: Sequence[float],
) -> PairedTTestResult:
    """Two-sided paired Student t-test without a SciPy dependency."""

    if len(base_values) != len(sensory_values) or len(base_values) < 2:
        raise ValueError("paired samples must have equal lengths of at least two")
    differences = np.asarray(sensory_values, dtype=np.float64) - np.asarray(
        base_values,
        dtype=np.float64,
    )
    if not np.isfinite(differences).all():
        raise ValueError("paired samples must be finite")
    sample_count = differences.size
    mean_difference = float(differences.mean())
    standard_deviation = float(differences.std(ddof=1))
    degrees_of_freedom = sample_count - 1
    if standard_deviation == 0.0:
        if mean_difference == 0.0:
            t_statistic = 0.0
            p_value = 1.0
        else:
            t_statistic = math.copysign(math.inf, mean_difference)
            p_value = 0.0
    else:
        t_statistic = mean_difference / (
            standard_deviation / math.sqrt(sample_count)
        )
        beta_x = degrees_of_freedom / (
            degrees_of_freedom + t_statistic * t_statistic
        )
        p_value = _regularized_incomplete_beta(
            beta_x,
            degrees_of_freedom / 2.0,
            0.5,
        )
    return PairedTTestResult(
        mean_difference=mean_difference,
        t_statistic=t_statistic,
        degrees_of_freedom=degrees_of_freedom,
        p_value=p_value,
    )


@dataclass(frozen=True)
class PairedMetricSummary:
    base_mean: float
    sensory_mean: float
    base_sample_std: float
    sensory_sample_std: float
    mean_difference: float
    relative_change_percent: Optional[float]
    t_statistic: float
    p_value: float


def aggregate_paired_results(
    base_runs: Sequence[Mapping[str, float]],
    sensory_runs: Sequence[Mapping[str, float]],
    *,
    required_runs: int = 5,
) -> Mapping[str, PairedMetricSummary]:
    """Aggregate matched run metrics and compute a t-test per metric."""

    if len(base_runs) != required_runs or len(sensory_runs) != required_runs:
        raise ValueError(f"exactly {required_runs} paired runs are required")
    if required_runs < 2:
        raise ValueError("required_runs must be at least two")
    metric_names = set(base_runs[0])
    for run in (*base_runs, *sensory_runs):
        if set(run) != metric_names:
            raise ValueError("all runs must contain identical metric keys")

    summaries: Dict[str, PairedMetricSummary] = {}
    for metric in sorted(metric_names):
        base_values = np.asarray(
            [float(run[metric]) for run in base_runs],
            dtype=np.float64,
        )
        sensory_values = np.asarray(
            [float(run[metric]) for run in sensory_runs],
            dtype=np.float64,
        )
        if not np.isfinite(base_values).all() or not np.isfinite(sensory_values).all():
            raise ValueError("run metrics must be finite")
        test = paired_t_test(base_values, sensory_values)
        base_mean = float(base_values.mean())
        sensory_mean = float(sensory_values.mean())
        relative_change = (
            None
            if base_mean == 0.0
            else 100.0 * (sensory_mean - base_mean) / base_mean
        )
        summaries[metric] = PairedMetricSummary(
            base_mean=base_mean,
            sensory_mean=sensory_mean,
            base_sample_std=float(base_values.std(ddof=1)),
            sensory_sample_std=float(sensory_values.std(ddof=1)),
            mean_difference=test.mean_difference,
            relative_change_percent=relative_change,
            t_statistic=test.t_statistic,
            p_value=test.p_value,
        )
    return MappingProxyType(summaries)


@dataclass(frozen=True)
class FiveRunPairedResult:
    runs: Tuple[BaseSensRunResult, ...]
    summaries: Mapping[str, PairedMetricSummary]


def run_five_seed_pairs(
    model_factory: Callable[[bool], nn.Module],
    source: SequenceDataset,
    *,
    objective: str,
    config: RecommendationTrainingConfig,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    item_side_info: Optional[Tensor] = None,
) -> FiveRunPairedResult:
    """Train five matched pairs and aggregate their restored-checkpoint tests."""

    if len(seeds) != 5 or len(set(int(seed) for seed in seeds)) != 5:
        raise ValueError("seeds must contain five distinct values")
    runs = tuple(
        train_base_sens_pair(
            model_factory,
            source,
            objective=objective,
            config=replace(config, seed=int(seed)),
            item_side_info=item_side_info,
        )
        for seed in seeds
    )
    summaries = aggregate_paired_results(
        [run.base_test.metrics for run in runs],
        [run.sensory_test.metrics for run in runs],
    )
    return FiveRunPairedResult(runs=runs, summaries=summaries)
