"""Deterministic Amazon review preprocessing used by the reproduction.

The manuscript leaves a few preprocessing choices implicit.  This module makes
the following assumptions explicit and testable:

* Amazon files contain one JSON object per gzip-compressed line.  Historical
  dumps sometimes use Python-dict literals, so :func:`ast.literal_eval` is the
  only fallback; arbitrary ``eval`` is never used.
* Duplicate interaction rows are retained.  K-core degrees therefore count
  interaction rows, matching the usual review-dataset interpretation.
* Interactions are ordered by numeric Unix timestamp.  Equal timestamps retain
  source-file order, providing a stable and deterministic tie break.
* K-core filtering removes low-degree users and items simultaneously and
  repeats until convergence.
* Raw user and ASIN strings are sorted lexicographically before assigning
  contiguous IDs.  ID 0 is always reserved for padding.
* Leave-one-out uses the final item for test and the penultimate item for
  validation.  Prefix examples are formed only from the remaining training
  sequence and never contain either held-out target.
"""

from __future__ import annotations

import ast
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

PathLike = Union[str, os.PathLike]
T = TypeVar("T")


@dataclass(frozen=True)
class Interaction:
    """One raw user--item event with its source-file position."""

    user: str
    item: str
    timestamp: Union[int, float]
    source_index: int
    record: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class EncodedInteraction:
    """An interaction after user/item IDs have been made contiguous."""

    user_id: int
    item_id: int
    timestamp: Union[int, float]
    source_index: int


@dataclass(frozen=True)
class IdMappings:
    """Bidirectional mappings with index 0 reserved for padding."""

    user_to_id: Mapping[str, int]
    item_to_id: Mapping[str, int]
    id_to_user: Tuple[Optional[str], ...]
    id_to_item: Tuple[Optional[str], ...]


@dataclass(frozen=True)
class SequenceDataset:
    """Chronological, integer-encoded interaction sequences."""

    mappings: IdMappings
    interactions: Tuple[EncodedInteraction, ...]
    sequences: Mapping[int, Tuple[int, ...]]

    @property
    def num_users(self) -> int:
        return len(self.mappings.id_to_user) - 1

    @property
    def num_items(self) -> int:
        return len(self.mappings.id_to_item) - 1


@dataclass(frozen=True)
class LeaveOneOut:
    """A user's train sequence and single validation/test targets."""

    train: Tuple[int, ...]
    validation: int
    test: int


@dataclass(frozen=True)
class PrefixExample:
    """A causal next-item example generated from a training sequence."""

    user_id: int
    prefix: Tuple[int, ...]
    target: int


@dataclass(frozen=True)
class DatasetStatistics:
    """Interaction statistics reported in the manuscript."""

    users: int
    items: int
    interactions: int
    average_sequence_length: float


PAPER_AMAZON_2014_STATISTICS: Mapping[str, DatasetStatistics] = MappingProxyType(
    {
        "beauty": DatasetStatistics(22_363, 12_101, 198_502, 8.9),
        "sports": DatasetStatistics(35_598, 18_357, 296_337, 8.3),
        "toys": DatasetStatistics(19_412, 11_924, 167_597, 8.6),
        "video_games": DatasetStatistics(24_303, 10_672, 231_780, 9.5),
        "grocery": DatasetStatistics(14_681, 8_713, 151_254, 10.3),
    }
)

_DOMAIN_ALIASES = {
    "beauty": "beauty",
    "sports": "sports",
    "sports_outdoors": "sports",
    "sports_and_outdoors": "sports",
    "toys": "toys",
    "toys_games": "toys",
    "toys_and_games": "toys",
    "games": "video_games",
    "video_games": "video_games",
    "grocery": "grocery",
    "grocery_gourmet_food": "grocery",
    "grocery_and_gourmet_food": "grocery",
}


def parse_json_or_literal(line: str, *, max_chars: int = 16 * 1024 * 1024) -> Dict[str, Any]:
    """Safely parse one Amazon JSON-lines record.

    Official Amazon dumps are JSON, while some older mirrors contain the
    printed representation of a Python dictionary.  ``ast.literal_eval``
    supports the latter without allowing imports, calls, or other executable
    expressions.
    """

    if not isinstance(line, str):
        raise TypeError("line must be text")
    if len(line) > max_chars:
        raise ValueError(f"record exceeds the {max_chars}-character safety limit")
    stripped = line.strip().lstrip("\ufeff")
    if not stripped:
        raise ValueError("record is empty")

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as error:
            raise ValueError("record is neither valid JSON nor a Python literal") from error

    if not isinstance(value, Mapping):
        raise ValueError("each Amazon line must decode to an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError("Amazon record keys must be strings")
    return dict(value)


def iter_amazon_json_gz(
    path: PathLike,
    *,
    max_line_chars: int = 16 * 1024 * 1024,
) -> Iterator[Dict[str, Any]]:
    """Yield objects from a gzip-compressed Amazon JSON-lines file.

    Blank lines are ignored.  Parse errors include the physical line number but
    deliberately omit record contents, which may contain review text.
    """

    with gzip.open(path, mode="rt", encoding="utf-8", errors="strict", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield parse_json_or_literal(line, max_chars=max_line_chars)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{os.fspath(path)}:{line_number}: {error}") from error


def _required_identifier(record: Mapping[str, Any], key: str) -> str:
    if key not in record or record[key] is None:
        raise ValueError(f"record is missing required field {key!r}")
    value = str(record[key]).strip()
    if not value:
        raise ValueError(f"record field {key!r} is empty")
    return value


def _numeric_timestamp(value: Any, key: str) -> Union[int, float]:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"record field {key!r} must be a numeric Unix timestamp")
    if isinstance(value, int):
        return value
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"record field {key!r} must be a numeric Unix timestamp") from error
    if not math.isfinite(parsed):
        raise ValueError(f"record field {key!r} must be finite")
    return int(parsed) if parsed.is_integer() else parsed


def interactions_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    user_key: str = "reviewerID",
    item_key: str = "asin",
    timestamp_key: str = "unixReviewTime",
) -> List[Interaction]:
    """Convert Amazon records to interactions while retaining source order."""

    interactions: List[Interaction] = []
    for source_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError("each record must be a mapping")
        if timestamp_key not in record:
            raise ValueError(f"record is missing required field {timestamp_key!r}")
        interactions.append(
            Interaction(
                user=_required_identifier(record, user_key),
                item=_required_identifier(record, item_key),
                timestamp=_numeric_timestamp(record[timestamp_key], timestamp_key),
                source_index=source_index,
                record=record,
            )
        )
    return interactions


def load_amazon_interactions(
    path: PathLike,
    *,
    user_key: str = "reviewerID",
    item_key: str = "asin",
    timestamp_key: str = "unixReviewTime",
    max_line_chars: int = 16 * 1024 * 1024,
) -> List[Interaction]:
    """Read an Amazon 2014/2018 review dump into typed interactions."""

    return interactions_from_records(
        iter_amazon_json_gz(path, max_line_chars=max_line_chars),
        user_key=user_key,
        item_key=item_key,
        timestamp_key=timestamp_key,
    )


def stable_time_sort(interactions: Iterable[Interaction]) -> List[Interaction]:
    """Sort by user and timestamp, breaking timestamp ties by source order."""

    indexed = list(enumerate(interactions))
    indexed.sort(
        key=lambda pair: (
            pair[1].user,
            pair[1].timestamp,
            pair[1].source_index,
            pair[0],
        )
    )
    return [interaction for _, interaction in indexed]


def iterative_k_core(
    interactions: Iterable[Interaction],
    *,
    min_user_interactions: Optional[int] = 5,
    min_item_interactions: Optional[int] = 5,
) -> List[Interaction]:
    """Apply simultaneous user/item frequency filtering until convergence.

    Passing ``None`` for either threshold disables filtering on that side.
    The original order of retained records is preserved.
    """

    for name, threshold in (
        ("min_user_interactions", min_user_interactions),
        ("min_item_interactions", min_item_interactions),
    ):
        if threshold is not None and threshold < 1:
            raise ValueError(f"{name} must be positive or None")

    current = list(interactions)
    while current:
        user_counts = Counter(event.user for event in current)
        item_counts = Counter(event.item for event in current)
        valid_users = (
            set(user_counts)
            if min_user_interactions is None
            else {
                user
                for user, count in user_counts.items()
                if count >= min_user_interactions
            }
        )
        valid_items = (
            set(item_counts)
            if min_item_interactions is None
            else {
                item
                for item, count in item_counts.items()
                if count >= min_item_interactions
            }
        )
        filtered = [
            event
            for event in current
            if event.user in valid_users and event.item in valid_items
        ]
        if len(filtered) == len(current):
            break
        current = filtered
    return current


def build_sequence_dataset(
    interactions: Iterable[Interaction],
    *,
    min_user_interactions: Optional[int] = None,
    min_item_interactions: Optional[int] = None,
) -> SequenceDataset:
    """Optionally k-core filter, chronologically sort, and integer encode."""

    retained = list(interactions)
    if min_user_interactions is not None or min_item_interactions is not None:
        retained = iterative_k_core(
            retained,
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
        )
    ordered = stable_time_sort(retained)

    raw_users = sorted({event.user for event in ordered})
    raw_items = sorted({event.item for event in ordered})
    user_to_id = {raw_id: index for index, raw_id in enumerate(raw_users, start=1)}
    item_to_id = {raw_id: index for index, raw_id in enumerate(raw_items, start=1)}
    mappings = IdMappings(
        user_to_id=MappingProxyType(user_to_id),
        item_to_id=MappingProxyType(item_to_id),
        id_to_user=(None, *raw_users),
        id_to_item=(None, *raw_items),
    )

    encoded: List[EncodedInteraction] = []
    sequence_lists: Dict[int, List[int]] = defaultdict(list)
    for event in ordered:
        encoded_event = EncodedInteraction(
            user_id=user_to_id[event.user],
            item_id=item_to_id[event.item],
            timestamp=event.timestamp,
            source_index=event.source_index,
        )
        encoded.append(encoded_event)
        sequence_lists[encoded_event.user_id].append(encoded_event.item_id)

    sequences = {
        user_id: tuple(sequence_lists[user_id])
        for user_id in sorted(sequence_lists)
    }
    return SequenceDataset(
        mappings=mappings,
        interactions=tuple(encoded),
        sequences=MappingProxyType(sequences),
    )


def prepare_amazon_dataset(
    path: PathLike,
    *,
    min_user_interactions: Optional[int] = None,
    min_item_interactions: Optional[int] = None,
    user_key: str = "reviewerID",
    item_key: str = "asin",
    timestamp_key: str = "unixReviewTime",
) -> SequenceDataset:
    """Load and prepare a dump in one call.

    Set both minimums to 5 to reproduce the paper's stated 5-core protocol.
    They default to ``None`` because silently re-filtering an already 5-core
    release can otherwise conceal a preprocessing mismatch.
    """

    return build_sequence_dataset(
        load_amazon_interactions(
            path,
            user_key=user_key,
            item_key=item_key,
            timestamp_key=timestamp_key,
        ),
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )


def leave_one_out_split(
    sequences: Mapping[int, Sequence[int]],
    *,
    drop_short: bool = False,
) -> Dict[int, LeaveOneOut]:
    """Hold out the last item for test and penultimate item for validation.

    A sequence needs at least three events to leave a non-empty training
    history.  By default an underspecified sequence raises rather than silently
    changing the evaluation population.
    """

    splits: Dict[int, LeaveOneOut] = {}
    for user_id in sorted(sequences):
        sequence = tuple(int(item) for item in sequences[user_id])
        if any(item <= 0 for item in sequence):
            raise ValueError("sequences may not contain padding/non-positive item IDs")
        if len(sequence) < 3:
            if drop_short:
                continue
            raise ValueError(
                f"user {user_id} has {len(sequence)} interactions; at least 3 are required"
            )
        splits[user_id] = LeaveOneOut(
            train=sequence[:-2],
            validation=sequence[-2],
            test=sequence[-1],
        )
    return splits


def build_prefix_training_examples(
    splits: Mapping[int, LeaveOneOut],
    *,
    max_prefix_length: Optional[int] = None,
) -> List[PrefixExample]:
    """Generate every causal prefix/next-item pair from each train sequence.

    If ``max_prefix_length`` is provided, only the most recent items are kept;
    targets and example order remain unchanged.
    """

    if max_prefix_length is not None and max_prefix_length < 1:
        raise ValueError("max_prefix_length must be positive or None")
    examples: List[PrefixExample] = []
    for user_id in sorted(splits):
        train = splits[user_id].train
        for target_position in range(1, len(train)):
            start = 0
            if max_prefix_length is not None:
                start = max(0, target_position - max_prefix_length)
            examples.append(
                PrefixExample(
                    user_id=user_id,
                    prefix=train[start:target_position],
                    target=train[target_position],
                )
            )
    return examples


def normalize_asin(value: Any) -> str:
    """Return the canonical comparison form used for cross-release overlap."""

    if value is None:
        raise ValueError("ASIN is missing")
    asin = str(value).strip().upper()
    if not asin:
        raise ValueError("ASIN is empty")
    return asin


def _record_asin(record: Any, asin_key: str) -> str:
    if isinstance(record, Interaction):
        return normalize_asin(record.item)
    if isinstance(record, Mapping):
        if asin_key not in record:
            raise ValueError(f"record is missing ASIN field {asin_key!r}")
        return normalize_asin(record[asin_key])
    return normalize_asin(record)


def collect_asins(records: Iterable[Any], *, asin_key: str = "asin") -> FrozenSet[str]:
    """Collect canonical ASINs from mappings, interactions, or raw strings."""

    return frozenset(_record_asin(record, asin_key) for record in records)


def remove_asin_overlap(
    supervision_2018: Iterable[T],
    evaluation_2014: Iterable[Any],
    *,
    asin_key: str = "asin",
) -> Tuple[List[T], FrozenSet[str]]:
    """Remove 2018 records whose ASIN occurs in the 2014 evaluation set.

    The retained list preserves source order.  The returned ASIN set contains
    only overlaps actually encountered in ``supervision_2018``.
    """

    evaluation_asins = collect_asins(evaluation_2014, asin_key=asin_key)
    retained: List[T] = []
    overlap: set[str] = set()
    for record in supervision_2018:
        asin = _record_asin(record, asin_key)
        if asin in evaluation_asins:
            overlap.add(asin)
        else:
            retained.append(record)
    return retained, frozenset(overlap)


def compute_dataset_statistics(
    interactions: Iterable[Interaction],
) -> DatasetStatistics:
    """Compute the four quantities in the paper's Amazon dataset table."""

    events = list(interactions)
    users = {event.user for event in events}
    items = {event.item for event in events}
    average = len(events) / len(users) if users else 0.0
    return DatasetStatistics(
        users=len(users),
        items=len(items),
        interactions=len(events),
        average_sequence_length=average,
    )


def compute_sequence_statistics(
    sequences: Mapping[int, Sequence[int]],
) -> DatasetStatistics:
    """Compute statistics from already encoded chronological sequences."""

    user_sequences = [tuple(sequence) for sequence in sequences.values()]
    if any(item <= 0 for sequence in user_sequences for item in sequence):
        raise ValueError("unpadded sequences must contain only positive item IDs")
    items = {item for sequence in user_sequences for item in sequence}
    interaction_count = sum(len(sequence) for sequence in user_sequences)
    average = interaction_count / len(user_sequences) if user_sequences else 0.0
    return DatasetStatistics(
        users=len(user_sequences),
        items=len(items),
        interactions=interaction_count,
        average_sequence_length=average,
    )


def _canonical_domain(domain: str) -> str:
    normalized = domain.strip().lower().replace("&", "and")
    normalized = "_".join(normalized.replace("-", " ").split())
    try:
        return _DOMAIN_ALIASES[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(PAPER_AMAZON_2014_STATISTICS))
        raise KeyError(f"unknown Amazon domain {domain!r}; choose one of {supported}") from error


def validate_paper_statistics(
    domain: str,
    observed: DatasetStatistics,
    *,
    average_decimals: int = 1,
) -> Dict[str, Tuple[Union[int, float], Union[int, float]]]:
    """Return ``field -> (observed, expected)`` mismatches for a paper domain.

    The paper prints average length to one decimal place, so that field is
    compared after the same rounding rather than as an unreported exact value.
    """

    expected = PAPER_AMAZON_2014_STATISTICS[_canonical_domain(domain)]
    mismatches: Dict[str, Tuple[Union[int, float], Union[int, float]]] = {}
    for statistic_name in ("users", "items", "interactions"):
        actual_value = getattr(observed, statistic_name)
        expected_value = getattr(expected, statistic_name)
        if actual_value != expected_value:
            mismatches[statistic_name] = (actual_value, expected_value)
    rounded_average = round(observed.average_sequence_length, average_decimals)
    if rounded_average != expected.average_sequence_length:
        mismatches["average_sequence_length"] = (
            rounded_average,
            expected.average_sequence_length,
        )
    return mismatches


def assert_paper_statistics(
    domain: str,
    observed: DatasetStatistics,
    *,
    average_decimals: int = 1,
) -> None:
    """Raise a readable error when preprocessing misses a reported statistic."""

    mismatches = validate_paper_statistics(
        domain,
        observed,
        average_decimals=average_decimals,
    )
    if mismatches:
        details = ", ".join(
            f"{field}: observed={actual}, expected={expected}"
            for field, (actual, expected) in mismatches.items()
        )
        raise AssertionError(f"Amazon {domain} statistics do not match the paper ({details})")
