"""Command line entry: train-recommender / evaluate-recommender only.

Handlers are extracted verbatim from the research CLI so this entry
point is the code that produced the reported numbers."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def _add_json_output(
    parser: argparse.ArgumentParser,
    *,
    smoke_alias: bool = False,
) -> None:
    flags = ("--output", "--json-output") if smoke_alias else ("--json-output",)
    parser.add_argument(
        *flags,
        dest="json_output",
        help="also save the command summary as JSON",
    )


def _open_jsonl(path: str) -> Iterable[Mapping[str, Any]]:
    source = Path(path)
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    with opener(source, mode="rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            yield value


def _reordered_bank(
    artifact: Mapping[str, Any],
    item_to_id: Mapping[str, int],
) -> tuple[Dict[str, Any], int]:
    import torch

    from aser_rec.data import normalize_asin

    required = ("canonical", "confidence", "coverage")
    if any(name not in artifact for name in required):
        raise ValueError("bank artifact lacks canonical/confidence/coverage")
    bank_mapping = artifact.get("item_id_to_index")
    if not isinstance(bank_mapping, Mapping):
        raise ValueError("bank artifact lacks item_id_to_index")
    normalized_mapping = {
        normalize_asin(item_id): int(index) for item_id, index in bank_mapping.items()
    }
    canonical = artifact["canonical"]
    confidence = artifact["confidence"]
    coverage = artifact["coverage"]
    output_canonical = torch.zeros(
        len(item_to_id),
        *canonical.shape[1:],
        dtype=canonical.dtype,
    )
    output_confidence = torch.zeros(
        len(item_to_id),
        *confidence.shape[1:],
        dtype=confidence.dtype,
    )
    output_coverage = torch.zeros(
        len(item_to_id),
        *coverage.shape[1:],
        dtype=coverage.dtype,
    )
    matched = 0
    for raw_item, encoded_id in item_to_id.items():
        bank_index = normalized_mapping.get(normalize_asin(raw_item))
        if bank_index is None:
            continue
        row = encoded_id - 1
        output_canonical[row] = canonical[bank_index]
        output_confidence[row] = confidence[bank_index]
        output_coverage[row] = coverage[bank_index]
        matched += 1
    return {
        "canonical": output_canonical,
        "confidence": output_confidence,
        "coverage": output_coverage,
    }, matched


def _build_sensory_neighbor_graph(
    bank_path: str,
    item_to_id: Mapping[str, int],
    *,
    k: int,
) -> tuple[Any, Any, int]:
    """Top-k sensory-cosine neighbor graph over model item ids from a bank.

    Each item's sensory vector is the quality-weighted facet mix of its bank
    canonical embeddings (the same signal the fusion projector consumes).
    Returns ``neighbors`` and ``neighbor_weights`` of shape ``[num_items+1, k]``
    in 1-based item ids (row 0 padding), and the number of items that have at
    least one neighbor.
    """
    import torch

    try:
        artifact = torch.load(bank_path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(bank_path, map_location="cpu")
    reordered, _ = _reordered_bank(artifact, item_to_id)
    canonical = reordered["canonical"].float()  # [N, F, D]
    confidence = reordered["confidence"].float()
    coverage = reordered["coverage"].float()
    quality = torch.where(
        coverage > 0,
        confidence.clamp(0.0, 1.0) * torch.log1p(coverage.clamp_min(0.0)),
        torch.zeros_like(confidence),
    )
    total = quality.sum(dim=-1, keepdim=True)
    weights = torch.where(total > 1e-8, quality / (total + 1e-8), torch.zeros_like(quality))
    mixed = (weights.unsqueeze(-1) * canonical).sum(dim=-2)  # [N, D]
    norms = mixed.norm(dim=-1, keepdim=True)
    unit = torch.where(norms > 0, mixed / norms.clamp_min(1e-12), mixed)
    supported = norms.squeeze(-1) > 0

    num_items = len(item_to_id)
    neighbors = torch.zeros((num_items + 1, k), dtype=torch.long)
    neighbor_weights = torch.zeros((num_items + 1, k), dtype=torch.float32)

    supported_idx = torch.nonzero(supported, as_tuple=False).flatten()
    supported_vecs = unit[supported_idx]  # [S, D]
    matched = 0
    chunk = 512
    for start in range(0, supported_idx.numel(), chunk):
        rows = supported_idx[start : start + chunk]
        sims = unit[rows] @ supported_vecs.T  # [c, S]
        # Exclude self by matching global index.
        self_mask = rows.unsqueeze(1) == supported_idx.unsqueeze(0)
        sims = sims.masked_fill(self_mask, float("-inf"))
        top_values, top_cols = sims.topk(min(k, supported_vecs.shape[0] - 1), dim=1)
        top_global = supported_idx[top_cols]  # [c, k'] 0-based item rows
        positive = top_values > 0
        weight_sum = (top_values * positive).sum(dim=1, keepdim=True).clamp_min(1e-8)
        normalized = (top_values * positive) / weight_sum
        for local, row0 in enumerate(rows.tolist()):
            model_row = row0 + 1  # 1-based item id
            valid = positive[local]
            if not bool(valid.any()):
                continue
            cols = top_global[local] + 1  # 1-based neighbor item ids
            neighbors[model_row, : cols.numel()] = torch.where(
                valid, cols, torch.zeros_like(cols)
            )
            neighbor_weights[model_row, : cols.numel()] = torch.where(
                valid, normalized[local], torch.zeros_like(normalized[local])
            )
            matched += 1
    return neighbors, neighbor_weights, matched


def _load_item_side_info_jsonl(
    path: str,
    item_to_id: Mapping[str, int],
) -> tuple[Any, tuple[int, ...], Dict[str, Any]]:
    """Load one-based categorical item fields in evaluation-catalog order.

    Each JSONL row must contain ``item_id`` (or ``asin``) and a non-empty
    integer ``values`` array. Zero is reserved for missing/padding; positive
    IDs are embedded by DIFF. Rows outside the evaluation catalog are counted
    but ignored.
    """

    import torch

    from aser_rec.data import normalize_asin

    expected_ids = set(range(1, len(item_to_id) + 1))
    if set(int(value) for value in item_to_id.values()) != expected_ids:
        raise ValueError("item_to_id values must be contiguous one-based IDs")
    normalized_catalog: Dict[str, int] = {}
    for raw_item_id, encoded_id in item_to_id.items():
        normalized = normalize_asin(raw_item_id)
        previous = normalized_catalog.setdefault(normalized, int(encoded_id))
        if previous != int(encoded_id):
            raise ValueError(
                f"evaluation catalog contains a normalized ASIN collision: {normalized}"
            )

    field_count: Optional[int] = None
    table: Optional[Any] = None
    seen_items: set[str] = set()
    matched_items: set[int] = set()
    records = 0
    ignored_records = 0
    for record_number, row in enumerate(_open_jsonl(path), start=1):
        records += 1
        raw_item_id = row.get("item_id")
        if raw_item_id is None:
            raw_item_id = row.get("asin")
        if raw_item_id is None:
            raise ValueError(f"{path}: record {record_number}: missing item_id/asin")
        normalized = normalize_asin(raw_item_id)
        if normalized in seen_items:
            raise ValueError(f"{path}: record {record_number}: duplicate item_id/asin {normalized}")
        seen_items.add(normalized)

        raw_values = row.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"{path}: record {record_number}: values must be a non-empty array")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_values
        ):
            raise ValueError(
                f"{path}: record {record_number}: values must be non-negative integers"
            )
        if field_count is None:
            field_count = len(raw_values)
            table = torch.zeros(
                len(item_to_id) + 1,
                field_count,
                dtype=torch.long,
            )
        elif len(raw_values) != field_count:
            raise ValueError(
                f"{path}: record {record_number}: expected {field_count} values, "
                f"got {len(raw_values)}"
            )

        encoded_id = normalized_catalog.get(normalized)
        if encoded_id is None:
            ignored_records += 1
            continue
        assert table is not None
        table[encoded_id] = torch.tensor(raw_values, dtype=torch.long)
        matched_items.add(encoded_id)

    if table is None or field_count is None:
        raise ValueError(f"{path}: no side-information records were found")
    if not matched_items:
        raise ValueError(f"{path}: no side-information records match the evaluation catalog")
    vocabulary_sizes = tuple(int(table[:, field].max().item()) for field in range(field_count))
    if any(size <= 0 for size in vocabulary_sizes):
        raise ValueError(
            "each side-information field must contain at least one positive ID "
            "among matched items"
        )
    matched = len(matched_items)
    return (
        table,
        vocabulary_sizes,
        {
            "path": str(Path(path).resolve()),
            "records": records,
            "ignored_records": ignored_records,
            "matched_items": matched,
            "match_rate": matched / len(item_to_id) if item_to_id else 0.0,
            "field_count": field_count,
            "vocabulary_sizes": list(vocabulary_sizes),
        },
    )


def _handle_train_recommender(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    from aser_rec.data import prepare_amazon_dataset
    from aser_rec.manifest import sha256_file
    from aser_rec.rec_training import (
        RecommendationTrainingConfig,
        evaluate_recommender,
        seed_everything,
        synchronize_shared_parameters,
        train_recommender,
    )
    from aser_rec.recommenders import BERT4Rec, BSARec, DIFF, SASRec

    source = prepare_amazon_dataset(
        args.input,
        min_user_interactions=(args.min_user_interactions if args.iterative_k_core else None),
        min_item_interactions=(args.min_item_interactions if args.iterative_k_core else None),
    )
    if args.item_side_info and args.model != "diff":
        raise ValueError("--item-side-info is only supported for --model diff")
    item_side_info = None
    side_info_summary: Dict[str, Any] = {
        "path": None,
        "records": 0,
        "ignored_records": 0,
        "matched_items": 0,
        "match_rate": None,
        "field_count": 0,
        "vocabulary_sizes": [],
    }
    side_vocab_sizes: tuple[int, ...] = ()
    if args.item_side_info:
        item_side_info, side_vocab_sizes, side_info_summary = _load_item_side_info_jsonl(
            args.item_side_info,
            source.mappings.item_to_id,
        )
        side_info_summary["sha256"] = sha256_file(args.item_side_info)
    else:
        side_info_summary["sha256"] = None
    model_classes = {
        "sasrec": SASRec,
        "bert4rec": BERT4Rec,
        "bsarec": BSARec,
        "diff": DIFF,
    }
    if not 0.0 <= args.minimum_bank_match_rate <= 1.0:
        raise ValueError("--minimum-bank-match-rate must lie in [0, 1]")
    sensory_bank = None
    bank_items_matched = 0
    bank_match_rate = None
    sensory_bank_artifact: Dict[str, Any] = {
        "path": None,
        "sha256": None,
        "persistent_in_model_state": False,
        "reattach_required": False,
    }
    if args.setting == "sens":
        if not args.bank:
            raise ValueError("--bank is required for --setting sens")
        try:
            artifact = torch.load(args.bank, map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(args.bank, map_location="cpu")
        sensory_bank, bank_items_matched = _reordered_bank(
            artifact,
            source.mappings.item_to_id,
        )
        sensory_bank_artifact = {
            "path": str(Path(args.bank).resolve()),
            "sha256": sha256_file(args.bank),
            "persistent_in_model_state": False,
            "reattach_required": True,
        }
        bank_match_rate = bank_items_matched / source.num_items
        if bank_match_rate + 1e-12 < args.minimum_bank_match_rate:
            raise ValueError(
                "sensory bank coverage "
                f"{bank_items_matched}/{source.num_items} "
                f"({bank_match_rate:.6f}) is below "
                "--minimum-bank-match-rate="
                f"{args.minimum_bank_match_rate:.6f}"
            )
    model_kwargs: Dict[str, Any] = {
        "max_seq_len": args.max_seq_len,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "use_sensory": args.setting == "sens" and not getattr(args, "relational_fusion", False),
        "sensory_bank": sensory_bank,
        "initial_sensory_weight": args.initial_sensory_weight,
        "sensory_min_coverage": args.sensory_min_coverage,
        "symmetric_fusion": getattr(args, "symmetric_fusion", False),
        "relational_fusion": getattr(args, "relational_fusion", False),
    }
    if sensory_bank is not None:
        model_kwargs.update(
            bank_dim=int(sensory_bank["canonical"].shape[-1]),
            num_facets=int(sensory_bank["canonical"].shape[-2]),
        )
    if args.model == "bsarec":
        model_kwargs.update(
            alpha=args.bsarec_alpha,
            frequency_cutoff=(5 if args.frequency_cutoff is None else args.frequency_cutoff),
        )
    if args.model == "diff":
        model_kwargs.update(
            frequency_cutoff=(3 if args.frequency_cutoff is None else args.frequency_cutoff),
            id_path_weight=args.diff_id_path_weight,
            side_vocab_sizes=side_vocab_sizes,
        )
    model_class = model_classes[args.model]
    synchronized_parameters: Sequence[str] = ()
    if args.setting == "sens":
        base_template_kwargs = {
            **model_kwargs,
            "use_sensory": False,
            "sensory_bank": None,
            "relational_fusion": False,
        }
        seed_everything(args.seed)
        base_template = model_class(
            source.num_items,
            **base_template_kwargs,
        )
        seed_everything(args.seed)
        model = model_class(source.num_items, **model_kwargs)
        synchronized_parameters = synchronize_shared_parameters(
            base_template,
            model,
        )
        del base_template
    else:
        seed_everything(args.seed)
        model = model_class(source.num_items, **model_kwargs)

    # Training-time sensory regularizer (option A): a frozen sensory kNN graph
    # shapes the ID embedding during training. Inference stays pure ID, so this
    # is available for the base setting and needs no bank at eval time.
    sensory_reg_summary: Dict[str, Any] = {"enabled": False}
    if getattr(args, "sensory_reg_lambda", 0.0) and args.sensory_reg_lambda > 0.0:
        if not args.sensory_reg_bank:
            raise ValueError("--sensory-reg-lambda requires --sensory-reg-bank")
        neighbors, weights, matched = _build_sensory_neighbor_graph(
            args.sensory_reg_bank,
            source.mappings.item_to_id,
            k=args.sensory_reg_k,
        )
        model.set_sensory_regularizer(
            neighbors, weights, reg_lambda=args.sensory_reg_lambda
        )
        sensory_reg_summary = {
            "enabled": True,
            "bank": str(Path(args.sensory_reg_bank).resolve()),
            "k": int(args.sensory_reg_k),
            "lambda": float(args.sensory_reg_lambda),
            "items_with_neighbors": int(matched),
        }
    objective = "bert4rec" if args.model == "bert4rec" else "causal"
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else (1e-3 if args.model == "sasrec" else 1e-4)
    )
    config = RecommendationTrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        mask_probability=args.mask_probability,
        num_negative_samples=args.negative_samples,
        gradient_clip_norm=args.gradient_clip_norm,
        patience=args.patience,
        min_delta=args.min_delta,
        catalogue_chunk_size=args.catalogue_chunk_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        deterministic_algorithms=args.deterministic_algorithms,
        sensory_reg_sample_size=getattr(args, "sensory_reg_sample_size", 1024),
    )
    training = train_recommender(
        model,
        source,
        objective=objective,
        config=config,
        item_side_info=item_side_info,
    )
    test_result = None
    if not args.skip_test_evaluation:
        test_result = evaluate_recommender(
            training.model,
            source,
            partition="test",
            objective=objective,
            max_seq_len=args.max_seq_len,
            catalogue_chunk_size=args.catalogue_chunk_size,
            device=args.device,
            item_side_info=item_side_info,
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_model_kwargs = {
        key: value for key, value in model_kwargs.items() if key != "sensory_bank"
    }
    training_config = json.loads(json.dumps(asdict(config)))
    torch.save(
        {
            "model": training.model.state_dict(),
            "model_name": args.model,
            "model_kwargs": checkpoint_model_kwargs,
            "objective": objective,
            "training_config": training_config,
            "setting": args.setting,
            "seed": args.seed,
            "best_epoch": training.best_epoch,
            "history": [
                {
                    "epoch": record.epoch,
                    "training_loss": record.training_loss,
                    "validation_ndcg_at_10": record.validation_ndcg_at_10,
                    "validation_metrics": dict(record.validation_metrics),
                }
                for record in training.history
            ],
            "test_metrics": (
                test_result.to_dict() if test_result is not None else None
            ),
            "raw_user_to_id": dict(source.mappings.user_to_id),
            "raw_item_to_id": dict(source.mappings.item_to_id),
            "bank_path": args.bank,
            "sensory_bank_artifact": sensory_bank_artifact,
            "bank_items_matched": bank_items_matched,
            "bank_match_rate": bank_match_rate,
            "minimum_bank_match_rate": args.minimum_bank_match_rate,
            "synchronized_parameter_count": len(synchronized_parameters),
            "sensory_regularizer": sensory_reg_summary,
            "item_side_info": side_info_summary,
            "item_side_info_table": (
                item_side_info.detach().cpu().clone() if item_side_info is not None else None
            ),
        },
        destination,
    )
    return {
        "command": "train-recommender",
        "output": str(destination.resolve()),
        "model": args.model,
        "setting": args.setting,
        "objective": objective,
        "training_config": training_config,
        "users": source.num_users,
        "items": source.num_items,
        "bank_items_matched": bank_items_matched,
        "bank_match_rate": bank_match_rate,
        "minimum_bank_match_rate": args.minimum_bank_match_rate,
        "synchronized_parameter_count": len(synchronized_parameters),
        "item_side_info": side_info_summary,
        "sensory_bank_artifact": sensory_bank_artifact,
        "sensory_regularizer": sensory_reg_summary,
        "diff_side_info_enabled": item_side_info is not None,
        "best_epoch": training.best_epoch,
        "validation": training.best_validation.to_dict(),
        "test": test_result.to_dict() if test_result is not None else None,
    }


def _handle_evaluate_recommender(args: argparse.Namespace) -> Dict[str, Any]:
    """Evaluate a selected checkpoint without retraining or touching validation."""
    import torch

    from aser_rec.data import prepare_amazon_dataset
    from aser_rec.rec_training import evaluate_recommender
    from aser_rec.recommenders import BERT4Rec, BSARec, DIFF, SASRec

    try:
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    source = prepare_amazon_dataset(args.input)
    if dict(checkpoint.get("raw_item_to_id", {})) != dict(
        source.mappings.item_to_id
    ):
        raise ValueError("checkpoint item mapping differs from evaluation input")
    if dict(checkpoint.get("raw_user_to_id", {})) != dict(
        source.mappings.user_to_id
    ):
        raise ValueError("checkpoint user mapping differs from evaluation input")

    model_classes = {
        "sasrec": SASRec,
        "bert4rec": BERT4Rec,
        "bsarec": BSARec,
        "diff": DIFF,
    }
    model_name = str(checkpoint["model_name"])
    model_kwargs = dict(checkpoint["model_kwargs"])
    bank_path = args.bank or checkpoint.get("bank_path")
    bank_items_matched = 0
    if checkpoint.get("setting") == "sens":
        if not bank_path:
            raise ValueError("Sens checkpoint requires --bank or checkpoint bank_path")
        try:
            artifact = torch.load(bank_path, map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(bank_path, map_location="cpu")
        sensory_bank, bank_items_matched = _reordered_bank(
            artifact,
            source.mappings.item_to_id,
        )
        model_kwargs["sensory_bank"] = sensory_bank
    model = model_classes[model_name](source.num_items, **model_kwargs)
    model.load_state_dict(checkpoint["model"])
    item_side_info = checkpoint.get("item_side_info_table")
    metrics = evaluate_recommender(
        model,
        source,
        partition=args.partition,
        objective=str(checkpoint["objective"]),
        max_seq_len=int(checkpoint["training_config"]["max_seq_len"]),
        catalogue_chunk_size=args.catalogue_chunk_size,
        device=args.device,
        item_side_info=item_side_info,
    )
    return {
        "command": "evaluate-recommender",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input": str(Path(args.input).resolve()),
        "model": model_name,
        "setting": checkpoint.get("setting"),
        "best_epoch": checkpoint.get("best_epoch"),
        "partition": args.partition,
        "bank": str(Path(bank_path).resolve()) if bank_path else None,
        "bank_items_matched": bank_items_matched,
        "metrics": metrics.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="aser-rec")
    subparsers = parser.add_subparsers(required=True)
    
    recommender = subparsers.add_parser(
        "train-recommender",
        help="train one Base or Sens sequential recommender",
    )
    recommender.add_argument("--input", required=True)
    recommender.add_argument(
        "--model",
        choices=("sasrec", "bert4rec", "bsarec", "diff"),
        required=True,
    )
    recommender.add_argument(
        "--setting",
        choices=("base", "sens"),
        required=True,
    )
    recommender.add_argument("--bank")
    recommender.add_argument(
        "--minimum-bank-match-rate",
        type=float,
        default=1.0,
        help=(
            "minimum fraction of evaluation-catalog items required in a Sens "
            "bank (default: complete coverage)"
        ),
    )
    recommender.add_argument("--output", required=True)
    recommender.add_argument("--max-seq-len", type=int, default=50)
    recommender.add_argument("--hidden-size", type=int, default=64)
    recommender.add_argument("--num-layers", type=int, default=2)
    recommender.add_argument("--num-heads", type=int, default=2)
    recommender.add_argument("--dropout", type=float, default=0.2)
    recommender.add_argument("--initial-sensory-weight", type=float, default=0.02)
    recommender.add_argument(
        "--sensory-reg-lambda",
        type=float,
        default=0.0,
        help=(
            "training-time sensory embedding regularizer weight; pulls "
            "sensorially similar items' ID embeddings together during training "
            "only (inference stays pure ID). 0.0 disables it"
        ),
    )
    recommender.add_argument("--sensory-reg-bank", help="bank for the regularizer neighbor graph")
    recommender.add_argument("--sensory-reg-k", type=int, default=10)
    recommender.add_argument("--sensory-reg-sample-size", type=int, default=1024)
    recommender.add_argument(
        "--symmetric-fusion",
        action="store_true",
        help=(
            "also add the sensory vector to candidate embeddings at scoring "
            "time, so a target's own sensory content shifts its score"
        ),
    )
    recommender.add_argument(
        "--relational-fusion",
        action="store_true",
        help=(
            "learn a candidate-level relational residual (user facet prototype "
            "vs candidate facet compatibility) instead of additive fusion; "
            "requires --setting sens --bank"
        ),
    )
    recommender.add_argument(
        "--sensory-min-coverage",
        type=float,
        default=0.0,
        help=(
            "suppress the sensory contribution for items whose summed facet "
            "coverage is below this threshold; 0.0 keeps every supported item "
            "(ungated fusion)"
        ),
    )
    recommender.add_argument("--bsarec-alpha", type=float, default=0.7)
    recommender.add_argument("--frequency-cutoff", type=int)
    recommender.add_argument("--diff-id-path-weight", type=float, default=0.5)
    recommender.add_argument(
        "--item-side-info",
        help=(
            "DIFF-only JSONL with item_id/asin and a non-empty integer values "
            "array; zero denotes missing/padding"
        ),
    )
    recommender.add_argument("--epochs", type=int, default=100)
    recommender.add_argument("--batch-size", type=int, default=128)
    recommender.add_argument("--learning-rate", type=float)
    recommender.add_argument("--weight-decay", type=float, default=0.0)
    recommender.add_argument("--mask-probability", type=float, default=0.2)
    recommender.add_argument("--negative-samples", type=int, default=0)
    recommender.add_argument("--gradient-clip-norm", type=float)
    recommender.add_argument("--patience", type=int, default=10)
    recommender.add_argument("--min-delta", type=float, default=0.0)
    recommender.add_argument("--catalogue-chunk-size", type=int, default=4096)
    recommender.add_argument("--num-workers", type=int, default=0)
    recommender.add_argument("--seed", type=int, default=42)
    recommender.add_argument("--device", default="cpu")
    recommender.add_argument("--deterministic-algorithms", action="store_true")
    recommender.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help="select hyperparameters on validation without evaluating the test split",
    )
    recommender.add_argument("--iterative-k-core", action="store_true")
    recommender.add_argument("--min-user-interactions", type=int, default=5)
    recommender.add_argument("--min-item-interactions", type=int, default=5)
    _add_json_output(recommender)
    recommender.set_defaults(handler=_handle_train_recommender)
    
    
    recommender_evaluation = subparsers.add_parser(
        "evaluate-recommender",
        help="full-ranking evaluation of a selected recommender checkpoint",
    )
    recommender_evaluation.add_argument("--input", required=True)
    recommender_evaluation.add_argument("--checkpoint", required=True)
    recommender_evaluation.add_argument("--bank")
    recommender_evaluation.add_argument(
        "--partition",
        choices=("validation", "test"),
        default="test",
    )
    recommender_evaluation.add_argument(
        "--catalogue-chunk-size",
        type=int,
        default=4096,
    )
    recommender_evaluation.add_argument("--device", default="cpu")
    _add_json_output(recommender_evaluation)
    recommender_evaluation.set_defaults(handler=_handle_evaluate_recommender)
    
    args = parser.parse_args()
    result = args.handler(args)
    if result is not None:
        payload = json.dumps(result, indent=2, sort_keys=True, default=str)
        target = getattr(args, "json_output", None)
        if target:
            Path(target).write_text(payload + "\n", encoding="utf-8")
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
