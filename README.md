# ASER — recommender-side release

Reproduces the sensory-aware sequential recommendation results (all four
backbones, five Amazon-2014 domains) from the precomputed sensory facet banks.

## What is included

- `aser_rec/` — recommender training/evaluation code (SASRec, BERT4Rec, BSARec,
  DIFF) plus `aser_rec.relation`, the bounded relational sensory integration
  that produces the reported results. Verbatim from the research codebase.
- `banks/` — precomputed item facet banks per domain (`[n_items, 5, 768]`),
  **the reproducible artifact**: the only object that enters the recommender.
  Distributed separately because of file-size limits; see `banks/README.md`
  for the download location and `scripts/verify_banks.py` for the checksum
  check.
- `scripts/reproduce_table.sh` — regenerates every main-table cell.
- `configs/manifest.json` — SHA-256 of every shipped file.

## What is not included, and why it does not matter here

The annotation pipeline (seed labelling, teacher, student) was produced under a
third-party agreement and is not released. It is not required to reproduce any
reported recommendation result: the banks fully determine the sensory input.

## Evaluation protocol

Standard leave-one-out: per user, the last interaction is the test target, the
second-to-last is validation, the rest train. Full-catalogue ranking, no negative
sampling. HR@K / NDCG@K, K in {5, 10, 20}. Seed 42, deterministic algorithms on.

## Reproduce

```bash
pip install -r requirements.txt
# download the facet banks first (banks/README.md), then:
python scripts/verify_banks.py
bash scripts/reproduce_table.sh   # ~2 days on 2 GPUs (BERT4Rec base training dominates)
# Each cell reports base vs base+relation from a single evaluator.
```

## Anonymity

This repository is released anonymously for peer review. It contains no author,
institution, or funding information, and the artifacts carry no filesystem paths
from the machine that produced them.
