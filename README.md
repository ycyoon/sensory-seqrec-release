# ASER — recommender-side release

Official code release for

> **Sensory-Aware Sequential Recommendation via Review-Distilled Representations**
> Yeo-Chan Yoon\*, Chanjun Park\*, Kyuhan Koh
> *Knowledge-Based Systems*, accepted 2026. (\* equal contribution)

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

- `models/` — the distilled sensory student encoder that produces the banks,
  so banks can be built for items outside this release. Distributed separately;
  see `models/README.md`.

## What is not included, and why it does not matter here

The seed annotations and the teacher model were produced under a third-party
agreement and are not released. They are not required to reproduce any reported
recommendation result: the banks fully determine the sensory input to the
recommender, and the released student reproduces the banks. The seed prompt and
the extraction schema are published in the paper's appendix, so the seed stage
can be re-run with any instruction-following model.

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

## Archived artifacts

The facet banks and the student encoder are archived at
<https://doi.org/10.5281/zenodo.22008657> (CC-BY-4.0). Download them into
`banks/` and `models/` as described in the README of each directory, then run
`python scripts/verify_banks.py` to check every file against the SHA-256
checksums in `configs/manifest.json`.

## Authors

- Yeo-Chan Yoon (Jeju National University, corresponding author,
  <ycyoon@jejunu.ac.kr>)
- Chanjun Park (Soongsil University)
- Kyuhan Koh (California State University Stanislaus)

Yeo-Chan Yoon and Chanjun Park contributed equally to this research.

This repository is the permanent, non-anonymized home of the code that was
reviewed under an anonymous mirror; the content is unchanged.

## Citation

```bibtex
@article{yoon2026sensory,
  title   = {Sensory-Aware Sequential Recommendation via Review-Distilled Representations},
  author  = {Yoon, Yeo-Chan and Park, Chanjun and Koh, Kyuhan},
  journal = {Knowledge-Based Systems},
  year    = {2026},
  note    = {Accepted}
}
```
