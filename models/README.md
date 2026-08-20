# Sensory student encoder

`student-discriminative-v1.pt` is the distilled multi-facet student that produces
the facet banks. It is the five-domain encoder used for every result reported in
the paper.

With this checkpoint the pipeline is reusable rather than merely verifiable: a
bank can be built for items or domains that are not part of this release,
directly from item text, without the seed model and without the teacher.

Download it into this directory alongside the banks:

    https://doi.org/10.5281/zenodo.22008657

    models/student-discriminative-v1.pt      (617 MB)

Verify with `python scripts/verify_banks.py`, which checks every artifact in
`configs/manifest.json`.

Building a bank for new items:

    python -m aser_rec.cli build-bank \
        --checkpoint models/student-discriminative-v1.pt \
        --supervision <your-inference-supervision>.jsonl \
        --output banks/bank-<domain>.pt \
        --presence-threshold 0.5 --adapter-dim 128 --embedding-dim 768

Optimizer and scheduler states were removed; the checkpoint carries the model
weights, the training metadata, and the validation history.

**Not included.** The seed annotations, the teacher, and the extraction prompts
were produced under a third-party agreement and are not released. They are not
required to reproduce any result reported in the paper: the banks fully
determine the sensory input to the recommender.
