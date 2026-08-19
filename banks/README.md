# Sensory facet banks

The five per-domain facet banks are the reproducible artifact of this work: they
are the only object that enters the recommender, so every reported
recommendation result regenerates from them without the annotation pipeline.

They are distributed separately because each exceeds the 100 MB per-file limit
of a Git repository (916 MB in total). Download them into this directory:

    <ANONYMOUS_ARCHIVE_URL>

Expected layout after download:

    banks/bank-beauty.pt
    banks/bank-grocery.pt
    banks/bank-sports.pt
    banks/bank-toys.pt
    banks/bank-video_games.pt

Then verify integrity against the checksums recorded at release time:

    python scripts/verify_banks.py

`scripts/reproduce_table.sh` will not run until this check passes.
