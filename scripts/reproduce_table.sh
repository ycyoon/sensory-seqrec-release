#!/bin/bash
# Reproduce the main results table: for each (domain, backbone),
#   stage 1  train the ID-only base recommender
#   stage 2  train the relational sensory integration on the frozen base and
#            report base vs base+relation on the held-out test split
# Both columns of the table come from stage 2's evaluator, so the comparison is
# under a single ranking implementation.
#
# Data: Amazon-2014 5-core review files (README). Banks ship in banks/.
set -eu
PY=${PYTHON:-python}
DATA=${DATA_DIR:-data/raw/amazon2014}
DEVICE=${DEVICE:-cuda:0}
declare -A REVS=( [beauty]=reviews_Beauty_5.json.gz
                  [grocery]=reviews_Grocery_and_Gourmet_Food_5.json.gz
                  [sports]=reviews_Sports_and_Outdoors_5.json.gz
                  [toys]=reviews_Toys_and_Games_5.json.gz
                  [video_games]=reviews_Video_Games_5.json.gz )
declare -A LR=( [bsarec]=5e-4 [sasrec]=1e-3 [bert4rec]=1e-4 [diff]=1e-4 )
mkdir -p out
for d in beauty grocery sports toys video_games; do
  for m in bsarec sasrec bert4rec diff; do
    base="out/${d}_${m}_base"
    if [ ! -f "$base.pt" ]; then
      epochs=20 patience=3; [ "$m" = bert4rec ] && { epochs=400; patience=20; }
      CUBLAS_WORKSPACE_CONFIG=:4096:8 $PY -m aser_rec.cli train-recommender \
        --input "$DATA/${REVS[$d]}" --model $m --setting base \
        --max-seq-len 50 --hidden-size 64 --num-layers 2 --num-heads 2 \
        --dropout 0.2 --epochs $epochs --batch-size 256 --learning-rate ${LR[$m]} \
        --gradient-clip-norm 5.0 --patience $patience --num-workers 4 \
        --seed 42 --device "$DEVICE" --deterministic-algorithms \
        --skip-test-evaluation \
        --output "$base.pt" --json-output "$base.json"
    fi
    rel="out/${d}_${m}_relation"
    [ -f "$rel.json" ] && continue
    $PY -u -m aser_rec.relation \
      --reviews "$DATA/${REVS[$d]}" \
      --base-checkpoint "$base.pt" \
      --bank "banks/bank-$d.pt" --aggregation attention --seed 42 \
      --output "$rel.json"
  done
done
$PY - <<'PYEOF'
import json, glob
print(f"{'cell':26s} {'base HR@10':>10s} {'+rel HR@10':>10s} {'dHR@10':>8s} {'dNDCG@10':>9s}")
for f in sorted(glob.glob('out/*_relation.json')):
    d=json.load(open(f)); tb,tp,td=d['test_base'],d.get('test_pick'),d['test_delta']
    print(f"{f[4:-14]:26s} {tb['HR@10']:10.5f} {tp['HR@10']:10.5f} {td['HR@10']:+8.5f} {td['NDCG@10']:+9.5f}")
PYEOF
