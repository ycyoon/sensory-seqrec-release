"""Relation-aware sensory student: learn user-history -> candidate compatibility.

Targets the objective mismatch. Instead of extraction (item text -> descriptor)
or a hand-designed cosine re-rank, this learns, directly from interactions, a
metric over the frozen sensory bank: a per-facet projection and facet weights
trained with in-batch-negative next-item ranking. The BSARec backbone is NOT
touched (that over-amplified before); the learned sensory score is used only as
a trust-region re-rank residual, with its weight selected on validation.

  user_proto_f = normalize(mean_t quality * P_f(z_{hist_t, f}))
  cand_f       = normalize(P_f(z_{i, f}))
  s(u,i)       = sum_f softplus(w_f) * <user_proto_f, cand_f>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _facet_bank(bank_path, source, device):
    from aser_rec.cli import _reordered_bank
    art = torch.load(bank_path, map_location="cpu", weights_only=False)
    bank, _ = _reordered_bank(art, source.mappings.item_to_id)
    canonical = bank["canonical"].float()      # [N, F, D]
    confidence = bank["confidence"].float()
    coverage = bank["coverage"].float()
    quality = torch.where(coverage > 0, confidence.clamp(0, 1) * torch.log1p(coverage),
                          torch.zeros_like(confidence))
    F_, D = canonical.shape[1], canonical.shape[2]
    z = torch.cat([torch.zeros(1, F_, D), canonical]).to(device)          # [N+1, F, D]
    q = torch.cat([torch.zeros(1, F_), quality]).to(device)              # [N+1, F]
    return z, q, F_, D


class RelationScorer(nn.Module):
    def __init__(self, num_facets, dim, proj_dim=64, aggregation="mean", asymmetric=False):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, proj_dim, bias=False) for _ in range(num_facets)])
        # A shared projection makes the score z_h^T (P^T P) z_c - a symmetric PSD
        # metric that can only express "alike". Reading the failure cases showed the
        # model needs "what follows what" (brush -> hair oil, lipstick -> skincare),
        # so give the candidate side its own map: z_h^T (A^T B) z_c is unconstrained.
        self.asymmetric = asymmetric
        self.proj_c = (nn.ModuleList([nn.Linear(dim, proj_dim, bias=False) for _ in range(num_facets)])
                       if asymmetric else None)
        self.facet_w = nn.Parameter(torch.zeros(num_facets))
        self.F = num_facets
        self.aggregation = aggregation
        if aggregation == "attention":
            # per-facet learned attention query, plus learned quality/recency biases
            self.attn_query = nn.Parameter(torch.randn(num_facets, proj_dim) * 0.02)
            self.quality_scale = nn.Parameter(torch.ones(num_facets))
            self.recency_scale = nn.Parameter(torch.zeros(num_facets))

    def proto(self, z, q, hist_ids):
        # hist_ids [B, L] (0 = pad). z [N+1,F,D], q [N+1,F]
        hz = z[hist_ids]                 # [B, L, F, D]
        valid = (hist_ids > 0)           # [B, L]
        hq = q[hist_ids]                 # [B, L, F]
        L = hist_ids.shape[1]
        # recency in [0,1]: newest position = 1 (padding is left, so rightmost is newest)
        recency = torch.arange(L, device=hist_ids.device).float() / max(L - 1, 1)
        protos = []
        for f in range(self.F):
            pf = self.proj[f](hz[:, :, f, :])            # [B, L, d]
            if self.aggregation == "attention":
                logit = (pf * self.attn_query[f]).sum(-1)         # [B, L] content attention
                logit = logit + self.quality_scale[f] * torch.log(hq[:, :, f].clamp_min(1e-6))
                logit = logit + self.recency_scale[f] * recency.unsqueeze(0)
                logit = logit.masked_fill(~valid, float("-inf"))
                alpha = torch.softmax(logit, dim=1).unsqueeze(-1)  # [B, L, 1]
                mu = (alpha * pf).sum(1)
            else:
                w = (hq[:, :, f] * valid).unsqueeze(-1)           # [B, L, 1]
                mu = (w * pf).sum(1) / w.sum(1).clamp_min(1e-9)
            protos.append(F.normalize(mu, dim=-1))
        return torch.stack(protos, dim=1)                # [B, F, d]

    def item_proj(self, z, item_ids):
        cz = z[item_ids]                                  # [..., F, D]
        proj = self.proj_c if self.asymmetric else self.proj
        outs = [F.normalize(proj[f](cz[..., f, :]), dim=-1) for f in range(self.F)]
        return torch.stack(outs, dim=-2)                  # [..., F, d]

    def score_matrix(self, proto, cand):
        # proto [B, F, d], cand [B, F, d] -> [B, B] all pairs
        w = F.softplus(self.facet_w)                      # [F]
        # sim[i,j,f] = <proto_i,f , cand_j,f>
        sim = torch.einsum("ifd,jfd->ijf", proto, cand)
        return (sim * w).sum(-1)

    def score_catalog(self, proto, cand_all):
        # proto [B, F, d], cand_all [N, F, d] -> [B, N]
        w = F.softplus(self.facet_w)
        return torch.einsum("bfd,nfd->bnf", proto, cand_all).mul(w).sum(-1)


def build_prefixes(splits, max_len):
    hist, tgt = [], []
    for s in splits.values():
        seq = list(s.train)
        for t in range(1, len(seq)):
            pref = seq[max(0, t - max_len):t]
            hist.append(pref)
            tgt.append(seq[t])
    return hist, tgt


def pad(batch, max_len, device):
    out = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, h in enumerate(batch):
        out[i, max_len - len(h):] = torch.tensor(h[-max_len:])
    return out.to(device)


def evaluate(model, z, q, cases, base_scores, device, max_len, lambdas, topns,
             log_pop=None, gate_as=(0.0,)):
    """Re-rank cached base scores with the learned sensory score; grid metrics."""
    model.eval()
    with torch.no_grad():
        cand_all = model.item_proj(z, torch.arange(1, z.shape[0], device=device))  # [N, F, d]
    # Popularity gate: on the DIFF study sensory paid off on the coldest quartile
    # (+19% HR@10) and cost accuracy on the hottest (-4.4%), so damp the residual
    # where the ID signal is already strong. a=0 reproduces the ungated model.
    # log_pop is indexed by item id (row 0 is the padding item); the residual is
    # indexed by item_id - 1, so drop the pad row before gating.
    if log_pop is not None and log_pop.shape[0] == z.shape[0]:
        log_pop = log_pop[1:]
    gates = {}
    for a in gate_as:
        if a == 0.0 or log_pop is None:
            gates[a] = None
        else:
            c = float(log_pop[log_pop > 0].median())
            gates[a] = torch.sigmoid(a * (c - log_pop))
    results = {(lam, n, a): {"hr": 0, "ndcg": 0.0}
               for lam in lambdas for n in topns for a in gate_as}
    base_hr = base_ndcg = 0
    cnt = 0
    with torch.no_grad():
        B = 256
        for st in range(0, len(cases), B):
            chunk = cases[st:st + B]
            hist = pad([[i for i in c.query.item_ids if i > 0] for c in chunk], max_len, device)
            proto = model.proto(z, q, hist)                     # [b, F, d]
            sens = model.score_catalog(proto, cand_all)         # [b, N]
            for bi, c in enumerate(chunk):
                s, tgt0, seen = base_scores[st + bi]
                fin = torch.isfinite(s)
                zb = torch.where(fin, (s - s[fin].mean()) / (s[fin].std() + 1e-8), s)
                sv = sens[bi].clone()
                sv[torch.tensor(seen, device=device) - 1] = 0.0
                sv = (sv - sv.mean()) / (sv.std() + 1e-8)
                order = torch.argsort(s, descending=True)
                br = int((s > s[tgt0]).sum()) + 1
                base_hr += int(br <= 10)
                base_ndcg += (1 / math.log2(br + 1)) if br <= 10 else 0.0
                for n in topns:
                    tm = torch.zeros_like(s, dtype=torch.bool)
                    tm[order[:n]] = True
                    for a in gate_as:
                        g = gates[a]
                        svg = sv if g is None else sv * g
                        for lam in lambdas:
                            final = zb + lam * svg * tm.float()
                            rr = int((final > final[tgt0]).sum()) + 1
                            results[(lam, n, a)]["hr"] += int(rr <= 10)
                            results[(lam, n, a)]["ndcg"] += (1 / math.log2(rr + 1)) if rr <= 10 else 0.0
                cnt += 1
    out = {"base": {"HR@10": base_hr / cnt, "NDCG@10": base_ndcg / cnt}, "grid": {}}
    for (lam, n, a), m in results.items():
        out["grid"][f"lam{lam}_n{n}_a{a}"] = {"lambda": lam, "topN": n, "pop_gate_a": a,
                                              "HR@10": m["hr"] / cnt, "NDCG@10": m["ndcg"] / cnt}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reviews", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--aggregation", choices=("mean", "attention"), default="mean")
    p.add_argument("--asymmetric", action="store_true",
                   help="separate history/candidate projections so transitions, not just similarity, can be learned")
    p.add_argument("--pop-gate-a", default="0",
                   help="comma-separated popularity-gate strengths to sweep (0 = off)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    from aser_rec.data import leave_one_out_split, prepare_amazon_dataset
    from aser_rec.recommenders import BSARec
    from aser_rec.rec_training import build_ranking_cases

    source = prepare_amazon_dataset(args.reviews)
    device = torch.device(args.device)
    z, q, Fc, D = _facet_bank(args.bank, source, device)
    splits = leave_one_out_split(source.sequences)

    # frozen base model -> per-case full-catalog score vectors (cached once per partition)
    # model-agnostic: pick the arch from ckpt["model_name"] so DIFF/SASRec/etc. work too.
    from aser_rec.recommenders import BERT4Rec, DIFF, SASRec
    _ARCH = {"sasrec": SASRec, "bert4rec": BERT4Rec, "bsarec": BSARec, "diff": DIFF}
    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    base = _ARCH[str(ckpt.get("model_name", "bsarec"))](source.num_items, **dict(ckpt["model_kwargs"]))
    base.load_state_dict(ckpt["model"])
    base.to(device).eval()
    max_seq = int(ckpt["training_config"]["max_seq_len"])

    def base_scores_fn(cases):
        cache = {}
        with torch.no_grad():
            B = 128
            for st in range(0, len(cases), B):
                chunk = cases[st:st + B]
                ids = torch.tensor([list(c.query.item_ids) for c in chunk], device=device)
                sc = base.predict_next(ids)
                for bi, c in enumerate(chunk):
                    s = sc[bi].clone()
                    seen = torch.tensor(list(c.seen_items), device=device) - 1
                    s[seen] = float("-inf")
                    cache[st + bi] = (s, c.target_item - 1, list(c.seen_items))
        return cache

    # cache base scores once per partition
    val_cases = build_ranking_cases(source, partition="validation", objective="causal", max_seq_len=max_seq)
    test_cases = build_ranking_cases(source, partition="test", objective="causal", max_seq_len=max_seq)
    val_base = base_scores_fn(val_cases)
    test_base = base_scores_fn(test_cases)

    # training data
    hist, tgt = build_prefixes(splits, args.max_len)
    # Train-split popularity only (each user's validation/test items are excluded
    # by construction), so the gate carries no evaluation leakage.
    pop = torch.zeros(z.shape[0], device=device)
    for _s in splits.values():
        for it in _s.train:
            pop[it] += 1
    log_pop = torch.log1p(pop)
    gate_as = tuple(float(a) for a in args.pop_gate_a.split(",")) if args.pop_gate_a else (0.0,)
    tgt = torch.tensor(tgt, device=device)
    n = len(hist)
    model = RelationScorer(Fc, D, args.proj_dim, args.aggregation,
                           asymmetric=args.asymmetric).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    lambdas = [0.05, 0.1, 0.25, 0.5]
    topns = [50, 100]
    best_val = -1.0
    best_state = None
    best_pick = None
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for st in range(0, n, args.batch_size):
            idx = perm[st:st + args.batch_size]
            hb = pad([hist[i] for i in idx], args.max_len, device)
            tb = tgt[torch.tensor(idx, device=device)]
            proto = model.proto(z, q, hb)                 # [B,F,d]
            cand = model.item_proj(z, tb)                 # [B,F,d]
            logits = model.score_matrix(proto, cand)      # [B,B]
            loss = F.cross_entropy(logits, torch.arange(len(idx), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        val = evaluate(model, z, q, val_cases, val_base, device, args.max_len, lambdas, topns,
                       log_pop=log_pop, gate_as=gate_as)
        pick = max(val["grid"].values(), key=lambda g: g["NDCG@10"])
        vd = pick["NDCG@10"] - val["base"]["NDCG@10"]
        print(f"epoch {epoch}: train_loss={tot/n:.4f} val_base_NDCG@10={val['base']['NDCG@10']:.6f} "
              f"best(lam={pick['lambda']},N={pick['topN']}) dNDCG@10={vd:+.6f}")
        if pick["NDCG@10"] > best_val:
            best_val = pick["NDCG@10"]
            best_pick = pick
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test = evaluate(model, z, q, test_cases, test_base, device, args.max_len,
                    [best_pick["lambda"]], [best_pick["topN"]],
                    log_pop=log_pop, gate_as=(best_pick.get("pop_gate_a", 0.0),))
    tp = list(test["grid"].values())[0]
    report = {
        "command": "train-relation-student",
        "val_best": best_pick, "val_base": None,
        "pop_gate_a": best_pick.get("pop_gate_a", 0.0),
        "asymmetric": bool(args.asymmetric),
        "test_base": test["base"], "test_pick": tp,
        "test_delta": {"HR@10": tp["HR@10"] - test["base"]["HR@10"],
                       "NDCG@10": tp["NDCG@10"] - test["base"]["NDCG@10"]},
        "learned_facet_weights": F.softplus(model.facet_w).detach().cpu().tolist(),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("TEST base  HR@10=%.6f NDCG@10=%.6f" % (test["base"]["HR@10"], test["base"]["NDCG@10"]))
    print("TEST learned relation  HR@10=%.6f (%+.6f) NDCG@10=%.6f (%+.6f)" % (
        tp["HR@10"], report["test_delta"]["HR@10"], tp["NDCG@10"], report["test_delta"]["NDCG@10"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
