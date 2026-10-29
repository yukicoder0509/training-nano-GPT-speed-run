"""
path_eff.py - summarize PATH_EFF=1 output (<log>.patheff.jsonl, optim_dynamics.PathEfficiency) of runs logs/<run>-<jobid>.out.

Per parameter group and training phase (windows grouped by their end step), the median over windows of
  eff_K     chord / path = ||theta_t - theta_{t-K}|| / sum of the K per-step update norms, for K = 50 and 200
  eff200/eff50
            1 for a straight path, 0.5 for a random walk (eff ~ 1/sqrt(K)); > 1: the chord grows faster than
            the path, i.e. bounce that cancels within 50 steps over a drift that persists past 200
  drift_K   chord / sqrt(sum of squared step norms) = eff / (random-walk eff); 1: random walk, < 1: steps cancel
            (bounce), > 1: steps persist (drift)
  t_eff200  median over the group's tensors of the per-tensor eff_200 (the group value mixes tensors)

Example
  python path_eff.py patheff-s0 patheff-s1 patheff-rec46-s0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.abspath(__file__))
PHASES = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2400), (2400, 2701)]
ORDER = ["attn_q", "attn_k", "attn_v", "attn_o", "mlp_up", "mlp_down", "embed", "lm_head", "lm_head_bias", "norm", "bias"]


def records(run: str) -> list[dict]:
    outs = glob.glob(os.path.join(ROOT, "logs", f"{run}-*.out"))
    if not outs:
        raise SystemExit(f"no log for {run}")
    txt = open(max(outs, key=os.path.getmtime), errors="replace").read()
    m = re.search(r"(logs/[0-9a-f-]+)\.txt", txt)
    return [json.loads(l) for l in open(os.path.join(ROOT, m[1] + ".patheff.jsonl"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    for run in args.runs:
        R = records(run)
        groups = [g for g in ORDER if g in R[0]["groups"]] + sorted(set(R[0]["groups"]) - set(ORDER))
        print(f"\n######## {run}")
        print(f"  {'group':13} {'phase':>10} {'eff50':>7} {'eff200':>7} {'200/50':>7} {'drift50':>8} {'drift200':>9} {'t_eff200':>9}")
        for g in groups:
            for lo, hi in PHASES:
                v = {k: [r["groups"][g] for r in R if r["K"] == k and lo < r["step"] <= hi] for k in (50, 200)}
                if not v[50] or not v[200]:
                    continue
                med = lambda k, f: st.median(x[f] for x in v[k])
                e50, e200 = med(50, "eff"), med(200, "eff")
                print(f"  {g:13} {f'{lo}-{min(hi, 2700)}':>10} {e50:7.3f} {e200:7.3f} {e200 / max(e50, 1e-30):7.2f} "
                      f"{med(50, 'drift'):8.2f} {med(200, 'drift'):9.2f} {med(200, 'tensor_eff_median'):9.3f}")


if __name__ == "__main__":
    main()
