"""
tsnr.py - temporal consistency of Muon's step directions from TSNR_DIAG output (<log>.tsnr/anchor<t>.pt) of runs
logs/<run>-<jobid>.out.

At each anchor step t0 the run fixes the singular pairs (u_i, v_i) of Muon's input (SOAP form) and their within-step BBP
SNR, then follows the weights over the next k steps along each pair: d_i(t) = u_i^T dW_t v_i (dW includes the EMA-Nesterov
jumps), net_i = sum_t d_i(t), path_i = sum_t |d_i(t)|, and the gradient g_i(t) = u_i^T G_t v_i. Pairs are binned by their
within-step SNR (--by snr) or by the rank of their singular value (--by rank, top = largest); per bin:
  share   fraction of the pairs (= of Muon's step at t0, every pair has singular value 1)
  eff     mean per-pair path efficiency |net_i| / path_i (a random walk: ~1/sqrt(k); steady drift: 1)
  fwd     fraction of pairs whose net motion continues the anchor's step (net_i < 0, Muon moves W by -lr u_i v_i^T)
  path%   the bin's share of the total path (motion spent)
  net%    the bin's share of sum_i |net_i| (motion that persisted)
  meanT   mean t-statistic of g_i over the k steps (< 0: later gradients oppose the anchor's step, i.e. overshoot; the
          weights keep moving along the pairs, so this mixes signal with the optimizer's own feedback)
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
SNR_BINS = [("<0.01", -1, 0.01), ("0.01-0.1", 0.01, 0.1), ("0.1-0.3", 0.1, 0.3), ("0.3-1", 0.3, 1), (">1", 1, np.inf)]
RANK_BINS = [("top 1%", 0, 0.01), ("1-10%", 0.01, 0.1), ("10-50%", 0.1, 0.5), ("50-100%", 0.5, 1.01)]
PHASES = [(0, 700), (700, 1400), (1400, 2000), (2000, 2701)]


def tsnr_dir(run: str) -> str:
    outs = glob.glob(os.path.join(ROOT, "logs", f"{run}-*.out"))
    if not outs:
        raise SystemExit(f"no log for {run}")
    txt = open(max(outs, key=os.path.getmtime), errors="replace").read()
    return os.path.join(ROOT, re.search(r"(logs/[0-9a-f-]+)\.txt", txt)[1] + ".tsnr")


def group(name: str) -> str:
    return "attn" if ".attn." in name else "mlp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--by", default="snr", choices=["snr", "rank"])
    a = ap.parse_args()
    BINS = SNR_BINS if a.by == "snr" else RANK_BINS
    # (group, phase, bin) -> lists of per-anchor-matrix stats
    acc = {}
    for run in a.runs:
        for f in glob.glob(os.path.join(tsnr_dir(run), "anchor*.pt")):
            d = torch.load(f, weights_only=False)
            ph = next((i for i, (lo, hi) in enumerate(PHASES) if lo <= d["t0"] < hi), None)
            for r in d["recs"]:
                if a.k not in r["snaps"]:
                    continue
                g, g2, net, path = (x.double().numpy() for x in r["snaps"][a.k])
                k = a.k
                T = (g / k) / (np.sqrt(np.maximum((g2 - g * g / k) / (k - 1), 1e-300)) / np.sqrt(k))
                eff = np.abs(net) / np.maximum(path, 1e-30)
                key = r["snr"].double().numpy() if a.by == "snr" else np.arange(len(g)) / len(g)
                P, N = path.sum(), np.abs(net).sum()
                for b, (_, lo, hi) in enumerate(BINS):
                    m = (key > lo) & (key <= hi) if a.by == "snr" else (key >= lo) & (key < hi)
                    acc.setdefault((group(r["name"]), ph, b), []).append(
                        (m.mean(), eff[m].sum(), (net[m] < 0).sum(), path[m].sum() / P, np.abs(net[m]).sum() / N,
                         T[m].sum(), m.sum()))
    print(f"k = {a.k} steps after each anchor, runs = {len(a.runs)}, bins by {a.by}; random-walk eff ~ {a.k ** -0.5:.3f}")
    for grp in ("attn", "mlp"):
        for ph, (lo, hi) in enumerate(PHASES):
            if not any((grp, ph, b) in acc for b in range(len(BINS))):
                continue
            print(f"\n{grp}  anchors in [{lo}, {hi})")
            print(f"  {a.by + ' bin':>10} {'share':>7} {'eff':>7} {'fwd':>7} {'path%':>7} {'net%':>7} {'meanT':>7}")
            for b, (lab, _, _) in enumerate(BINS):
                v = acc.get((grp, ph, b))
                if not v:
                    continue
                v = np.array(v)
                n = max(v[:, 6].sum(), 1)
                print(f"  {lab:>10} {v[:, 0].mean():7.3f} {v[:, 1].sum() / n:7.3f} {v[:, 2].sum() / n:7.3f} "
                      f"{v[:, 3].mean():7.3f} {v[:, 4].mean():7.3f} {v[:, 5].sum() / n:7.2f}")


if __name__ == "__main__":
    main()
