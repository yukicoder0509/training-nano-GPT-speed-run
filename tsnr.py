"""
tsnr.py - temporal consistency of Muon's step directions from TSNR_DIAG output (<log>.tsnr/anchor<t>.pt) of runs
logs/<run>-<jobid>.out.

At each anchor step t0 the run fixes the singular pairs (u_i, v_i) of Muon's input (SOAP form) and their within-step BBP
SNR, then sums g_i(t) = u_i^T G_t v_i over the next k steps (t0 < t <= t0 + k; G_t the summed gradient, independent of
the anchor's noise). Muon's step at t0 moves -lr u_i v_i^T along every pair, so g_i is that pair's first-order descent.
Per pair: the temporal t-statistic T_i = mean_t g_i / (sd_t g_i / sqrt(k)). Pairs are binned by their within-step SNR
(0 = inside the noise bulk); per bin:
  share   fraction of the pairs (= of Muon's step, every pair has singular value 1)
  T>2     fraction descending consistently over the k steps (pure noise: ~2-3%)
  T<-2    fraction ascending consistently
  meanT   mean t-statistic
  desc    share of the total first-order descent sum_i sum_t g_i(t) over all pairs

Example
  python tsnr.py tsnr-s0 tsnr-s1 tsnr-s2 --k 50
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
BINS = [("0 (bulk)", 0, 0), ("(0,1]", 0, 1), ("(1,4]", 1, 4), (">4", 4, np.inf)]
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
    a = ap.parse_args()
    # (group, phase, bin) -> lists of per-anchor-matrix stats
    acc = {}
    for run in a.runs:
        for f in glob.glob(os.path.join(tsnr_dir(run), "anchor*.pt")):
            d = torch.load(f, weights_only=False)
            ph = next((i for i, (lo, hi) in enumerate(PHASES) if lo <= d["t0"] < hi), None)
            for r in d["recs"]:
                if a.k not in r["snaps"]:
                    continue
                g, g2 = (x.double().numpy() for x in r["snaps"][a.k])
                k = a.k
                mean = g / k
                sd = np.sqrt(np.maximum((g2 - g * g / k) / (k - 1), 1e-300))
                T = mean / (sd / np.sqrt(k))
                snr = r["snr"].double().numpy()
                tot = g.sum()
                for b, (_, lo, hi) in enumerate(BINS):
                    m = (snr == 0) if hi == 0 else (snr > lo) & (snr <= hi)
                    acc.setdefault((group(r["name"]), ph, b), []).append(
                        (m.mean(), (T[m] > 2).sum(), (T[m] < -2).sum(), T[m].sum(), m.sum(), g[m].sum() / tot))
    print(f"k = {a.k} steps after each anchor, runs = {len(a.runs)}")
    for grp in ("attn", "mlp"):
        for ph, (lo, hi) in enumerate(PHASES):
            if not any((grp, ph, b) in acc for b in range(len(BINS))):
                continue
            print(f"\n{grp}  anchors in [{lo}, {hi})")
            print(f"  {'SNR bin':>10} {'share':>7} {'T>2':>7} {'T<-2':>7} {'meanT':>7} {'desc':>7}")
            for b, (lab, _, _) in enumerate(BINS):
                v = acc.get((grp, ph, b))
                if not v:
                    continue
                v = np.array(v)
                n = max(v[:, 4].sum(), 1)
                print(f"  {lab:>10} {v[:, 0].mean():7.3f} {v[:, 1].sum() / n:7.3f} {v[:, 2].sum() / n:7.3f} "
                      f"{v[:, 3].sum() / n:7.2f} {v[:, 5].mean():7.3f}")


if __name__ == "__main__":
    main()
