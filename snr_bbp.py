"""
snr_bbp.py - per-rank SNR of Muon's input from SNR_DIAG=N output (<log>.snr/step<k>.pt) of runs logs/<run>-<jobid>.out.

Model (spiked rectangular matrix): Y = sum_i s_i u_i v_i^T + Z, Y m x n, Z iid entries of variance tau2. A spike s is
detectable iff s^2 > tau2 sqrt(mn) (BBP transition); above it the observed singular value is
    sigma^2 = (s^2 + tau2 m)(s^2 + tau2 n) / s^2,
below it sigma sits in the noise bulk, edge tau (sqrt(m) + sqrt(n)). Inverting, B = sigma^2 - tau2 (m + n):
    s^2 = (B + sqrt(B^2 - 4 tau2^2 mn)) / 2      (sigma above the edge; else s = 0)
    SNR_i = s_i^2 / (tau2 sqrt(mn))              (1 at the BBP threshold, 0 inside the bulk)
tau2 = the matrix's entrywise noise variance (global) or the noise variance along (u_i, v_i) (--dir, 7 dof per value).
The fraction of singular values with SNR > --thr is the share of Muon's orthogonalized step (all singular values set
to 1) spent on signal directions.

Example
  python snr_bbp.py snr-s0 snr-s1 snr-s2 --thr 4 --plot /path/snr.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
TYPES = ["q", "k", "v", "o", "fc", "proj"]


def kind(name: str) -> str:
    return {"attn.q": "q", "attn.k": "k", "attn.v": "v", "attn.proj": "o", "mlp.fc": "fc",
            "mlp.proj": "proj"}[name.split(".", 2)[2].rsplit(".", 1)[0]]


def snr_dir(run: str) -> str:
    outs = glob.glob(os.path.join(ROOT, "logs", f"{run}-*.out"))
    if not outs:
        raise SystemExit(f"no log for {run}")
    txt = open(max(outs, key=os.path.getmtime), errors="replace").read()
    return os.path.join(ROOT, re.search(r"(logs/[0-9a-f-]+)\.txt", txt)[1] + ".snr")


def bbp_snr(s: np.ndarray, tau2, m: int, n: int) -> np.ndarray:
    B = s ** 2 - tau2 * (m + n)
    disc = np.maximum(B ** 2 - 4 * tau2 ** 2 * m * n, 0.0)
    a = np.where(s > np.sqrt(tau2) * (np.sqrt(m) + np.sqrt(n)), (B + np.sqrt(disc)) / 2, 0.0)
    return a / (tau2 * np.sqrt(m * n))


def load(run: str, key: str, use_dir: bool, thrs: list[float]):
    """-> {step: {type: (fractions over thrs, mean over layers), ...}}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(snr_dir(run), "step*.pt")), key=lambda f: int(re.search(r"step(\d+)", f)[1])):
        d = torch.load(f, weights_only=False)
        by = {}
        for r in d["recs"]:
            if key not in r:
                continue
            m, n = r["shape"]
            x = r[key]
            tau2 = x["tau2_dir"].double().numpy() if use_dir else x["tau2"]
            snr = bbp_snr(x["s"].double().numpy(), tau2, m, n)
            by.setdefault(kind(r["name"]), []).append([(snr > t).mean() for t in thrs])
        out[d["step"]] = {k: np.mean(v, axis=0) for k, v in by.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--key", default="soap", choices=["momentum", "soap"], help="soap = the input Newton-Schulz sees")
    ap.add_argument("--dir", action="store_true", help="noise variance along each singular pair instead of global")
    ap.add_argument("--thr", default="1,4,16")
    ap.add_argument("--plot", default="")
    a = ap.parse_args()
    thrs = [float(t) for t in a.thr.split(",")]
    per_run = [load(r, a.key, a.dir, thrs) for r in a.runs]
    steps = sorted(set.intersection(*[set(p) for p in per_run]))
    mean = {s: {k: np.mean([p[s][k] for p in per_run], axis=0) for k in TYPES if k in per_run[0][s]} for s in steps}
    types = [k for k in TYPES if k in mean[steps[0]]]
    print(f"key={a.key} tau2={'per-direction' if a.dir else 'global'} runs={len(a.runs)}  fraction of singular values with SNR > thr")
    for j, t in enumerate(thrs):
        print(f"\nSNR > {t:g}")
        print("  step " + " ".join(f"{k:>6}" for k in types))
        for s in steps:
            print(f"{s:6d} " + " ".join(f"{mean[s][k][j]:6.3f}" for k in types))
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(thrs), figsize=(5 * len(thrs), 4), sharey=True)
        for j, (ax, t) in enumerate(zip(np.atleast_1d(axes), thrs)):
            for k in types:
                ax.plot(steps, [mean[s][k][j] for s in steps], label=k)
            ax.set_title(f"SNR > {t:g} ({a.key}, {'dir' if a.dir else 'global'} noise)")
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel("fraction of singular values")
        np.atleast_1d(axes)[0].legend()
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120)
        print(f"\nplot -> {a.plot}")


if __name__ == "__main__":
    main()
