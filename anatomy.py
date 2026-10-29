"""
anatomy.py - take the row-gain effect apart from ANATOMY=1 dumps (train_gpt.py) of a control and variant forks.

Each run is a directory CKPT_DIR = /work/$USER/nanogpt-ckpt/<run>-seed<s> with anatomy/rows.pt (rank 0, every step:
radial gradient gamma, log row norm before/after the optimizer step, turn angle theta, Muon lr), anatomy/muon_rank*.pt
(owning rank, every step: u/w after RowFloor, RowFloor flag, row-gain length step), anatomy/tokloss_rank*.pt
(per-token val loss at the stop step) and the val timeline in logs/<run>-<jobid>.out.

Per matrix type (q, k, v, o, fc, proj; all layers pooled), for every run:
  acf        pooled autocorrelation of gamma (row means removed) at lags 1, 2, 5, 10, 20, 50
  VR(k)      variance ratio Var(sum of k consecutive gamma) / (k Var gamma), row means removed; < 1: the push reverses
             within k steps, > 1: it persists
  |t|        whole-run |sum gamma| / sqrt(sum gamma^2), median over rows (pure noise: 0.67)
  corr(g,d)  corr(gamma_t, d_t), d = log r minus its centred 201-step moving average; > 0: a row longer than its
             trend is pushed to shrink (a restoring force)
  tau_d      lag at which the autocorrelation of d drops below 1/e (steps): how long a length deviation lasts
  corr(g,dr) corr(gamma_t, log r_post - log r_pre) per step: < 0 when the step shortens rows that are pushed to shrink
  theta/lr   per-row turn angle per unit Muon lr (median over rows of the row mean), and its CV over rows
  |dlogr|    mean |log r_post - log r_pre| per step (optimizer) and |log r_pre(t+1) - log r_post(t)| (EMA-Nesterov shift)
  u/w, floor mean Muon u/w after RowFloor and the fraction of rows the RowFloor lifts; |gain| mean |row-gain step|
Over steps windows given by --windows (default 1000:1500,1500:2000,2000:2690).

Function (val steps, 16 val sequences): attention / MLP output RMS relative to the residual stream per layer, fc unit
firing rate, dead units (firing < 0.1 %). Per-token loss (stop step, Tail-EMA readout), variant minus control: mean
difference per target-token frequency quintile, context position bucket and control-loss decile.
Validation timeline: val_ema of each run at every val step, variant minus control.

Example
  python anatomy.py --control anat-C --variants anat-G,anat-G0,anat-Goff2000,anat-Gon2000,anat-Gproj --seeds 0,1
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = f"/work/{os.environ.get('USER', 'user')}/nanogpt-ckpt"
KINDS = ["q", "k", "v", "o", "fc", "proj"]
KIND = {"attn.q": "q", "attn.k": "k", "attn.v": "v", "attn.proj": "o", "mlp.fc": "fc", "mlp.proj": "proj"}
VAL = re.compile(r"step:(\d+)/\d+ val_loss:([\d.]+) val_ema_loss:([\d.]+)")


def kind(name: str) -> str:
    name = name.removeprefix("blocks.")
    return KIND[name.split(".", 1)[1].rsplit(".", 1)[0]]


def run_dir(run: str, seed: int) -> str:
    return f"{CKPT}/{run}-s{seed}-seed{seed}"


def load(run: str, seed: int) -> dict:
    d = os.path.join(run_dir(run, seed), "anatomy")
    rows = torch.load(f"{d}/rows.pt", weights_only=False)
    muon = {}
    for f in sorted(glob.glob(f"{d}/muon_rank*.pt")):
        muon.update(torch.load(f, weights_only=False))
    tok = [torch.load(f, weights_only=False) for f in sorted(glob.glob(f"{d}/tokloss_rank*.pt"))]
    idx, off = {k: [] for k in KINDS}, 0
    for n, r in rows["rows"]:
        idx[kind(n)].append(torch.arange(off, off + r))
        off += r
    rows["idx"] = {k: torch.cat(v) for k, v in idx.items() if v}
    rows["muon"], rows["tok"] = muon, tok
    rows["steps"] = torch.tensor(rows["steps"])
    return rows


def val_timeline(run: str, seed: int) -> dict[int, float]:
    out = {}
    for path in glob.glob(os.path.join(ROOT, "logs", f"{run}-s{seed}-*.out")):
        with open(path, errors="replace") as f:
            for line in f:
                m = VAL.match(line)
                if m:
                    out[int(m[1])] = float(m[3])
    return out


def moving_avg(x: torch.Tensor, w: int) -> torch.Tensor:
    """Centred moving average along dim 0 (edges: shorter windows)."""
    c = torch.cat([torch.zeros_like(x[:1]), x.cumsum(0)])
    T, h = x.size(0), w // 2
    lo = torch.clamp(torch.arange(T) - h, min=0)
    hi = torch.clamp(torch.arange(T) + h + 1, max=T)
    return (c[hi] - c[lo]) / (hi - lo).unsqueeze(1).to(x.dtype)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()


def row_stats(R: dict, k: str, lo: int, hi: int) -> dict:
    sel = (R["steps"] >= lo) & (R["steps"] < hi)
    if sel.sum() < 50:
        return {}
    ix = R["idx"][k]
    g = R["gamma"][sel][:, ix].float()
    T = g.size(0)
    gc = g - g.mean(0)
    var = gc.square().sum()
    out = {f"acf{l}": ((gc[:-l] * gc[l:]).sum() / var).item() for l in (1, 2, 5, 10, 20, 50) if l < T}
    for b in (10, 50, 200):
        nb = T // b
        if nb >= 2:
            s = gc[: nb * b].reshape(nb, b, -1).sum(1)
            out[f"VR{b}"] = (s.square().mean() / (b * gc[: nb * b].square().mean())).item()
    out["|t|"] = (g.sum(0).abs() / g.square().sum(0).sqrt().clamp_min(1e-30)).median().item()
    lr = R["logr_pre"][sel][:, ix].float()
    d = lr - moving_avg(lr, 201)
    out["corr(g,d)"] = corr(g, d)
    dc = d - d.mean(0)
    dv = dc.square().sum()
    tau = next((l for l in range(1, min(T // 2, 400)) if (dc[:-l] * dc[l:]).sum() / dv < math.exp(-1)), float("nan"))
    out["tau_d"] = tau
    lpost = R["logr_post"][sel][:, ix].float()
    dr = lpost - lr
    out["corr(g,dr)"] = corr(g, dr)
    out["|dlogr|opt"] = dr.abs().mean().item()
    out["|dlogr|shift"] = (lr[1:] - lpost[:-1]).abs().mean().item()
    eta = torch.tensor(R["lr"])[sel].float().unsqueeze(1)
    th = R["theta"][sel][:, ix].float() / eta
    m = th.mean(0)
    out["theta/lr"], out["theta/lr cv"] = m.median().item(), (m.std() / m.mean()).item()
    return out


def muon_stats(R: dict, k: str, lo: int, hi: int) -> dict:
    sel = (R["steps"] >= lo) & (R["steps"] < hi)
    xs = [v[sel] for n, v in R["muon"].items() if kind(n) == k]   # (T, 3, rows)
    if not xs or sel.sum() == 0:
        return {}
    x = torch.cat(xs, dim=2).float()
    return {"u/w": x[:, 0].mean().item(), "floor": x[:, 1].mean().item(), "|gain|": x[:, 2].abs().mean().item()}


def func_stats(R: dict) -> dict:
    out = {}
    for step, f in sorted(R["func"].items()):
        out[step] = dict(attn=(f["rms_attn"] / f["rms_x_attn"]).tolist(), mlp=(f["rms_mlp"] / f["rms_x_mlp"]).tolist(),
                         fire=f["fc_fire"].float().mean(1).tolist(), dead=(f["fc_fire"].float() < 1e-3).float().mean(1).tolist())
    return out


def token_diff(C: dict, V: dict) -> dict:
    tgt = torch.cat([t["targets"] for t in C["tok"]])
    a = torch.cat([t["tailema"] for t in C["tok"]]).double()
    b = torch.cat([t["tailema"] for t in V["tok"]]).double()
    assert torch.equal(tgt, torch.cat([t["targets"] for t in V["tok"]]))
    diff, n = b - a, a.numel()
    out = {"total": diff.mean().item()}
    freq = torch.bincount(tgt.flatten().long(), minlength=50304)[tgt.long()]
    q = torch.quantile(freq.flatten().double()[:: max(1, n // 1_000_000)], torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64))
    fb = torch.bucketize(freq.double(), q)
    out["freq quintile (rare->common)"] = [(diff[fb == i].sum() / n).item() for i in range(5)]
    pos = torch.arange(tgt.size(1)).expand_as(tgt)
    edges = [0, 16, 128, 512, tgt.size(1)]
    out["position (0-15,16-127,128-511,512+)"] = [(diff[(pos >= l) & (pos < h)].sum() / n).item()
                                                  for l, h in zip(edges, edges[1:])]
    qa = torch.quantile(a.flatten()[:: max(1, n // 1_000_000)], torch.linspace(0.1, 0.9, 9, dtype=torch.float64))
    lb = torch.bucketize(a, qa)
    out["control-loss decile (easy->hard)"] = [(diff[lb == i].sum() / n).item() for i in range(10)]
    return out


def fmt(v) -> str:
    return f"{v:+.4f}" if isinstance(v, float) and abs(v) < 10 else f"{v:.4g}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--windows", default="1000:1500,1500:2000,2000:2690")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    windows = [tuple(int(x) for x in w.split(":")) for w in args.windows.split(",")]
    runs = [args.control] + args.variants.split(",")
    for seed in seeds:
        print(f"\n######## seed {seed}")
        data = {}
        for r in runs:
            try:
                data[r] = load(r, seed)
            except FileNotFoundError as e:
                print(f"  {r}: missing ({e.filename})")
        tl = {r: val_timeline(f"{r}", seed) for r in data}
        if args.control in tl:
            print("\n== val_ema, variant - control, at every common val step")
            steps = sorted(set.intersection(*[set(v) for v in tl.values() if v]) if tl else [])
            print("  step  " + " ".join(f"{r:>14}" for r in data if r != args.control) + f"   {args.control}")
            for s in steps:
                c = tl[args.control][s]
                print(f"  {s:4d}  " + " ".join(f"{tl[r][s] - c:+14.5f}" for r in data if r != args.control) + f"   {c:.5f}")
        for lo, hi in windows:
            print(f"\n== rows, steps {lo}-{hi}")
            for k in KINDS:
                print(f"  [{k}]")
                for r, R in data.items():
                    st = {**row_stats(R, k, lo, hi), **muon_stats(R, k, lo, hi)}
                    if st:
                        print(f"    {r:>16}: " + "  ".join(f"{a}={fmt(b)}" for a, b in st.items()))
        print("\n== function (last val step): attn/x and mlp/x output RMS per layer, fc firing, dead units")
        for r, R in data.items():
            fs = func_stats(R)
            if fs:
                s, f = max(fs.items())
                print(f"  {r:>16} @ {s}: attn {[round(x, 3) for x in f['attn']]}")
                print(f"  {'':>16}        mlp  {[round(x, 3) for x in f['mlp']]}")
                print(f"  {'':>16}        fire {[round(x, 3) for x in f['fire']]}  dead {[round(x, 4) for x in f['dead']]}")
        if args.control in data and data[args.control]["tok"]:
            print("\n== per-token loss (Tail-EMA), variant - control; buckets are contributions to the mean difference")
            for r, R in data.items():
                if r != args.control and R["tok"]:
                    for key, v in token_diff(data[args.control], R).items():
                        print(f"  {r:>16} {key}: " + (fmt(v) if isinstance(v, float) else " ".join(f"{x:+.5f}" for x in v)))


if __name__ == "__main__":
    main()
