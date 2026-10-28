"""
fork_compare.py - score fork experiments against same-checkpoint control forks.

A fork resumes a parent run's checkpoint at step k (RESUME_FROM) with a change switched on; its control is the
parent run itself plus any control forks from the same checkpoint (identical state, so their spread is only kernel
non-determinism). Slurm logs are found by name:

    parent:   logs/<parent>-s<seed>-<jobid>.out
    controls: logs/<control>-s<seed>-k<k>[-<tag>]-<jobid>.out
    variant:  logs/<variant>-s<seed>-k<k>[-<tag>]-<jobid>.out

For each variant, checkpoint step k and seed:
  delta       val_ema@END(variant) - mean val_ema@END(controls)
  steps_saved -delta / slope, with slope the controls' val_ema drop per step over the dense validation window
              (DENSE_VAL, END-40..END): how many steps earlier the variant reaches the controls' final loss
              (positive = earlier). Linear in delta, so it is defined for losses too.
  pass        first dense-val step with val_ema <= --target (3.28), variant vs controls
Then the mean and standard error over seeds, and the control-fork noise (pooled SD of the replicas).

Example
  python fork_compare.py fork-vgate fork-vohead --parent ckpt-rec46 --control fork-ctrl
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.abspath(__file__))
VAL = re.compile(r"step:(\d+)/\d+ val_loss:([\d.]+) val_ema_loss:([\d.]+)")


def read(path: str) -> dict[int, float]:
    out = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = VAL.match(line)
            if m:
                out[int(m[1])] = float(m[3])
    return out


def logs(prefix: str, seed: int, k: int | None) -> list[str]:
    pat = f"{prefix}-s{seed}-k{k}" if k is not None else f"{prefix}-s{seed}"
    hits = glob.glob(os.path.join(ROOT, "logs", pat + "-*.out"))
    rx = re.compile(re.escape(pat) + (r"(-[a-z])?-\d+\.out$" if k is not None else r"-\d+\.out$"))
    return sorted(h for h in hits if rx.search(os.path.basename(h)))


def slope(reps: list[dict[int, float]], end: int, span: int = 40) -> float:
    """Mean val_ema drop per step over [end - span, end]."""
    return st.mean((r[end - span] - r[end]) / span for r in reps if end - span in r)


def first_pass(v: dict[int, float], target: float, end: int):
    return next((s for s in sorted(v) if end - 100 <= s <= end and v[s] <= target), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="+", help="variant run-name prefixes (e.g. fork-vgate)")
    ap.add_argument("--parent", default="ckpt-rec46", help="parent run-name prefix (also a control replica)")
    ap.add_argument("--control", default="fork-ctrl", help="control-fork run-name prefix")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ks", nargs="+", type=int, default=[1000, 1500, 2000])
    ap.add_argument("--end", type=int, default=2690)
    ap.add_argument("--target", type=float, default=3.28)
    args = ap.parse_args()
    E = args.end

    # control replicas per (seed, k)
    ctrl = {}
    ss, dof = 0.0, 0
    for s in args.seeds:
        parent = [read(p) for p in logs(args.parent, s, None)]
        for k in args.ks:
            reps = [r for r in parent + [read(p) for p in logs(args.control, s, k)] if E in r]
            if not reps:
                continue
            ctrl[(s, k)] = reps
            if len(reps) > 1:
                m = st.mean(r[E] for r in reps)
                ss += sum((r[E] - m) ** 2 for r in reps)
                dof += len(reps) - 1
    if dof:
        print(f"control fork noise: pooled SD of val_ema@{E} = {(ss / dof) ** .5:.6f} ({dof} dof)\n")

    for var in args.variants:
        for k in args.ks:
            rows = []
            for s in args.seeds:
                vs = [read(p) for p in logs(var, s, k)]
                vs = [v for v in vs if E in v]
                if not vs or (s, k) not in ctrl:
                    continue
                c = ctrl[(s, k)]
                c_end, sl = st.mean(r[E] for r in c), slope(c, E)
                for v in vs:
                    rows.append(dict(seed=s, delta=v[E] - c_end, saved=-(v[E] - c_end) / sl,
                                     vpass=first_pass(v, args.target, E),
                                     cpass=first_pass({t: st.mean(r[t] for r in c if t in r) for t in c[0]},
                                                      args.target, E), n_ctrl=len(c)))
            if not rows:
                continue
            print(f"{var}  fork at k={k}")
            print(f"  {'seed':>4} {'delta':>10} {'steps_saved':>12} {'pass var/ctrl':>14} {'#ctrl':>6}")
            for r in rows:
                sv = "-" if r["saved"] is None else f"{r['saved']:+.1f}"
                print(f"  {r['seed']:>4} {r['delta']:>+10.5f} {sv:>12} {str(r['vpass']) + '/' + str(r['cpass']):>14}"
                      f" {r['n_ctrl']:>6}")
            d = [r["delta"] for r in rows]
            sv = [r["saved"] for r in rows if r["saved"] is not None]
            se = st.stdev(d) / len(d) ** .5 if len(d) > 1 else float("nan")
            line = f"  mean delta {st.mean(d):+.5f} +- {se:.5f} (SE, n={len(d)})"
            if sv:
                line += f"   mean steps_saved {st.mean(sv):+.1f}"
            print(line + "\n")


if __name__ == "__main__":
    main()
