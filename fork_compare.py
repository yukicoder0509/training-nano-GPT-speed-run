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

Readout lab (--readouts <prefix>): reads the <log>.readout.json of each run <prefix>-s<seed>-k<k> (READOUT_LAB=1)
and ranks every readout by its paired difference from the rec-46 readout (Tail-EMA lambda=0.6), mean and SE over
seeds, also converted to steps with the controls' end-of-run slope.

Full runs (--full <prefix> [--full-controls p1,p2,...]): for runs logs/<prefix>-s<seed>-<jobid>.out, the Track 3
rule (3.28 - mean) * sqrt(n) >= 0.004 at every dense validation step, the earliest step that passes, and the
seed-paired difference from the control runs (the mean over control prefixes that have that seed).

Example
  python fork_compare.py --full cand-rg20 --full-controls ckpt-rec46,aux-control,ctrl-rec46
  python fork_compare.py fork-vgate fork-vohead --parent ckpt-rec46 --control fork-ctrl
  python fork_compare.py --readouts lab --ks 2000
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--readouts", help="readout-lab run-name prefix")
    ap.add_argument("--full", help="full-run name prefix (Track 3 significance per dense step)")
    ap.add_argument("--full-controls", default="", help="comma list of control full-run prefixes")
    ap.add_argument("--max-seed", type=int, default=20)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("variants", nargs="*", help="variant run-name prefixes (e.g. fork-vgate)")
    ap.add_argument("--parent", default="ckpt-rec46", help="parent run-name prefix (also a control replica)")
    ap.add_argument("--control", default="fork-ctrl", help="control-fork run-name prefix")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ks", nargs="+", type=int, default=[1000, 1500, 2000])
    ap.add_argument("--end", type=int, default=2690)
    ap.add_argument("--target", type=float, default=3.28)
    args = ap.parse_args()
    E = args.end
    if args.readouts:
        return readouts(args)
    if args.full:
        return full(args)

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


def full(args):
    def runs(prefix):
        out = {}
        for s in range(args.max_seed):
            vs = [read(p) for p in logs(prefix, s, None)]
            vs = [v for v in vs if args.end in v]
            if vs:
                out[s] = vs[0]
        return out
    cand = runs(args.full)
    ctrls = [runs(p) for p in args.full_controls.split(",") if p]
    n = len(cand)
    print(f"{args.full}: n={n} seeds {sorted(cand)}")
    steps = sorted(t for t in next(iter(cand.values())) if args.end - 50 <= t <= args.end and all(t in v for v in cand.values()))
    print(f"  {'step':>5} {'mean':>9} {'(3.28-mean)*sqrt(n)':>20} pass")
    first = None
    for t in steps:
        m = st.mean(v[t] for v in cand.values())
        z = (args.target - m) * n ** .5
        if z >= 0.004 and first is None:
            first = t
        print(f"  {t:>5} {m:>9.5f} {z:>20.5f} {'yes' if z >= 0.004 else ''}")
    print(f"  earliest passing step: {first}")
    if ctrls:
        diffs = []
        for s, v in sorted(cand.items()):
            cs = [c[s][args.end] for c in ctrls if s in c]
            if cs:
                diffs.append(v[args.end] - st.mean(cs))
                print(f"  seed {s}: {v[args.end]:.5f} vs control {st.mean(cs):.5f} ({len(cs)} runs)  diff {diffs[-1]:+.5f}")
        if len(diffs) > 1:
            print(f"  paired diff @ {args.end}: {st.mean(diffs):+.5f} +- {st.stdev(diffs) / len(diffs) ** .5:.5f} (SE, n={len(diffs)})")


def uuid_of(out_path: str) -> str | None:
    with open(out_path, errors="replace") as f:
        m = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", f.read())
    return m.group(0) if m else None


def readouts(args):
    per = {}  # readout -> {seed: val}
    base_name = "tailema lam=0.6"
    slopes = []
    for s in args.seeds:
        for k in args.ks:
            for out in logs(args.readouts, s, k):
                u = uuid_of(out)
                path = os.path.join(ROOT, "logs", f"{u}.readout.json")
                if not u or not os.path.exists(path):
                    continue
                js = json.load(open(path))
                ho = dict(js.get("heldout", []))
                for name, v in dict(js["results"]).items():
                    per.setdefault(name, {})[s] = (v, ho.get(name))
                v = read(out)
                if args.end - 40 in v and args.end in v:
                    slopes.append((v[args.end - 40] - v[args.end]) / 40)
    if base_name not in per:
        print("no readout results found")
        return
    sl = st.mean(slopes) if slopes else float("nan")
    rows = []
    for name, d in per.items():
        seeds = sorted(set(d) & set(per[base_name]))
        diffs = [d[s][0] - per[base_name][s][0] for s in seeds]
        hdiffs = [d[s][1] - per[base_name][s][1] for s in seeds if d[s][1] is not None and per[base_name][s][1] is not None]
        if len(diffs) < 1:
            continue
        se = st.stdev(diffs) / len(diffs) ** .5 if len(diffs) > 1 else float("nan")
        rows.append((st.mean(diffs), se, st.mean(hdiffs) if hdiffs else None, name, len(diffs)))
    rows.sort(key=lambda r: r[0])
    print(f"readouts vs '{base_name}' at the stop step (negative = better); val slope {sl:.6f}/step\n"
          "  val = the selection set; heldout = last train shard, not used for selection\n")
    print(f"  {'val diff':>10} {'SE':>8} {'steps':>7} {'heldout':>10} {'n':>2}  readout")
    for m, se, h, name, n in rows[:args.top]:
        hs = "-" if h is None else f"{h:+.5f}"
        print(f"  {m:>+10.5f} {se:>8.5f} {-m / sl:>+7.1f} {hs:>10} {n:>2}  {name}")


if __name__ == "__main__":
    main()
