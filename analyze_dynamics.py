"""
analyze_dynamics.py - bounce statistics from the local optimizer-dynamics logs (<uuid>.dyn.jsonl).

Per group (block type, or each (layer, type)), from the 2-step window a=t-1, b=t that
OptimDynamicsLogger measures every DYN_EVERY steps (v = dW_a, the update applied at step a):

  grad_cos_prev  cos(g_a, g_b), consecutive gradients on different batches.
                 ~0 (or slightly > 0) = minibatch noise; clearly < 0 = the gradient reverses
                 each step ("bounce"): a step-size / curvature problem, not noise.
  align_b        -D1 / grad_norm = cos(v, -g_b): this step's update vs the next batch's
                 descent direction. < 0 overshot along its own direction, > 0 stopped short.
                 g_b is from an independent batch, so the sign has no self-batch bias.
  rho            (g_b.v) / (g_a.v), from g_b.v = D1*|v| and g_a.v = g_b.v - dir_sharpness*|v|^2.
                 In a 1-D quadratic along v, (one-step best step) / (actual step) = 1 / (1 - rho):
                 rho < 0 <=> ratio < 1 <=> overshoot. This favours shorter steps than are best for
                 a whole run, so compare it across groups and runs, not as an absolute.
  step_ratio     1 / (1 - rho) of the window median of rho.
  upd_rel        |dW| / |W|, how far each step moves.

The table value is the median (or mean, --agg mean) over log steps in --steps (default 600-1900,
mid-training: after the warmup / EMA-Nesterov prefill, before the LR cooldown, where the tiny update
makes dir_sharpness noise-dominated).

Runs are named by any of: the Slurm log name or a prefix of it (rec-46, aux-control), the Slurm job id,
the log uuid, or a path to a .dyn.jsonl / .out file. `--list` shows what is available.

Examples
  python analyze_dynamics.py rec-09 rec-29 rec-44 rec-46                    # by block type
  python analyze_dynamics.py rec-46 --by layer --types norm attn_v attn_o   # where is the bounce?
  python analyze_dynamics.py rec-46 aux-control aux-gain --metrics rho align_b --series --types norm
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = ["logs", "logs_v18", "experiment-logs"]
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
METRICS = ["grad_cos_prev", "align_b", "rho", "step_ratio", "upd_rel"]
TYPE_ORDER = ["attn_q", "attn_k", "attn_v", "attn_o", "mlp_up", "mlp_down", "norm", "bias", "embed", "lm_head"]


# -- run discovery ----------------------------------------------------------------------
def _dyn_by_uuid() -> dict[str, str]:
    out = {}
    for d in SEARCH_DIRS:
        for p in glob.glob(os.path.join(ROOT, d, "**", "*.dyn.jsonl"), recursive=True):
            out[os.path.basename(p)[: -len(".dyn.jsonl")]] = p
    return out


def discover() -> list[tuple[str, str]]:
    """[(run name, dyn path)] for every Slurm .out log whose run wrote a dynamics log."""
    dyn = _dyn_by_uuid()
    runs = []
    for out in sorted(glob.glob(os.path.join(ROOT, "logs", "*.out"))):
        with open(out, errors="replace") as f:
            m = UUID.search(f.read())
        if m and m.group(0) in dyn:
            runs.append((os.path.basename(out)[:-4], dyn[m.group(0)]))
    return runs


def resolve(sel: str, runs: list[tuple[str, str]]) -> tuple[str, str]:
    if sel.endswith(".dyn.jsonl") and os.path.exists(sel):
        name = next((n for n, p in runs if os.path.samefile(p, sel)), os.path.basename(sel))
        return name, sel
    if sel.endswith(".out") and os.path.exists(sel):
        sel = os.path.basename(sel)[:-4]
    exact = [r for r in runs if r[0] == sel]
    if exact:
        return exact[0]
    hits = [r for r in runs
            if r[0].startswith(sel + "-") or r[0].endswith("-" + sel) or sel in os.path.basename(r[1])]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sys.exit(f"no dynamics log for run '{sel}' (see --list)")
    sys.exit(f"'{sel}' is ambiguous: {', '.join(n for n, _ in hits)}")


# -- metrics ------------------------------------------------------------------------------
def derive(rec: dict, prefix: str) -> dict | None:
    """The bounce metrics of one group at one log step; None if the group is missing."""
    g = lambda k: rec.get(f"{prefix}/{k}")
    d1, d2, vn, gn, gc, ur = g("D1"), g("dir_sharpness"), g("upd_norm"), g("grad_norm"), g("grad_cos_prev"), g("upd_rel")
    if d1 is None or d2 is None or not vn or not gn:
        return None
    gb_v = d1 * vn
    ga_v = gb_v - d2 * vn * vn
    return {
        "grad_cos_prev": gc,
        "align_b": -d1 / gn,
        "rho": gb_v / ga_v if ga_v else None,
        "upd_rel": ur,
    }


def groups_of(rec: dict, by: str) -> list[tuple[str, str]]:
    """[(group label, key prefix)] present in a record."""
    if by == "global":
        return [(("global",), "global")]
    pat = re.compile(r"^type/([^/]+)/D1$" if by == "type" else r"^layer/(\d+)/([^/]+)/D1$")
    out = []
    for k in rec:
        m = pat.match(k)
        if m:
            out.append(((m.group(1),) if by == "type" else (int(m.group(1)), m.group(2)), k[:-3]))
    return out


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(recs: list[dict], by: str, lo: int, hi: int, agg: str) -> tuple[dict, dict]:
    """-> ({group: {metric: value}}, {group: {metric: [(step, value)]}})"""
    series: dict = {}
    for rec in recs:
        if not lo <= rec["step"] <= hi:
            continue
        for label, prefix in groups_of(rec, by):
            vals = derive(rec, prefix)
            if vals is None:
                continue
            s = series.setdefault(label, {m: [] for m in vals})
            for m, v in vals.items():
                if v is not None and math.isfinite(v):
                    s[m].append((rec["step"], v))
    f = statistics.median if agg == "median" else statistics.fmean
    table = {}
    for label, s in series.items():
        row = {m: (f([v for _, v in pts]) if pts else None) for m, pts in s.items()}
        r = row.get("rho")
        row["step_ratio"] = 1.0 / (1.0 - r) if r is not None and r != 1 else None
        table[label] = row
    return table, series


# -- output -------------------------------------------------------------------------------
def fmt(v, metric):
    if v is None:
        return "-"
    return f"{v:.4f}" if metric == "upd_rel" else f"{v:+.3f}" if metric != "step_ratio" else f"{v:.2f}"


def print_table(title: str, rows: list[str], cols: list[str], cell, width=9):
    first = max([len(title)] + [len(r) for r in rows]) + 2
    print(title.ljust(first) + "".join(c[:width].rjust(width + 1) for c in cols))
    for r in rows:
        print(r.ljust(first) + "".join(cell(r, c).rjust(width + 1) for c in cols))
    print()


def type_key(t: str):
    return (TYPE_ORDER.index(t) if t in TYPE_ORDER else len(TYPE_ORDER), t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run selectors (name / prefix, Slurm job id, uuid, or path)")
    ap.add_argument("--list", action="store_true", help="list the runs that have a dynamics log")
    ap.add_argument("--by", choices=["type", "layer", "global"], default="type",
                    help="group by block type (default), by each (layer, type), or the whole model")
    ap.add_argument("--types", nargs="+", help="only these block types (e.g. norm bias attn_v attn_o)")
    ap.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS)
    ap.add_argument("--steps", nargs=2, type=int, default=[600, 1900], metavar=("LO", "HI"),
                    help="log-step window, inclusive (default 600 1900)")
    ap.add_argument("--agg", choices=["median", "mean"], default="median")
    ap.add_argument("--series", action="store_true", help="also print the per-step values")
    ap.add_argument("--csv", help="write every (run, group, metric, value) row to this CSV")
    args = ap.parse_args()

    runs = discover()
    if args.list or not args.runs:
        for name, path in runs:
            print(f"{name:45s} {os.path.relpath(path, ROOT)}")
        return

    lo, hi = args.steps
    results = []  # (run, table, series)
    for sel in args.runs:
        name, path = resolve(sel, runs)
        table, series = summarize(load(path), args.by, lo, hi, args.agg)
        if args.types:
            keep = lambda g: g[-1] in args.types
            table = {g: v for g, v in table.items() if keep(g)}
            series = {g: v for g, v in series.items() if keep(g)}
        results.append((name, table, series))
    names = [n for n, _, _ in results]
    print(f"window: steps {lo}-{hi}, {args.agg} over log steps; "
          "rho < 0 / align_b < 0 / step_ratio < 1 = overshoot\n")

    if args.by in ("type", "global"):
        groups = sorted({g for _, t, _ in results for g in t}, key=lambda g: type_key(g[-1]))
        labels = {g: g[-1] for g in groups}
        for m in args.metrics:
            if len(results) == 1:  # one run: groups as rows, metrics as columns, once
                break
            print_table(m, [labels[g] for g in groups], names,
                        lambda r, c, m=m: fmt(dict(zip(names, [t for _, t, _ in results]))[c]
                                                .get(next(g for g in groups if labels[g] == r), {}).get(m), m),
                        width=max(9, min(20, max(len(n) for n in names))))
        if len(results) == 1:
            _, t, _ = results[0]
            print_table(names[0], [labels[g] for g in groups], args.metrics,
                        lambda r, c: fmt(t.get(next(g for g in groups if labels[g] == r), {}).get(c), c), width=13)
    else:
        for name, t, _ in results:
            layers = sorted({g[0] for g in t})
            types = sorted({g[1] for g in t}, key=type_key)
            for m in args.metrics:
                print_table(f"{name}: {m}", [f"L{l:02d}" for l in layers], types,
                            lambda r, c, m=m: fmt(t.get((int(r[1:]), c), {}).get(m), m))
            # concentration: is the bounce in a few layers? (per type, from rho)
            print(f"{name}: rho by layer, per type (min = most bounce)")
            for typ in types:
                vals = sorted((t[(l, typ)]["rho"], l) for l in layers
                              if (l, typ) in t and t[(l, typ)].get("rho") is not None)
                if not vals:
                    continue
                med = statistics.median(v for v, _ in vals)
                worst = ", ".join(f"L{l:02d} {v:+.2f}" for v, l in vals[:3])
                print(f"  {typ:9s} median {med:+.3f}  range [{vals[0][0]:+.2f}, {vals[-1][0]:+.2f}]  worst: {worst}")
            print()

    if args.series:
        for name, _, series in results:
            for g in sorted(series, key=lambda g: (type_key(g[-1]), g)):
                label = "/".join(f"L{x:02d}" if isinstance(x, int) else x for x in g)
                for m in args.metrics:
                    if m == "step_ratio":
                        continue
                    pts = series[g].get(m, [])
                    print(f"{name} {label} {m}: " + " ".join(f"{s}:{fmt(v, m)}" for s, v in pts))
            print()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "layer", "type", "metric", "value"])
            for name, t, _ in results:
                for g, row in t.items():
                    layer, typ = (g if len(g) == 2 else ("", g[0]))
                    for m in args.metrics:
                        w.writerow([name, layer, typ, m, row.get(m)])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
