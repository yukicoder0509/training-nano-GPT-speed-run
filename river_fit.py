"""
river_fit.py - offline forecasting of the "river" from a TRAJ_REC=1 recording (optim_dynamics.TrajRecorder).

From the recorded theta_t, jump_t, g_t of a parameter group: Delta_t = theta_{t+1} - theta_t (whole iteration) and the
inner optimizer's own step u_t = Delta_t - jump_t. At an evaluation step t (every --every steps) each estimator sees
only the past (theta_0..theta_t, Delta/u/g up to t-1, the lr schedule, which is fixed in advance) and forecasts
  target  U_t(H) = sum_{j=t}^{t+H-1} u_j     the inner optimizer's displacement over the next H steps
          (--target total: theta_{t+H} - theta_t, which includes the run's own future jumps)
Scores per group, phase (by t) and H, over the flattened group vector:
  cos     median over t of cos(forecast, U)
  R2*     (sum_t <f,U>)^2 / (sum_t |f|^2 sum_t |U|^2): fraction of the target explained with the best single scale
          c* = sum <f,U> / sum |f|^2 in that phase (reported as c*)
  R2      1 - sum |U - f|^2 / sum |U|^2 with the estimator's own scale (velocity estimators are scaled to H steps)

Estimators (velocity = per-step displacement, forecast = velocity * H unless noted):
  ema<b>        EMA (beta b) of Delta; ema0.99 is the EMA-Nesterov direction
  uema<b>       EMA of the inner step u only
  msm-ademamix  ema0.9 + 1 * ema0.999 (AdEMAMix-style fast + slow momentum, as a velocity)
  msm-eq        mean of ema0.9, ema0.99, ema0.999
  ls<W>         least-squares slope of theta over the last W steps
  lslr<W>       least-squares slope of theta against cumulative lr over W steps, forecast slope * (s_{t+H} - s_t)
  quad<W>       quadratic fit over W steps, forecast = fitted curve at t+H minus at t
  exp<tau>      theta = a + b exp(-(x - t)/tau) fit over 3 tau steps, forecast b (exp(-H/tau) - 1)
  holt          Holt's linear trend (alpha 0.1, beta 0.02) on theta, forecast trend * H
  holtd         damped Holt (phi 0.99), forecast trend * sum_h phi^h
  pca<W>        top principal direction of the last W Delta's, scaled by the mean Delta's projection on it
  -gema         minus EMA 0.99 of the gradient (cos only)
  -adam         minus m / sqrt(v) of a slow Adam (0.99, 0.999) on the gradient (cos only)
  fit-ema       online least-squares combination of the EMAs (0.5 .. 0.999) fitted to the targets seen so far
                (only targets that ended before t; forgetting 0.98 per evaluation step); multi-scale momentum with
                learned weights
  fit-all       the same over EMAs + ls/quad/holt/pca/-gema/-adam features

Example
  python river_fit.py /work/$USER/nanogpt-ckpt/traj-rec-s0-seed0/traj --groups norm,bias --H 10,50,200
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

PHASES = [(0, 300), (300, 1000), (1000, 1500), (1500, 1950), (1950, 2400), (2400, 2701)]
EMAS = (0.5, 0.9, 0.97, 0.99, 0.995, 0.999)


def load(d):
    man = json.load(open(os.path.join(d, "manifest.json")))
    D = man["numel"]
    jump = np.fromfile(os.path.join(d, "jump.f32"), dtype=np.float32).reshape(-1, D)
    grad = np.fromfile(os.path.join(d, "grad.f32"), dtype=np.float32).reshape(-1, D)
    theta = np.fromfile(os.path.join(d, "theta.f32"), dtype=np.float32).reshape(-1, D)
    steps = [json.loads(l) for l in open(os.path.join(d, "steps.jsonl"))]
    N = min(len(jump), len(grad), len(steps), len(theta) - 1)
    return man, theta[:N + 1], jump[:N], grad[:N], steps[:N]


def group_idx(man, g):
    return np.concatenate([np.arange(p["offset"], p["offset"] + p["numel"]) for p in man["params"] if p["group"] == g])


def lsq_proj(x):
    """rows of the pseudo-inverse of the Vandermonde-like basis x (n x k) -> (k x n)."""
    return np.linalg.pinv(x)


def estimators(th, De, u, g, s, evals, Hs):
    """th (N+1, D) float64, De/u/g (N, D), s (N+1,) cumulative lr. Returns {name: {H: (len(evals), D)}}."""
    N, D = De.shape
    out = {}
    ev = set(evals)
    pos = {t: i for i, t in enumerate(evals)}

    def put(name, t, vec, scale=None):
        d = out.setdefault(name, {H: np.zeros((len(evals), D), np.float32) for H in Hs})
        for H in Hs:
            d[H][pos[t]] = vec * (H if scale is None else scale(H))

    # recursive estimators: one scan over t
    ema = {b: np.zeros(D) for b in EMAS}
    uema = {b: np.zeros(D) for b in (0.99, 0.995)}
    gema, am, av = np.zeros(D), np.zeros(D), np.zeros(D)
    lvl, tr, lvl_d, tr_d = th[0].copy(), np.zeros(D), th[0].copy(), np.zeros(D)
    phi = 0.99
    for t in range(N + 1):
        if t > 0:
            x, y, gg = De[t - 1], u[t - 1], g[t - 1]
            for b in EMAS:
                ema[b] += (1 - b) * (x - ema[b])
            for b in uema:
                uema[b] += (1 - b) * (y - uema[b])
            gema += 0.01 * (gg - gema)
            am += 0.01 * (gg - am)
            av += 0.001 * (gg * gg - av)
            y_t = th[t]
            l_new = 0.1 * y_t + 0.9 * (lvl + tr)
            tr = 0.02 * (l_new - lvl) + 0.98 * tr
            lvl = l_new
            l_new = 0.1 * y_t + 0.9 * (lvl_d + phi * tr_d)
            tr_d = 0.02 * (l_new - lvl_d) + 0.98 * phi * tr_d
            lvl_d = l_new
        if t not in ev:
            continue
        for b in EMAS:
            put(f"ema{b}", t, ema[b])
        for b in uema:
            put(f"uema{b}", t, uema[b])
        put("msm-ademamix", t, ema[0.9] + ema[0.999])
        put("msm-eq", t, (ema[0.9] + ema[0.99] + ema[0.999]) / 3)
        put("-gema", t, -gema, scale=lambda H: 1.0)
        put("-adam", t, -am / (np.sqrt(av) + 1e-12), scale=lambda H: 1.0)
        put("holt", t, tr)
        put("holtd", t, tr_d, scale=lambda H: sum(phi ** h for h in range(1, H + 1)))

    # window fits at the evaluation steps
    for t in evals:
        for W in (50, 100, 200, 400):
            if t < W:
                continue
            x = np.arange(-W, 1, dtype=np.float64)
            win = th[t - W:t + 1]
            xc = x - x.mean()
            slope = xc @ win / (xc @ xc)
            put(f"ls{W}", t, slope)
            if W in (100, 200, 400):
                sx = s[t - W:t + 1] - s[t]
                sc = sx - sx.mean()
                sl = sc @ win / max(sc @ sc, 1e-30)
                st_ = s
                put(f"lslr{W}", t, sl, scale=lambda H, t=t: st_[min(t + H, len(st_) - 1)] - st_[t])
            if W in (200, 400):
                xs = x / W
                coef = lsq_proj(np.stack([np.ones_like(xs), xs, xs * xs], 1)) @ win   # (3, D)
                d = out.setdefault(f"quad{W}", {H: np.zeros((len(evals), D), np.float32) for H in Hs})
                for H in Hs:
                    h = H / W
                    d[H][pos[t]] = coef[1] * h + coef[2] * h * h
        for tau in (100, 300, 1000):
            W = min(3 * tau, t)
            if W < 50:
                continue
            x = np.arange(-W, 1, dtype=np.float64)
            coef = lsq_proj(np.stack([np.ones_like(x), np.exp(-x / tau)], 1)) @ th[t - W:t + 1]
            d = out.setdefault(f"exp{tau}", {H: np.zeros((len(evals), D), np.float32) for H in Hs})
            for H in Hs:
                d[H][pos[t]] = coef[1] * (np.exp(-H / tau) - 1)
        for W in (200,):
            if t < W:
                continue
            X = De[t - W:t].astype(np.float64)
            G = X @ X.T
            w, V = np.linalg.eigh(G)
            pc = X.T @ V[:, -1]
            pc /= max(np.linalg.norm(pc), 1e-30)
            mean = X.mean(0)
            put(f"pca{W}", t, pc * (pc @ mean))
    return out


def fit_online(feats, target, evals, H, forget=0.98, ridge=1e-6):
    """feats: list of (len(evals), D) arrays; target (len(evals), D). Least-squares weights from targets that ended
    before t (evals[j] + H <= t), pooled over coordinates, exponentially forgotten."""
    K = len(feats)
    F = np.stack(feats, 1).astype(np.float64)   # (E, K, D)
    A, b = np.zeros((K, K)), np.zeros(K)
    out = np.zeros_like(target)
    nxt = 0
    for i, t in enumerate(evals):
        while nxt < len(evals) and evals[nxt] + H <= t:
            Fi = F[nxt]
            A = forget * A + Fi @ Fi.T
            b = forget * b + Fi @ target[nxt]
            nxt += 1
        if nxt >= 3:
            reg = ridge * np.trace(A) / K + 1e-30
            w = np.linalg.solve(A + reg * np.eye(K), b)
            out[i] = w @ F[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--groups", default="norm,bias")
    ap.add_argument("--H", default="10,50,200")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--target", default="inner", choices=("inner", "total"))
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    Hs = [int(h) for h in args.H.split(",")]
    man, theta, jump, grad, steps = load(args.traj)
    N = len(jump)
    step0 = steps[0]["step"]
    lr = np.array([r["lr_aux"] for r in steps])
    s = np.concatenate([[0.0], np.cumsum(lr)])
    # the lr after the recording ends is unknown: extend with the last value (only affects lslr near the end)
    s = np.concatenate([s, s[-1] + lr[-1] * np.arange(1, max(Hs) + 1)])
    evals = [t for t in range(max(50, args.every), N - min(Hs) + 1, args.every)]
    results = {}
    for gname in args.groups.split(","):
        idx = group_idx(man, gname)
        th = theta[:, idx].astype(np.float64)
        De = np.diff(th, axis=0)
        u = De - jump[:, idx]
        g = grad[:, idx].astype(np.float64)
        Cu = np.concatenate([np.zeros((1, len(idx))), np.cumsum(u, 0)])
        est = estimators(th, De, u, g, s, evals, Hs)
        for H in Hs:
            ev_ok = [i for i, t in enumerate(evals) if t + H <= N]
            tgt = np.zeros((len(evals), len(idx)), np.float32)
            for i in ev_ok:
                t = evals[i]
                tgt[i] = (Cu[t + H] - Cu[t]) if args.target == "inner" else (th[t + H] - th[t])
            names_ema = [f"ema{b}" for b in EMAS]
            est.setdefault("fit-ema", {})[H] = fit_online([est[n][H] for n in names_ema], tgt, evals, H)
            names_all = names_ema + [n for n in ("ls100", "ls200", "quad200", "holt", "pca200", "-gema", "-adam")
                                     if n in est]
            est.setdefault("fit-all", {})[H] = fit_online([est[n][H] for n in names_all], tgt, evals, H)
            for name, d in est.items():
                f = d[H]
                dots = np.einsum("ij,ij->i", f, tgt).astype(np.float64)
                ff = np.einsum("ij,ij->i", f, f).astype(np.float64)
                uu = np.einsum("ij,ij->i", tgt, tgt).astype(np.float64)
                err = np.einsum("ij,ij->i", tgt - f, tgt - f).astype(np.float64)
                for lo, hi in PHASES:
                    sel = [i for i in ev_ok if lo <= evals[i] + step0 < hi and ff[i] > 0]
                    if len(sel) < 3:
                        continue
                    cos = np.median(dots[sel] / np.sqrt(ff[sel] * uu[sel] + 1e-60))
                    sd, sf, su = dots[sel].sum(), ff[sel].sum(), uu[sel].sum()
                    results.setdefault(gname, {}).setdefault(H, {}).setdefault(name, {})[f"{lo}-{hi}"] = dict(
                        cos=float(cos), r2s=float(sd * sd / (sf * su + 1e-60)), c=float(sd / (sf + 1e-60)),
                        r2=float(1 - err[sel].sum() / (su + 1e-60)))
    phases = [f"{lo}-{hi}" for lo, hi in PHASES]
    for gname, byH in results.items():
        for H, byE in byH.items():
            print(f"\n## {gname}  H={H}  target={args.target}   cells: cos / R2* (c*) / R2")
            ph = [p for p in phases if any(p in v for v in byE.values())]
            print("| estimator | " + " | ".join(ph) + " |")
            print("|---|" + "---|" * len(ph))
            for name, v in byE.items():
                cells = [f"{v[p]['cos']:.2f} / {v[p]['r2s']:.2f} ({v[p]['c']:.2g}) / {v[p]['r2']:.2f}" if p in v else "–"
                         for p in ph]
                print(f"| {name} | " + " | ".join(cells) + " |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
