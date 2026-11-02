# River vs settling: when should Muon's lr decay?

The model is undertrained, so steps spent moving along the river (high lr) are worth something, and decaying too early wastes them; decaying too late leaves too little time to settle. The best point of this trade-off has not been measured on the record stack (rec-46 + our four changes, the defaults of `train_gpt.py`).

Current schedule (`set_hparams`): PowerCool lr = min(lr0, c·(T_end − t)^1.2), T_end = 2900. With its constant, Muon is flat until step 514 and then decays; at the 2640 stop it is at 7 % of lr0. The Adam groups have their own constants (the embedding starts decaying at about 1490). The EMA-Nesterov jump takes its lr ratio from the embedding group, so the Muon-only knobs below leave it unchanged.

## Knobs (branch `river-schedule`, commit `423f21a`)

| Env | Effect |
|---|---|
| `MUON_DECAY_START=s`, `MUON_LR_POWER=p` | Muon's c = lr0 / (T_end − s)^p: flat until s, then (T_end − t)^p, still reaching 0 at T_end. Unset s = the record's 514. |
| `NO_DECAY=muon` / `all` | Muon only / every group stays at lr0 for the whole run. |
| `SF_BETA=β` (`SF_START`, default the resume step) | Schedule-free (Defazio et al. 2024) around Muon: the gradient is taken at y = (1−β)·z + β·x, Muon (SOAP, radius pin, row gains, everything) steps the base iterate z, x is the lr²-weighted average of z, and validation evaluates x. An EMA-Nesterov jump made on y also moves z. The Adam groups are unchanged. |
| `LAB_LAMBDAS` | λ grid of the readout lab's extra EMAs (`READOUT_LAB=1`, `LAB_EMAS="start:tau,..."`), scored at the stop step on val and held-out data. |

## Protocol

All four experiments first differ from the record at step 514 (or, for schedule-free, at the step it starts), so every variant forks from a **step-500 checkpoint**: parents `p500-s{0,1,2}` (commit `44f7b43` = the record defaults, `CKPT_STEPS=500`, stop 2690), which are also control replicas, plus control forks `p500-ctrl-s{0,1,2}-k500` for the fork noise. Variants `<name>-s<seed>-k500`, stop 2690, scored with `python tools/fork_compare.py <variants> --parent p500 --control p500-ctrl --ks 500 --seeds 0 1 2` (Δ val_ema at 2690 vs the same-checkpoint controls, converted to steps with the controls' slope).

| Experiment | Variants | Tests |
|---|---|---|
| 1. Later decay start | `d800` (s = 800), `d1200` (s = 1200), p = 1.2 | whether more steps on the river pay off before settling |
| 2. No decay + tuned Tail-EMA | `nd-muon` (`NO_DECAY=muon`), `nd-all` (`NO_DECAY=all`), both with the readout lab over EMA start ∈ {1500, 2000, 2300} × τ ∈ {150…800} × λ ∈ {0.5…1.0} | the gap that averaging cannot close |
| 3. Schedule-free around Muon/SOAP | `sf90` (β = 0.9), `sf70` (β = 0.7), with `NO_DECAY=muon TAILEMA_TAU=0`, from step 500 | the schedule-free idea directly; not in the Track 3 record list |
| 4. Later but faster drop | `s1200p2`, `s1600p2`, `s1600p3` (decay start, power) | more steps high, then a quick settle that Tail-EMA may absorb |

Muon lr as a fraction of lr0 (T_end = 2900):

| Schedule | t = 1000 | 1500 | 2000 | 2400 | 2640 (record stop) |
|---|---|---|---|---|---|
| record (514, 1.2) | 0.76 | 0.53 | 0.31 | 0.15 | 0.070 |
| `d800` | 0.89 | 0.62 | 0.36 | 0.18 | 0.082 |
| `d1200` | 1 | 0.79 | 0.47 | 0.23 | 0.105 |
| `s1200p2` | 1 | 0.68 | 0.28 | 0.087 | 0.023 |
| `s1600p2` | 1 | 1 | 0.48 | 0.15 | 0.040 |
| `s1600p3` | 1 | 1 | 0.33 | 0.057 | 0.008 |

## Results (2026-11-02, forks from step 500, seeds 0–2)

Control forks reproduce their parents to SD 0.00007 at 2690 (pooled, 3 dof), so every difference below is far outside the fork noise. Δ = val_ema at 2690 (the run's own Tail-EMA 2100/200/0.6 readout; schedule-free: its average x) minus the same-checkpoint controls; positive = worse. Steps are Δ over the controls' end slope; beyond ≈ 100 steps they only indicate size, since the conversion is linear.

| Variant | Muon lr at 2640 | Δ ± SE | ≈ steps |
|---|---|---|---|
| control fork (`p500-ctrl`) | 0.070 | +0.00004 ± 0.00003 | −1 |
| 1. `d800` (flat to 800) | 0.082 | +0.00239 ± 0.00003 | −40 |
| 1. `d1200` (flat to 1200) | 0.105 | +0.00965 ± 0.00017 | −164 |
| 4. `s1200p2` | 0.023 | +0.00954 ± 0.00015 | −162 |
| 4. `s1600p2` | 0.040 | +0.02027 ± 0.00014 | −344 |
| 4. `s1600p3` | 0.008 | +0.02792 ± 0.00009 | −473 |
| 2. `nd-muon` (Muon flat) | 1 | +0.12921 ± 0.00040 | (≫ 1000) |
| 2. `nd-all` (all groups flat) | 1 | +0.13987 ± 0.00093 | (≫ 1000) |
| 3. `sf90` (schedule-free β 0.9, Muon flat) | 1 | +0.03797 ± 0.00074 | −644 |
| 3. `sf70` (schedule-free β 0.7) | 1 | +0.16569 ± 0.00926 | (≫ 1000) |

**Readout search (experiment 2)**, mean of 3 seeds at 2690, val / held-out:

| Arm | raw weights | Tail-EMA 2100/200, λ 0.6 | best EMA readout |
|---|---|---|---|
| control | 3.27987 | **3.27530** / 3.31241 | the same (no extra EMA beats it) |
| `nd-muon` | 3.50496 | 3.40447 | EMA 2300/150, λ 1.0: 3.39055 / 3.42697 |
| `nd-all` | 3.52988 | 3.41513 | EMA 2300/150, λ 1.0: 3.39764 / 3.43376 |

- **Every later or slower decay loses, monotonically in how much lr is kept.** Flat to 800 costs ≈ 40 steps, to 1200 ≈ 160. The "later but faster" shapes lose too, even though they end lower than the record (0.8–4 % of lr0 vs 7 %): what matters is the lr through the middle of the run, not at its end.
- **More lr hurts at once and is never paid back.** `d800` differs from the record only over 514–800, by at most 14 % more Muon lr (5 % at step 625), and it is already behind at step 625 (seed 0: 3.7543 vs 3.7479); it stays behind to 2690. On this stack, the extra steps on the river buy no later progress that offsets the extra noise. The trade-off's best point, if it is not the record's, lies on the other side: earlier or more decay.
- **Averaging closes about half of the no-decay gap.** Without Muon decay the raw weights are 0.230 worse than the control's readout; the best weight average (the last ≈ 150 steps, λ = 1) recovers 0.114 of it and leaves +0.115. Longer windows (τ 300–800, starts 1500–2000) are worse, i.e. the iterate is moving along the river fast enough that old weights hurt.
- **Schedule-free (β 0.9) recovers far more than an EMA**, 83 % of the no-decay gap (3.505 → 3.313 vs the decayed 3.275), but is still ≈ 0.038 behind the decayed schedule. β 0.7 is much worse and unstable across seeds (Δ 0.149–0.181). This is new evidence for Track 3: a schedule-free Muon/SOAP without tuning (Muon momentum kept, no warmup re-weighting, lr0 unchanged) does not beat PowerCool + Tail-EMA here.

### Schedule-free, round 2: no base momentum, early iterates discounted (2026-11-02)

Commit `9fc98d3`: `SF_MU0=1` sets Muon's momentum to 0 while schedule-free is on (pure schedule-free has no base momentum); `SF_R=r` weights the average by t^r · lr² (t from the schedule-free start), so r > 0 discounts the early iterates. All arms β 0.9, Muon flat, same step-500 forks and controls.

| Arm | Muon momentum | r | Δ val at 2690 ± SE |
|---|---|---|---|
| `sf90` (round 1) | 0.95 | 0 | +0.03797 ± 0.00074 |
| `sf90-m0` | 0 | 0 | +0.13386 ± 0.00179 |
| `sf90-m0-r1` | 0 | 1 | +0.08470 ± 0.00014 |
| `sf90-m0-r2` | 0 | 2 | +0.07637 ± 0.00016 |
| `sf90-r2` | 0.95 | 2 | +0.06702 ± 0.00067 |

val of x, mean of 3 seeds (control: its Tail-EMA readout):

| Step | 1000 | 1500 | 2000 | 2500 | 2690 |
|---|---|---|---|---|---|
| control | 3.6225 | 3.5068 | 3.3894 | 3.2885 | 3.2753 |
| `sf90` | **3.5404** | **3.4545** | **3.3793** | 3.3258 | 3.3132 |
| `sf90-m0` | 3.5731 | 3.5796 | 3.5563 | 3.4422 | 3.4091 |
| `sf90-m0-r2` | 3.5502 | 3.4853 | 3.4180 | 3.3652 | 3.3516 |
| `sf90-r2` | 3.5999 | 3.4938 | 3.4088 | 3.3526 | 3.3423 |

- **Both "fairer" changes make it worse.** Without Muon momentum, schedule-free stalls (x barely moves from 1000 to 2000, 3.573 → 3.556); discounting the early iterates helps that arm (+0.134 → +0.076) but not enough. With momentum kept, r = 2 is worse than r = 0 (+0.067 vs +0.038). On this stack Muon/SOAP's momentum is doing work the y-interpolation does not replace, and the long uniform average is the useful part.
- **Schedule-free is ahead of the decayed schedule until ≈ step 2000** (`sf90`: −0.082 at 1000, −0.052 at 1500, −0.010 at 2000) and loses only in the last ≈ 600 steps, where the decay settles the weights and the Tail-EMA readout kicks in. So schedule-free does buy the "river" progress early; what it lacks is the final settle. A hybrid (schedule-free with flat lr, then a short decay at the end, starting from z or from x) is the natural next test; it is not run here.
