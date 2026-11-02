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

## Results

(pending)
