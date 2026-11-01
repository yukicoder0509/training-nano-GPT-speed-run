# Norm-gain jump: how path efficiency led to the 2640 record

Only measured results are listed. Details and per-seed numbers: `experiment.md`, sections "Path efficiency of the weight trajectory", "Per-group jump on the norm gains, early stage", "Self-tuning the norm-gain jump from many batches", "New record: 2640 steps".

## 1. The measurement

Path efficiency of the weight trajectory, per parameter group, from a weight snapshot every 50 steps:

eff_K = ‖θ_t − θ_{t−K}‖ / Σ_{j<K} ‖Δθ_{t−K+j}‖, for K = 50 and 200.

- Δθ is the whole iteration's change, including the EMA-Nesterov jump and weight decay.
- A group's norm is that of its tensors flattened into one vector. Each table cell is the median over the windows that end in that phase.
- **Reading eff200/eff50:**
  - 1: a straight path, a consistent drift.
  - 0.5: a random walk (eff ∝ 1/√K).
  - Below 0.5: the steps cancel, i.e. the parameters bounce around a point.

Code: `optim_dynamics.PathEfficiency`, switched on with `PATH_EFF=1`. Tables: `python tools/path_eff.py --table <run>`.

## 2. rec-46 baseline: where the drift is

Run `patheff-rec46-s0`: rec-46 settings, full run, seed 0. Cells: eff50 / eff200 (eff200/eff50).

| group | 0–500 | 500–1000 | 1000–1500 | 1500–2000 | 2000–2400 | 2400–2700 |
|---|---|---|---|---|---|---|
| attn_q | 0.42 / 0.19 (0.46) | 0.52 / 0.28 (0.54) | 0.53 / 0.30 (0.56) | 0.52 / 0.31 (0.58) | 0.51 / 0.28 (0.56) | 0.52 / 0.30 (0.57) |
| attn_k | 0.41 / 0.19 (0.47) | 0.52 / 0.28 (0.54) | 0.52 / 0.29 (0.56) | 0.52 / 0.30 (0.57) | 0.51 / 0.28 (0.55) | 0.52 / 0.29 (0.57) |
| attn_v | 0.41 / 0.20 (0.50) | 0.49 / 0.26 (0.52) | 0.49 / 0.26 (0.52) | 0.50 / 0.26 (0.53) | 0.48 / 0.25 (0.51) | 0.50 / 0.27 (0.53) |
| attn_o | 0.41 / 0.20 (0.47) | 0.50 / 0.25 (0.51) | 0.50 / 0.26 (0.51) | 0.52 / 0.27 (0.52) | 0.51 / 0.26 (0.51) | 0.53 / 0.28 (0.54) |
| mlp_up | 0.43 / 0.21 (0.50) | 0.54 / 0.29 (0.53) | 0.54 / 0.29 (0.54) | 0.55 / 0.30 (0.55) | 0.53 / 0.28 (0.52) | 0.55 / 0.30 (0.54) |
| mlp_down | 0.45 / 0.22 (0.50) | 0.55 / 0.29 (0.53) | 0.55 / 0.30 (0.54) | 0.56 / 0.30 (0.55) | 0.54 / 0.29 (0.53) | 0.56 / 0.31 (0.55) |
| embed | 0.38 / 0.19 (0.51) | 0.38 / 0.19 (0.49) | 0.38 / 0.18 (0.48) | 0.38 / 0.18 (0.48) | 0.37 / 0.18 (0.49) | 0.35 / 0.18 (0.52) |
| lm_head | 0.36 / 0.19 (0.54) | 0.37 / 0.20 (0.54) | 0.37 / 0.20 (0.52) | 0.38 / 0.20 (0.52) | 0.37 / 0.18 (0.49) | 0.39 / 0.20 (0.50) |
| lm_head_bias | 0.17 / 0.10 (0.58) | 0.18 / 0.09 (0.49) | 0.20 / 0.08 (0.41) | 0.22 / 0.08 (0.35) | 0.25 / 0.08 (0.32) | 0.31 / 0.10 (0.32) |
| **norm** | 0.44 / 0.40 (**0.90**) | 0.31 / 0.29 (**0.95**) | 0.22 / 0.17 (0.77) | 0.20 / 0.11 (0.54) | 0.21 / 0.07 (0.34) | 0.26 / 0.08 (0.32) |
| bias | 0.25 / 0.21 (**0.84**) | 0.21 / 0.15 (0.68) | 0.22 / 0.12 (0.53) | 0.23 / 0.10 (0.42) | 0.24 / 0.08 (0.32) | 0.27 / 0.09 (0.32) |

- **Hidden matrices, embedding, lm_head:** the ratio is 0.46–0.58 in every phase. Beyond about 50 steps they move like a random walk, so there is no long drift to exploit.
- **Norm gains:** the ratio is 0.90–0.95 before step 1000, a nearly straight path under the bounce. This is the one place a drift-following estimator could gain. Biases are similar but weaker (0.84 in 0–500).
- **All 1-D groups late:** from step 2000 the ratio is 0.32–0.34. They oscillate around a point that barely moves.
- **Robustness:** two seeds of the current-best setting (`patheff-s0`, `patheff-s1`) agree with each other to ±0.01 and with rec-46 to ±0.02 everywhere.

## 3. Jump size on the norm gains

The EMA-Nesterov jump moves every parameter by W ← W + β·b before the forward pass, where b is an EMA (0.99) of the per-step displacement and β = 0.3·η_t/η₀ over steps 300–1950. `ENA_GROUP=norm ENA_G_MULT=m` multiplies β for the norm gains only.

These are forks from step 50 of the current-best parents `ckpt-best`, seeds 0–2. Each is scored against the same-checkpoint control `ena-ctrl` (1×). Fork noise from step 50 is SD 0.00029.

| norm-gain jump | Δ val_ema@2690 | ≈ steps | runs |
|---|---|---|---|
| 0× (off) | +0.00123 ± 0.00020 | −21 | `ena-off` |
| 1× (rec-46 size) | – | – | `ena-ctrl` |
| 2× | −0.00072 ± 0.00017 | +12 | `ena-x2` |
| **2.5×** | **−0.00099 ± 0.00017** | **+17** | `ena-x25` |
| 3× | −0.00084 ± 0.00030 | +14 | `ena-x3` |
| 4× | +0.00086 ± 0.00012 | −15 | `ena-x4` |
| 2× during 300–1000, 1× after | −0.00063 ± 0.00013 | +11 | `ena-x2early` |
| 1× until 1000, 2× after | −0.00033 ± 0.00017 | +6 | `ena-x2late` |

Path efficiency of the norm gains for each jump size, mean of the 3 seeds. Cells: eff50 / eff200 (eff200/eff50).

| norm-gain jump | 50–500 | 500–1000 | 1000–1500 | 1500–2000 | 2000–2400 | 2400–2690 |
|---|---|---|---|---|---|---|
| 0× (off) | 0.41 / 0.37 (0.92) | 0.27 / 0.24 (0.89) | 0.20 / 0.13 (0.66) | 0.19 / 0.08 (0.43) | 0.21 / 0.07 (0.32) | 0.25 / 0.08 (0.31) |
| 1× | 0.44 / 0.41 (0.91) | 0.31 / 0.29 (0.95) | 0.23 / 0.17 (0.77) | 0.20 / 0.11 (0.54) | 0.21 / 0.07 (0.33) | 0.25 / 0.08 (0.31) |
| 2× | 0.51 / 0.43 (0.85) | 0.38 / 0.38 (1.00) | 0.30 / 0.26 (0.89) | 0.24 / 0.19 (0.80) | 0.20 / 0.07 (0.35) | 0.24 / 0.07 (0.30) |
| 2.5× | 0.56 / 0.44 (0.79) | 0.43 / 0.44 (1.02) | 0.37 / 0.34 (0.93) | 0.29 / 0.27 (0.93) | 0.19 / 0.07 (0.34) | 0.24 / 0.07 (0.30) |
| 3× | 0.58 / 0.46 (0.79) | 0.51 / 0.52 (1.02) | 0.47 / 0.45 (0.96) | 0.38 / 0.39 (1.02) | 0.19 / 0.06 (0.33) | 0.24 / 0.07 (0.30) |
| 4× | 0.58 / 0.49 (0.84) | 0.75 / 0.73 (0.97) | 0.92 / 0.89 (0.97) | 0.97 / 0.97 (1.00) | 0.20 / 0.07 (0.33) | 0.25 / 0.08 (0.32) |

The hidden matrices' path efficiency is unchanged by the norm-gain jump in every fork (±0.005).

**What the results show.**
1. **The jump on the norm gains is worth about 20 steps at rec-46's size, and that size is too small.** The optimum is 2.5× (+17 steps); 4× is worse than 1×.
2. **Most of the gain comes from 300–1000,** the phase where the norm gains' path is straightest in rec-46: 2× there alone gives +11, 2× after 1000 alone gives +6.
3. **Why the jump amplifies the drift, and why 4× breaks.** b averages the whole displacement, jump included. For a slowly varying inner update u, the displacement therefore settles at u/(1 − β): the jump amplifies the persistent part of the update, while b averages out the bounce. At full learning rate β is 0.3, 0.6, 0.75, 0.9 and 1.2 for 1×, 2×, 2.5×, 3× and 4×, giving amplification of 1.4×, 2.5×, 4×, 10× and unstable.

   At 4× the norm gains' 200-step chord reaches 194 at step 1500, against 9 in the control. Their path becomes a straight line (eff200 0.97 in 1500–2000), and the loss is 15 steps worse.
4. **Straightness is not the target by itself.** A self-tuning fork that ran at 3× during 50–600 had eff200 0.52 in 500–1000 and gained nothing (section 4).

## 4. What did not work

All of these are forks from step 50, 3 seeds each.

| variant | Δ steps | runs |
|---|---|---|
| jump window starting at 50 instead of 300 | −5 | `ena-early` |
| least-squares velocity instead of the EMA b (λ 0.99) | −4 | `ena-ls` |
| self-tuning from the one-batch sign of s = ⟨g, d⟩, window 50–1950 | −3 | `ena-auto` |
| same, window 50–2690 | +2 | `ena-autofull` |
| same sign rule, window 300–1950, floor 0.1× | +10.6 | `ena-autos300` |
| self-tuning from the z-score of s over ~50 batches | +7.7 | `ena-autoz` |
| self-tuning from the z-score of Adam's own step along the jump | +12.9 (stays at the 3× cap = fixed 3×) | `ena-autou` |

- **The slope signal says "shrink" late, even though a larger jump helps there.** In the control, s > 0 (overshoot along the jump) on 34 % of steps in 300–700, 59 % in 700–1200 and 92 % in 1200–1950. The slope-based rules shrink β to the floor by step 900–1100.
- **Adam's own step says "grow" throughout.** It keeps moving along the jump on 76–98 % of steps, so that rule runs to the cap.
- **Neither signal settles on a size in between.** Their results are set by their cap or floor, and a fixed size does as well or better.

## 5. Validation and the record

- **8-seed screen,** full runs, seeds 0–7, against `nocwd` (the current best before this change) on the same seeds:
  - `norm25` (2.5×): paired −0.00076 ± 0.00042 at 2690, 7 of 8 lower;
  - `norm3` (3×): −0.00064 ± 0.00031, 6 of 8 lower.

  Both pass the Track 3 rule at 2650, the earliest step those runs validated every 5 steps.
- **16-seed validation** (`norm25d`, 2.5×, seeds 0–15, validation every 5 steps from 2600): the earliest passing step is **2640** at both n = 8 (0.00419) and n = 16 (0.00447). 2635 fails both (0.00325, 0.00313).
- **Leaderboard:** `normjump25-2640-s0`, seed 0, `STOP_STEP=2640`: val_loss 3.27866.

## 6. The new record, measured the same way

Run `patheff-rec2640-s0`: the 2640-record settings (defaults on `main`), full run to 2700, seed 0. Cells: eff50 / eff200 (eff200/eff50).

| group | 0–500 | 500–1000 | 1000–1500 | 1500–2000 | 2000–2400 | 2400–2700 |
|---|---|---|---|---|---|---|
| attn_q | 0.42 / 0.19 (0.46) | 0.52 / 0.28 (0.54) | 0.53 / 0.30 (0.58) | 0.52 / 0.31 (0.58) | 0.50 / 0.28 (0.55) | 0.51 / 0.29 (0.56) |
| attn_k | 0.41 / 0.19 (0.47) | 0.52 / 0.28 (0.54) | 0.53 / 0.31 (0.58) | 0.52 / 0.30 (0.58) | 0.50 / 0.27 (0.54) | 0.51 / 0.29 (0.56) |
| attn_v | 0.41 / 0.20 (0.50) | 0.49 / 0.25 (0.52) | 0.49 / 0.26 (0.53) | 0.49 / 0.27 (0.54) | 0.48 / 0.25 (0.52) | 0.50 / 0.26 (0.53) |
| attn_o | 0.41 / 0.19 (0.47) | 0.50 / 0.25 (0.50) | 0.50 / 0.26 (0.52) | 0.51 / 0.27 (0.53) | 0.50 / 0.26 (0.51) | 0.52 / 0.27 (0.53) |
| mlp_up | 0.43 / 0.21 (0.50) | 0.54 / 0.29 (0.53) | 0.54 / 0.30 (0.55) | 0.54 / 0.30 (0.55) | 0.53 / 0.28 (0.52) | 0.54 / 0.29 (0.54) |
| mlp_down | 0.45 / 0.22 (0.50) | 0.55 / 0.29 (0.53) | 0.55 / 0.30 (0.54) | 0.56 / 0.30 (0.55) | 0.54 / 0.29 (0.53) | 0.56 / 0.31 (0.55) |
| embed | 0.38 / 0.19 (0.51) | 0.38 / 0.19 (0.49) | 0.38 / 0.18 (0.48) | 0.38 / 0.18 (0.48) | 0.37 / 0.18 (0.49) | 0.35 / 0.18 (0.52) |
| lm_head | 0.36 / 0.19 (0.54) | 0.37 / 0.20 (0.54) | 0.37 / 0.20 (0.53) | 0.38 / 0.20 (0.52) | 0.37 / 0.18 (0.49) | 0.39 / 0.20 (0.50) |
| lm_head_bias | 0.17 / 0.10 (0.59) | 0.18 / 0.09 (0.49) | 0.20 / 0.08 (0.42) | 0.22 / 0.08 (0.36) | 0.25 / 0.08 (0.32) | 0.31 / 0.10 (0.33) |
| **norm** | 0.51 / 0.42 (0.83) | 0.43 / 0.44 (**1.02**) | 0.37 / 0.35 (**0.94**) | 0.29 / 0.27 (**0.94**) | 0.19 / 0.07 (0.35) | 0.24 / 0.07 (0.30) |
| bias | 0.24 / 0.20 (0.85) | 0.21 / 0.14 (0.66) | 0.21 / 0.12 (0.57) | 0.22 / 0.09 (0.44) | 0.23 / 0.08 (0.33) | 0.27 / 0.08 (0.32) |

Against rec-46, only the norm row changes. The norm gains' path stays straight until the jump window ends at 1950:

| norm gains | 500–1000 | 1000–1500 | 1500–2000 |
|---|---|---|---|
| ratio, rec-46 | 0.95 | 0.77 | 0.54 |
| ratio, new record | 1.02 | 0.94 | 0.94 |
| eff200, rec-46 | 0.29 | 0.17 | 0.11 |
| eff200, new record | 0.44 | 0.35 | 0.27 |

From 2000 on, the norm row is the same as rec-46's. Every other group is within ±0.02 of rec-46.

## 7. Reproduce

```
python tools/path_eff.py --table patheff-rec46-s0 patheff-rec2640-s0
python tools/fork_compare.py ena-off ena-x2 ena-x25 ena-x3 ena-x4 ena-x2early ena-x2late --parent ckpt-best --control ena-ctrl --ks 50
python tools/fork_compare.py --full norm25d
```
