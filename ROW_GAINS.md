# Row gains: what they do and why they work

Only claims backed by measurements are listed. Full tables and run details: `experiment.md` (rounds 3–6, full-run validation, "Anatomy of the row-gain effect", "Frozen lengths without weight decay").

**Summary.** rec-46 changes a row's length only as a side effect of the orthogonalized Muon update. Row gains give each row's length its own optimizer: Adam on the gradient with respect to that length.
- **Size of the effect:** about 12 steps on its own. Together with the Tail-EMA retune it moved the record from 2690 to 2665 steps.
- **Where it acts:** v, o and mlp.fc. q and k contribute about nothing; on mlp.proj it hurts.
- **What it needs:** one length per row (not per matrix), switched on from step 1000.
- **What training reaches:** a different solution. The attention matrices end larger, and row lengths within a matrix are more spread out. That solution predicts better, mostly at context positions ≥ 128 and on rare tokens.

## 1. The change

For q, k, v, o (attn.proj) and fc (mlp.fc), from step 1000 on:
- The Muon step only rotates each row; the row is rescaled back to its old length.
- Then each length takes its own Adam step:
  - γ_i = ⟨g_i, W_i⟩ / ‖W_i‖, the gradient with respect to row i's length (raw gradient g);
  - Adam(0.9, 0.95), bias-corrected;
  - W_i ← W_i · (1 − a·(η_t/η₀)·m̂_i/√v̂_i), with a = 0.02.

Code: `train_gpt.py`, the `gain` branch of `Muon.step`. Math: `OPTIMIZER.md`, step 8. Settings: `ROW_GAIN_LR=0.02 ROW_GAIN_SCOPE=q,k,v,o,fc ROW_GAIN_START=1000` (the defaults on `main`).

## 2. Claims and evidence

**C1. It works, and the size is measured.**
- **Fork from step 1000, 3 seeds:** −0.00081 ± 0.00010, about +12 steps.
- **Learning-rate plateau:** 0.02–0.03 (+12.1 / +11.4 / +11.4 steps; 0.015: +10.5; 0.01: +7.0).
- **16-seed full validation with the Tail-EMA retune:** passes the Track 3 rule at 2665 (0.00530 at n = 16; 0.00403 at n = 8).
- **Paired with rec-46 at 2690 (seeds 0–7):** −0.00154 ± 0.00038, with all 8 seeds lower.

**C2. It has to be per row; one gain per matrix hurts.**
- One length per matrix, everything else the same: +0.00214 ± 0.00016 (−32 steps).
- Per-row gains at the same learning rate: −0.00050 (+7).
- Caveat: the per-matrix version reuses the per-row learning rate.

**C3. It helps where a row's length sets the output scale, and hurts on mlp.proj.** All of these are forks from step 1000.

| Gains on | Δ val_ema | Steps |
|---|---|---|
| q, k | −0.00005 ± 0.00007 | +0.7 |
| v, o | −0.00032 ± 0.00002 | +4.8 |
| fc only | −0.00045 ± 0.00006 | +6.7 |
| mlp.proj only | +0.00025 ± 0.00001 | −3.7 |

q, k, v, o, fc vs all six including mlp.proj (anatomy forks, seeds 0 / 1): −0.00057 / −0.00088 vs −0.00023 / −0.00049.

The model normalizes each q and k head (`q, k = norm(q), norm(k)`), so q/k row lengths can't set the output's scale. v, o and fc row lengths do.

**C4. The gain comes from lengths following their own gradient, not from weights getting bigger.** Weight decay is off on the gained matrices in both forks below (`ROW_GAIN_NO_CWD=1`), so they differ only in whether lengths are learned.

| Fork (from step 1000) | Δ vs control, s0 / s1 | Steps | Matrix norm vs control |
|---|---|---|---|
| Lengths learned (row gains) | −0.00076 / −0.00103 | +13 | +33 % to +91 % |
| Lengths frozen by the optimizer step | +0.00117 / +0.00103 | −16 | −13 % to +2 % |

- Learned vs frozen is worth about 30 steps.
- The frozen fork ends with roughly the control's matrix norms, yet it is 16 steps worse than the control.
- In the frozen fork the optimizer step changes no length (max 2·10⁻⁷ per step). The EMA-Nesterov lookahead still grows lengths about 20 % until step 1950, as it does in every run.

**C5. With row gains, lengths move more and their deviations die out faster.** These are measurements; whether they cause the loss gain is not tested. Control → row gains, mean of seeds 0–1:

| Statistic | Control → row gains |
|---|---|
| Per-step length change \|Δlog r\| | 2–6× larger (fc 2×) |
| Lifetime τ_d of a length deviation from its trend, q/k/v/o | 38 → 24 steps (1000–1500); 31–33 → 22–23 (2000–2690) |
| corr(γ, d): a row longer than its trend gets pushed to shrink, q/k/v/o | 2–5× stronger |
| Matrix norm at 2690 | q, k, v, o +10 % to +37 %; fc −5 % |
| Spread of row lengths within a matrix (CV) | +25 % to +65 % |

**C6. The benefit builds up during steps 1000–2000 and is kept.**
- Gains switched off at 2000 keep about 77 % of the advantage at 2690.
- Gains switched on only at 2000 get about 18 %.

Starting at step 0 is impossible. attn.proj and mlp.proj start at zero, and a zero-length row stays zero under a direction-only step and a multiplicative length step. The run starting at step 0 had loss 5.86 at step 125 (vs 4.44) and NaN at step 440. Starting at 500 or at 1000 both work.

**C7. Training reaches a different solution whose lengths are tied to its own directions.**
- Swapping row lengths between two *control* forks from the same checkpoint costs +0.002 (swapping matrix norms costs about 0).
- Swapping between control and row gains costs +0.28 to +0.51 by row, and +0.14 to +0.21 by matrix norm. This holds in both directions and both seeds.

**C8. The payoff is long context and rare tokens.**

| Measurement | Seed 0 / seed 1 |
|---|---|
| Share of the gain at context positions ≥ 128 | 88 % / 90 % |
| Share of the gain on the rarest 40 % of target tokens | 52 % / 98 % |
| Last-layer attention output vs residual stream | +27 % / +58 % |
| Layer-0 MLP output vs residual stream | +12 % to +14 % |

**C9. The Tail-EMA retune (start 2100, window 200) is a separate gain and stacks with row gains.**
- On its own it is worth about 8 steps, confirmed on a held-out shard.
- On the row-gain trajectory, rec-46's Tail-EMA (2400/150) is 9 steps worse.

## 3. How each statistic is computed

All per-row statistics come from `ANATOMY=1` dumps. A row i of a hidden matrix has weights W_i and length r_i = ‖W_i‖, and is recorded at every step t from the fork step.

| Statistic | Definition | Where |
|---|---|---|
| Δ val_ema, steps | val_ema@2690 of the fork − mean of same-checkpoint controls; steps = −Δ / 0.000067. Fork noise SD 0.000075 (fork at 1000) | `fork_compare.py` |
| Track 3 rule | (3.28 − mean val_ema)·√n ≥ 0.004 at each dense val step; seed-paired difference vs rec-46 | `fork_compare.py --full` |
| \|Δlog r\| | mean over rows and steps of \|log r_i(after step) − log r_i(before step)\| | `anatomy.py` |
| τ_d | d_i,t = log r_i,t − its centred 201-step moving average; first lag where the pooled autocorrelation of d drops below 1/e | `anatomy.py` |
| corr(γ, d) | γ_i,t = ⟨g_i, W_i⟩ / r_i; pooled correlation of γ_i,t with d_i,t over rows and steps | `anatomy.py` |
| Matrix norm ratio, row CV | log(‖W‖_F / ‖W_control‖_F) per matrix, averaged per type; CV = SD/mean of r_i within a matrix | snapshots at 2690 (`SNAP_STEPS`) |
| Frozen check | max over steps of \|Δlog r\| across the optimizer step | `anatomy/rows.pt` |
| Residual budget | RMS(block's attention or MLP output) / RMS(residual stream entering it), on 16 val sequences | `anatomy.py` (function section) |
| Per-token shares | Δ per-token Tail-EMA loss (variant − control), summed by context position and by target-token frequency quintile, divided by the token count | `anatomy.py` |
| Transplant Δ | L(directions of A with row or matrix lengths of B) − L(A) on val and held-out, read against a control-vs-control pair | `ROW_TRANSPLANT`, `anatomy.py --transplant` |

## 4. Runs behind each claim

Forks resume from the `ckpt-rec46` checkpoints at step 1000 unless noted. Fork controls are the parent `ckpt-rec46` plus `fork-ctrl`. Anatomy controls are `anat-C`, plus `anat-C2` for the transplant baseline.

| Claim | Runs (seeds) |
|---|---|
| C1 effect, learning rate | `abl-noproj-lr0.01`, `-lr0.015`, `-lr0.02`, `-lr0.025`, `-lr0.03`, `abl-vofc-lr0.02` (0–2); control `fork-ctrl` |
| C1 full validation | `cand-rg20` (0–15); controls `ckpt-rec46`, `aux-control`, `ctrl-rec46`; leaderboard `rowgain-tailema2100-2665-s0` |
| C2 per row vs per matrix | `abl-rg15-matrix` vs `fork-rg15` (0–2) |
| C3 where | `abl-rg15-qk`, `-vo`, `-attn`, `-fc`, `-mlp`, `-proj` (0–2); `anat-G` vs `anat-Gproj` (0–1) |
| C4 learned vs frozen | `anat-Gnc`, `anat-G0nc`, `anat-C` (0–1) |
| C5 length dynamics | `anat-C`, `anat-G` (0–1) |
| C6 timing | `anat-Goff2000`, `anat-Gon2000` (0–1); start step: `full-rg15-s0start` (diverged), `full-noproj-start500`, `full-noproj-start1000`, `full-rg15-1000start` (0–2) |
| C7 co-adaptation | `tp-CG` (0–1), `tp-CC2` (0) on snapshots of `anat-C`, `anat-G`, `anat-C2` |
| C8 function, tokens | `anat-C`, `anat-G` (0–1): func stats and `tokloss_rank*.pt` |
| C9 Tail-EMA | readout grids `lab`, `lab2`, `lab3` (forks from 2000, held-out check); `lab4` on the `full-noproj-start1000` trajectory |

Reproduce the analysis:

```
python fork_compare.py abl-noproj-lr0.02 abl-rg15-matrix abl-rg15-qk abl-rg15-vo --parent ckpt-rec46 --control fork-ctrl
python fork_compare.py --full cand-rg20 --full-controls ckpt-rec46,aux-control,ctrl-rec46
python anatomy.py --control anat-C --variants anat-G,anat-G0nc,anat-Gnc,anat-Goff2000,anat-Gon2000,anat-Gproj --seeds 0,1
python anatomy.py --transplant tp-CG --seeds 0,1
python anatomy.py --transplant tp-CC2 --seeds 0
```

Still running: `nocwd-s0` to `nocwd-s3`, full runs that test dropping weight decay on the gained matrices (the C4 forks hinted at about +2.5 steps). They are not part of this story until they finish.
