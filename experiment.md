# Experiments

## 2026-10-27 — Re-align the LR / momentum schedule with the 2690 stop

**Hypothesis.** rec-46 stops at step 2690, but its schedule ends at 2900. As a result the PowerCool LR never fully decays before the final eval, and the Muon momentum cooldown (2700–2900) never starts. Tail-EMA may be partly covering for the unfinished decay. #44 gained 25 steps by moving the endpoint from 2980 to 2900, so this experiment moves it further, toward 2700–2750, and shifts the momentum cooldown with it.

**Setup.**
- Code: branch `exp/schedule-realign`, commit `8c328c0`, based on rec-46 (`30980b3`). Unset knobs reproduce rec-46.
- Seed 0, single run per setting, 8×H200, stopped at `STOP_STEP=2690`.
- W&B: forge.coreweave.com → `bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt`, with optimization dynamics on.
- Launch: `T_END=<t> WANDB_RUN_NAME=sched-tend<t> sbatch --job-name=sched-tend<t> run_record.sbatch exp/schedule-realign`

| Knob (env) | Default (= rec-46) | Meaning |
|---|---|---|
| `T_END` | 2900 | PowerCool endpoint `t_end`. `power_c` is unchanged, so the decay keeps its shape and shifts earlier by `2900 - T_END` |
| `MU_END` | `T_END` | End of the Muon momentum cooldown (mu 0.95 → 0.85 over the 200 steps before it). Held at 0.85 after `MU_END` |
| `TAILEMA_TAU` | 150 | Tail-EMA time constant |
| `TAILEMA_START` | 2400 | Step where the Tail-EMA starts accumulating |
| `TAILEMA_LAMBDA` | 0.6 | Blend weight of the reported readout `(1-λ)·w + λ·ema` |
| `TAILEMA_LAMBDA_SWEEP` | `0.2,0.4,0.8,1.0` | Extra λ readouts at the final step only, logged as `val/ema_loss_lam<λ>`. Eval only; training is unaffected |

Only `T_END` was varied, so `MU_END = T_END`. All Tail-EMA settings stayed at their defaults. I also added extra eval steps at 2550, 2600, 2650 and 2675; these are eval only and don't change the trajectory.

Fraction of peak Muon LR remaining at step 2690: 0.14 % (`T_END`=2700), 0.6 % (2725), 1.2 % (2750), 2.5 % (2800), 5.4 % (2900).

**Results** (step 2690; "Tail-EMA" is the reported `val_ema_loss` with λ = 0.6):

| T_END | raw val | Tail-EMA | λ=0.2 | λ=0.4 | λ=0.8 | λ=1.0 | Slurm | W&B run |
|---|---|---|---|---|---|---|---|---|
| 2700 | 3.28357 | 3.28430 | 3.28370 | 3.28395 | 3.28476 | 3.28532 | 620179 | [6ip7nzp3](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/6ip7nzp3) |
| 2725 | 3.28037 | 3.28112 | 3.28047 | 3.28071 | 3.28167 | 3.28238 | 620180 | [fj50iv20](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/fj50iv20) |
| 2750 | **3.27968** | 3.28024 | 3.27966 | 3.27984 | 3.28087 | 3.28169 | 620181 | [kqimchzo](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/kqimchzo) |
| 2800 | 3.28033 | 3.28004 | 3.27985 | 3.27975 | 3.28071 | 3.28174 | 620182 | [or0g0bdl](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/or0g0bdl) |
| 2900 (control) | 3.28247 | **3.27844** | 3.28017 | 3.27884 | 3.27892 | 3.28027 | 620183 | [b8zjoa3e](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/b8zjoa3e) |

The control matches rec-46 (`val_ema_loss` 3.27848, run `b1ip7q9i`).

Reported metric (λ=0.6) at earlier steps:

| T_END | 2550 | 2600 | 2650 | 2675 | 2690 |
|---|---|---|---|---|---|
| 2700 | 3.28851 | 3.28630 | 3.28489 | 3.28447 | 3.28430 |
| 2725 | 3.28638 | 3.28373 | 3.28198 | 3.28139 | 3.28112 |
| 2750 | 3.28650 | 3.28347 | 3.28138 | 3.28062 | 3.28024 |
| 2800 | 3.28822 | 3.28455 | 3.28171 | 3.28060 | 3.28004 |
| 2900 | 3.28977 | 3.28505 | 3.28116 | 3.27938 | 3.27844 |

Logs: `logs/sched-tend<T_END>-<jobid>.out`. The training logs and `.dyn.jsonl` files sit next to them:
2700 `logs/5b9d4cee-aebe-4c52-9daf-902a75a4f013.txt`, 2725 `logs/93d3bbc5-c3f5-4315-a58a-36408bcb0b38.txt`,
2750 `logs/5c1ff6f9-c80d-4b5b-aca8-03d82297969b.txt`, 2800 `logs/6688847e-ecca-40a3-ad41-49b09cafdac6.txt`,
2900 `logs/db03a98b-6559-4b14-b762-a033717fa017.txt`.

**Conclusion: negative. Keep `T_END`=2900.**
- An earlier endpoint does improve the raw weights, by up to 0.0028 at `T_END`=2750.
- But the Tail-EMA gain disappears once the LR has fully decayed. It is +0.004 at 2900, about 0 at 2750–2800, and negative at 2700–2725.
- The rec-46 combination (partly decayed LR + Tail-EMA) wins by about 0.0012 over the best shortened setting.
- Retuning λ alone doesn't close the gap. On 2900, λ = 0.4–0.8 gives nearly the same result.
- Caveat: these are single runs with gaps around 0.001, which may be close to seed-to-seed variation. The trends in both columns are consistent across all five settings, though.

**Follow-ups (not run).** Retune the Tail-EMA together with a shorter schedule so the average covers the end of the decay. For example, `T_END=2750` with `TAILEMA_START`≈2550 or `TAILEMA_TAU`≈75. Another option is to set `MU_END` separately from `T_END`.

### Follow-up: retune the Tail-EMA together with a shorter schedule (6 runs)

**Setup.** Same code, commit and seed as above (`exp/schedule-realign` @ `8c328c0`, seed 0, stop at 2690). The grid is `T_END` ∈ {2750, 2800} × `TAILEMA_START` ∈ {2400, 2550} × `TAILEMA_TAU` ∈ {75, 150}. The 2400/150 cells are the runs above. `MU_END = T_END` and `TAILEMA_LAMBDA` = 0.6 (reported).
Launch: `T_END=<t> TAILEMA_START=<s> TAILEMA_TAU=<τ> WANDB_RUN_NAME=sched-tend<t>-ts<s>-tau<τ> sbatch --job-name=<same> run_record.sbatch exp/schedule-realign`

**Results** (step 2690; "Tail-EMA" is the reported `val_ema_loss` with λ = 0.6):

| T_END | start | τ | raw val | Tail-EMA | λ=0.2 | λ=0.4 | λ=0.8 | λ=1.0 | Slurm | W&B run |
|---|---|---|---|---|---|---|---|---|---|---|
| 2750 | 2400 | 150 | 3.27968 | 3.28024 | 3.27966 | 3.27984 | 3.28087 | 3.28169 | 620181 | [kqimchzo](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/kqimchzo) |
| 2750 | 2400 | 75 | 3.28093 | 3.28126 | 3.28096 | 3.28108 | 3.28154 | 3.28190 | 621140 | [g9l0qbfs](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/g9l0qbfs) |
| 2750 | 2550 | 150 | 3.27989 | 3.28098 | 3.28004 | 3.28041 | 3.28177 | 3.28278 | 621141 | [t7c26u21](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/t7c26u21) |
| 2750 | 2550 | 75 | 3.28036 | 3.28086 | 3.28043 | 3.28061 | 3.28122 | 3.28168 | 621142 | [6lbvofjb](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/6lbvofjb) |
| 2800 | 2400 | 150 | 3.28033 | 3.28004 | 3.27985 | 3.27975 | 3.28071 | 3.28174 | 620182 | [or0g0bdl](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/or0g0bdl) |
| 2800 | 2400 | 75 | 3.27916 | 3.27881 | 3.27886 | **3.27874** | 3.27907 | 3.27951 | 621143 | [2g09rgx1](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/2g09rgx1) |
| 2800 | 2550 | 150 | 3.28060 | 3.28103 | 3.28032 | 3.28047 | 3.28200 | 3.28338 | 621144 | [c0blrozl](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/c0blrozl) |
| 2800 | 2550 | 75 | 3.27904 | 3.27897 | 3.27881 | 3.27879 | 3.27935 | 3.27993 | 621145 | [xugj9pdq](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/xugj9pdq) |
| 2900 (control) | 2400 | 150 | 3.28247 | **3.27844** | 3.28017 | 3.27884 | 3.27892 | 3.28027 | 620183 | [b8zjoa3e](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/b8zjoa3e) |

Training logs: 2750/2400/75 `logs/3d32d2db-8b5d-4af0-80b8-d16c6881e512.txt`, 2750/2550/150 `logs/1fbbaeec-f278-4e5b-838f-b557e63993fe.txt`,
2750/2550/75 `logs/21ba6c3b-37bd-4992-8da6-2b62d43ba642.txt`, 2800/2400/75 `logs/84de63fd-91f1-4cab-bbea-0a83dd9474e5.txt`,
2800/2550/150 `logs/2137dfb7-7ffd-4b82-a83a-b4a6c613e804.txt`, 2800/2550/75 `logs/26de1961-b6fa-4d68-bfff-9bdc96d0484b.txt` (`.dyn.jsonl` next to each).

**Noise floor (important).** The Tail-EMA is eval-only, so runs with the same `T_END` train the same weights and should give identical raw val loss. They don't:
- `T_END` 2750: raw val ranges 3.27968–3.28093 (spread 0.0013).
- `T_END` 2800: raw val ranges 3.27904–3.28060 (spread 0.0016).

So even with the same seed, the training itself varies by about ±0.0007 between runs. The varying part is probably non-deterministic GPU kernels and the order of NCCL reductions. Every gap in this experiment is that size or smaller.

**Conclusion: no detectable gain.**
- The best retuned setting is `T_END`=2800 with τ=75 (start 2400 or 2550). It gives 3.2787–3.2790, versus 3.27844 for the control, so it is no better than the control within noise.
- τ=75 helps on the 2800 schedule; a later start with τ=150 does not.
- None of these single runs can show a difference smaller than about 0.0015. To claim any gain, the candidate (2800/2400/τ75/λ0.4) and the control would each need roughly 4+ seeds.
- Until then, keep rec-46's schedule (`T_END`=2900, `TAILEMA_START`=2400, τ=150, λ=0.6).

## 2026-10-28 — Newton-Muon right preconditioning on top of rec-46

**Hypothesis.** Record #15 (Newton-Muon, PR #281, arXiv:2604.01472) right-preconditions each hidden matrix with the inverse input-activation covariance, refreshed every 64 steps. This gained over plain Muon, but it has never been stacked on the SOAP-Muon SOTA.

**Setup.**
- Code on `main`, based on rec-46. The math is in `OPTIMIZER.md` §2 and §2a.
- K = (Σ + 0.2·tr(Σ)/d·I)⁻¹, where Σ is a running average (weight 0.05) of the layer-input covariance x·xᵀ.
  - Σ is collected on refresh steps only, and K is refreshed every 64 steps.
  - q/k/v share one K; mlp.proj uses 4 diagonal blocks of 768×768.
  - All constants are #15's.
- Muon's learning rate is unchanged at 0.0375 for every matrix type. #15's per-type learning rates and weight decay were *not* ported.
- Seed 0, stop at 2690, W&B `nano-gpt` with optimization dynamics on.
- Launch: `NEWTON_MODE=<mode> WANDB_RUN_NAME=<name> sbatch --job-name=<name> run_record.sbatch 1d5e250`

| Knob (env) | Values |
|---|---|
| `NEWTON_MUON` | 1 (on, default on `main`) / 0 (= rec-46) |
| `NEWTON_MODE` | `soap`: U ← SOAP(U)·K before the trust gate and Muon, rescaled to ‖SOAP(U)‖_S4 · `grad`: g ← g·K before momentum, as in #15; momentum, SOAP statistics and the trust gate all see g·K |

**Bug found and fixed.** The first run (`newton-muon`, run `6kghexwj`, commit `7232189`) collected statistics with forward hooks. Hooks registered after `model.compile` never fire: the refresh diagnostics showed count 0. So K stayed a scaled identity and that run was effectively rec-46 (3.27853 vs 3.27848). That run is **invalid**. Commit `1d5e250` collects the statistics in `Linear.forward` instead. Each refresh then sees 8 batches (one per GPU), K's relative distance from the identity is about 0.55 early and 0.27 late, and no Cholesky failed.

**Results** (step 2690):

| Run | Mode | raw val | val_ema_loss | Δ vs rec-46 | val @500 / 1000 / 1500 / 2000 | Slurm | W&B |
|---|---|---|---|---|---|---|---|
| rec-46 | – | 3.28265 | **3.27848** | – | 3.82844 / 3.63025 / 3.51395 / 3.39376 | 618363 | [b1ip7q9i](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/b1ip7q9i) |
| newton-muon-soap-v2 | soap | 3.31047 | 3.30709 | **+0.0286** | 3.86830 / 3.65258 / 3.53660 / 3.41582 | 628897 | [rcyjol6w](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/rcyjol6w) |
| newton-muon-grad-v2 | grad | 3.28982 | 3.28579 | **+0.0073** | 3.83443 / 3.63835 / 3.52259 / 3.40165 | 628898 | [oyb145r5](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/oyb145r5) |

**Conclusion: negative in both modes, and well beyond the ±0.0007 noise floor.**
- Both modes are behind rec-46 from step 500 onward, and the gap never closes.
- `soap` (K after SOAP) is much worse than `grad` (K before momentum, as in #15).
- Likely reason: SOAP already whitens the input side through R = E[gᵀg]. Multiplying by K afterwards whitens the input side a second time.
- In `grad` mode, SOAP's statistics are built on g·K, so SOAP partly undoes K. That fits the smaller loss.

**Possible follow-ups (not run):**
- Weaker K, i.e. a larger `NEWTON_DAMPING` (e.g. 1–5) so K → I.
- Use K *instead of* SOAP's right-side factor (left SOAP, right Newton).
- Port #15's per-type learning rate ratios in `grad` mode.

### Follow-up: weaker K, K replacing SOAP's input side, #15's per-type learning rates (4 runs)

**Setup.** Commit `fa37539` on `main`, seed 0, stop at 2690. All runs set `NEWTON_MUON=1` explicitly.
- `NEWTON_DAMPING=δ`: K = (Σ + δ·tr(Σ)/d·I)⁻¹. A larger δ means weaker preconditioning.
- `NEWTON_MODE=right`: SOAP on the output (row) side only, with K replacing SOAP's input-side whitening.
- `MUON_LR_MULT=0.8,1,1.2`: Muon η₀ for q/k/v / attn.proj / MLP = 0.030 / 0.0375 / 0.045, i.e. #15's ratios. The PowerCool constants are scaled by the same factors.

Launch: `NEWTON_MUON=1 NEWTON_MODE=<m> [NEWTON_DAMPING=δ] [MUON_LR_MULT=…] WANDB_RUN_NAME=<name> sbatch --job-name=<name> run_record.sbatch fa37539`

**Results** (val_ema_loss at step 2690; earlier columns are raw val):

| Run | Mode | δ | lr mult | @500 | @1000 | @1500 | @2000 | raw @2690 | **val_ema @2690** | Δ vs rec-46 | Slurm | W&B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rec-46 | – | – | – | 3.82844 | 3.63025 | 3.51395 | 3.39376 | 3.28265 | **3.27848** | – | 618363 | [b1ip7q9i](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/b1ip7q9i) |
| newton-grad-damp4 | grad | 4 | 1,1,1 | 3.82966 | 3.63213 | 3.51755 | 3.39619 | 3.28454 | 3.28041 | +0.0019 | 629019 | [cybyzovt](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/cybyzovt) |
| newton-grad-damp1 | grad | 1 | 1,1,1 | 3.82834 | 3.63404 | 3.51993 | 3.39824 | 3.28642 | 3.28231 | +0.0038 | 629018 | [evzum536](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/evzum536) |
| newton-muon-grad-v2 (earlier) | grad | 0.2 | 1,1,1 | 3.83443 | 3.63835 | 3.52259 | 3.40165 | 3.28982 | 3.28579 | +0.0073 | 628898 | [oyb145r5](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/oyb145r5) |
| newton-grad-lrmult | grad | 0.2 | 0.8,1,1.2 | 3.84499 | 3.65106 | 3.53347 | 3.41061 | 3.29401 | 3.28946 | +0.0110 | 629021 | [mkn476lo](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/mkn476lo) |
| newton-muon-soap-v2 (earlier) | soap | 0.2 | 1,1,1 | 3.86830 | 3.65258 | 3.53660 | 3.41582 | 3.31047 | 3.30709 | +0.0286 | 628897 | [rcyjol6w](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/rcyjol6w) |
| newton-right | right | 0.2 | 1,1,1 | 3.85714 | 3.65387 | 3.54216 | 3.42341 | 3.31832 | 3.31477 | +0.0363 | 629020 | [9ob2e6cs](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/9ob2e6cs) |

Wall-clock note: `right` mode runs at 251 ms/step versus about 322 ms/step for the others, because it skips the R/Q_R eigen-updates. Track 3 counts steps, not time, so this doesn't matter for the leaderboard.

**Conclusion: Newton-Muon does not help on top of SOAP-Muon in any form tried.**
- **Weaker K:** the loss gets monotonically closer to rec-46 as K gets weaker: +0.0073 at δ=0.2, +0.0038 at δ=1, +0.0019 at δ=4. Its limit is the identity, i.e. rec-46, so there is no sweet spot.
- **K replacing SOAP's input side:** the worst result (+0.036). SOAP's gradient-based input-side whitening is clearly better than the activation-covariance K for this setup.
- **#15's per-type learning rates:** they hurt (+0.011 vs +0.0073 without them). #15 tuned those ratios for plain Muon, and they don't transfer to SOAP + RowFloor + the radius pin.
- **Recommendation:** default `NEWTON_MUON=0` on `main`, keeping the code for ablations, and drop this direction.

## 2026-10-28 — 1-D Adam (norm gains / biases) knobs aimed at the bounce

**Hypothesis.** Norm gains bounce the most (rec-46 mid-training ρ = −0.58, align_b = −0.24). #44's biggest win raised β₂ for biases but left gains at 0.99. Lowering the gains' effective step should move gain ρ toward −0.3 and help.

**Setup.** Commit `04d2cf8` on `main`, `NEWTON_MUON=0` (= rec-46), seed 0, stop 2690. Knobs: `GAIN_BETA2`, `AUX_BETA1` (gains and biases), `AUX_LR_MULT` (scales η₀ and the PowerCool constant). Bounce stats from `analyze_dynamics.py`, median over log steps 600–1900.

| Run | Change | val @2000 | raw @2690 | **val_ema @2690** | Δ vs control | norm ρ | norm align_b | norm upd_rel | bias ρ | Slurm | W&B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rec-46 (earlier) | – | 3.39376 | 3.28265 | 3.27848 | −0.0013 | −0.581 | −0.236 | 0.0019 | −0.551 | 618363 | [b1ip7q9i](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/b1ip7q9i) |
| aux-control | – | 3.39517 | 3.28405 | **3.27980** | – | −0.591 | −0.239 | 0.0020 | −0.571 | 631717 | [stsjiyx6](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/stsjiyx6) |
| aux-gain-b2-0997 | gain β₂ 0.99 → 0.997 | 3.39384 | 3.28257 | 3.27847 | −0.0013 | −0.597 | −0.223 | 0.0019 | −0.567 | 631718 | [bpvo20uq](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/bpvo20uq) |
| aux-b1-09 | β₁ 0.8 → 0.9 (gains + biases) | 3.39542 | 3.28400 | 3.27983 | +0.0000 | −0.652 | −0.235 | 0.0013 | −0.632 | 631719 | [a88cqlyb](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/a88cqlyb) |
| aux-lr-07 | gain/bias LR × 0.7 | 3.39541 | 3.28386 | 3.27970 | −0.0001 | −0.592 | −0.227 | 0.0015 | −0.560 | 631720 | [clwuj293](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/clwuj293) |

**Conclusion: no knob moved the gains' bounce, and none moved the loss outside noise.**
- Gain β₂ = 0.997 is the only candidate (−0.0013 vs its same-commit control, ≈ rec-46). That is about 2× the single-run noise floor, so it needs seeds before it counts.
- ρ and align_b for norm gains did not respond to any knob, even LR × 0.7, which cut the gains' upd_rel by about 20%. In a quadratic, ρ − 1 scales with the step, so a 20% shorter step should have taken ρ from −0.59 to about −0.26. It stayed at −0.59.
- So the gains' bounce is not a step-size overshoot along a fixed curvature. It is LR-invariant, as expected for a sign-like (Adam-normalized) update jittering at the valley floor: the jitter amplitude scales with η, so the gradient it produces scales with η, and the ratio does not change. For Adam groups ρ is therefore not a step-size gauge; it still is for the Muon matrices, where the update floor fixes |dW|.
- Run-to-run noise in ρ (control vs rec-46, same config): about ±0.01 for norm and bias, ±0.03 for attn_v.

### Follow-up: gain β₂ 0.997 on seeds 1–2, and V/O step size (6 runs)

Commit `5bfcc02` on `main`, `NEWTON_MUON=0`, stop 2690. New knobs: `VO_LR_MULT` (Muon lr × for attn.v and attn.proj) and `VO_FLOOR_MULT` (their RowFloor target ×). Seeds 1–2 have same-seed controls.

**Gain β₂ 0.997, paired by seed** (val_ema @2690):

| Seed | control | gain β₂ 0.997 | Δ | norm ρ ctrl → β₂ | W&B (ctrl / β₂) |
|---|---|---|---|---|---|
| 0 | 3.27980 | 3.27847 | −0.00133 | −0.591 → −0.597 | [stsjiyx6](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/stsjiyx6) / [bpvo20uq](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/bpvo20uq) |
| 1 | 3.27942 | 3.27924 | −0.00018 | −0.593 → −0.529 | [98ygxss6](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/98ygxss6) / [73kr23op](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/73kr23op) |
| 2 | 3.27863 | 3.27720 | −0.00143 | −0.601 → −0.568 | [ownqz6hm](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/ownqz6hm) / [sv3iurme](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/sv3iurme) |
| mean | 3.27928 | 3.27830 | **−0.00098 ± 0.00040 (SE)** | −0.595 → −0.565 | |

All three seeds favour β₂ 0.997 (paired t ≈ 2.4, df 2, p ≈ 0.13). It is a small, probable gain, not yet a certain one. Norm ρ moves by only about +0.03, nowhere near −0.3.

**V/O step size** (seed 0, vs aux-control 3.27980):

| Run | val @1000 | val @2000 | **val_ema @2690** | Δ | V/O upd_rel | ρ attn_v | ρ attn_o | dir_sharpness v / o | Slurm | W&B |
|---|---|---|---|---|---|---|---|---|---|---|
| aux-control | 3.62924 | 3.39517 | 3.27980 | – | 0.0095 | −0.302 | −0.247 | 325 / 556 | 631717 | [stsjiyx6](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/stsjiyx6) |
| vo-lr-08 | 3.62238 | 3.39137 | 3.28016 | +0.0004 | 0.0077 | −0.289 | −0.251 | 907 / 1450 | 631828 | [gpuihm9r](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/gpuihm9r) |
| vo-floor-08 | 3.62127 | 3.39135 | 3.27938 | −0.0004 | 0.0078 | −0.286 | −0.243 | 641 / 1010 | 631829 | [u7rhny2r](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/u7rhny2r) |

- Both cut the V/O step by about 19% and both run ahead mid-training (−0.007 at step 1000, −0.004 at 2000), but the lead is gone by 2690.
- ρ for V/O did not move toward q/k (−0.06 to −0.09). In a quadratic, 19% shorter steps should have taken ρ to about −0.05 (v) and −0.01 (o).
- Instead the measured sharpness along the step rose to compensate: dir_sharpness·|v|² stays constant (attn_v 6.7k / 6.7k / 6.4k; attn_o 8.6k / 8.1k / 8.1k), so g_a·v, g_b·v and therefore ρ and align_b do not change.

**Conclusion.** ρ and align_b are invariant to step size, for both the Adam groups (aux-lr-07) and the Muon V/O matrices. The local curvature along the update adapts to the step, consistent with self-stabilization / edge-of-stability behaviour. So ρ is not a step-size gauge and "step_ratio = 1/(1−ρ)" is not actionable. What did move ρ across records (#44: q/k −0.18 → −0.07) was a *direction* change (SOAP), so ρ is better read as a direction-quality signal.

### Follow-up: gain β₂ 0.997 on seeds 3–4 (4 runs)

Commit `3a47570`, same setup. Seeds 3/4 control: [3gpqdh4y](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/3gpqdh4y) / [1th1x875](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/1th1x875); β₂ 0.997: [wcmiyzvt](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/wcmiyzvt) / [ibp9obnx](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/ibp9obnx).

| Seed | control | gain β₂ 0.997 | Δ |
|---|---|---|---|
| 0 | 3.27980 | 3.27847 | −0.00133 |
| 1 | 3.27942 | 3.27924 | −0.00018 |
| 2 | 3.27863 | 3.27720 | −0.00143 |
| 3 | 3.27698 | 3.27964 | +0.00266 |
| 4 | 3.27937 | 3.28027 | +0.00090 |
| mean | 3.27884 | 3.27896 | **+0.00012 ± 0.00076 (SE)** |

**Conclusion: no effect.** The 3-seed "probable gain" was noise. The controls alone span 3.27698–3.27980 (seed-to-seed SD ≈ 0.0011), larger than the ±0.0007 same-seed noise floor assumed so far; a paired single-seed difference has SD ≈ 0.0017, so effects below about 0.002 need 5+ seeds.

## 2026-10-28 — V/O SOAP direction variants: probe screen

**Setup.** Commit `fc7b5ad`, `VO_VARIANTS=1` (measured only, training unchanged), 1000 steps, probe every 100, run [v1ayjw5i](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/v1ayjw5i) (Slurm 632113). Each variant goes through the real attn.proj gate and Newton–Schulz; D = a²/2c on the fixed held-out probe batch. "V∪O" is the global group of the V/O-only directions, i.e. including the V–O cross curvature; it is the ranking that matters.

**D relative to the real direction (vo_ns):**

| Variant | V∪O, steps 100–400 | V∪O, steps 500–900 | attn_v alone, 500–900 | attn_o alone, 500–900 |
|---|---|---|---|---|
| vo_head (per-head statistics) | 1.00–1.02 | 1.05–1.07 | 1.8–2.8 | 1.01–1.02 |
| vo_shared (one V–O basis) | 0.97–1.01 | 0.95–1.01 | 0.94–1.11 | 1.01–1.02 |
| vo_p25 (V^¼ denominator) | 0.96–0.99 | 1.02–1.12 | 2.9–5.3 | 0.99–1.03 |
| vo_gate (trust gate on V) | 1.05–1.12 | 1.01–1.22 | 2.9–5.2 | 1 (O unchanged) |

- On attn_v the real SOAP direction is almost orthogonal to the probe gradient: its first-order gain a is about 10× smaller than with the gate or V^¼ (e.g. step 900: 0.044 vs 0.45 / 0.35), although those directions have cos 0.90–0.94 with it. Full whitening removes most of V's gradient-aligned component.
- The greedy-optimal step sizes (opt_rel 0.003–0.03) are at or below the probed ±0.02, so these D values are interpolated, not extrapolated.
- Taken alone, V gains 2–5×; with the V–O cross curvature counted, the gains shrink to 5–12%.
- vo_shared is dropped (≤ 1). vo_gate, vo_p25 and vo_head go to full runs (`V_GATE`, `VO_DENOM_POWER=0.25`, `VO_HEAD`, commit `257e962`), seeds 0 and 1, paired with the existing same-seed controls.

### V/O variants: full runs (6 runs)

Commit `257e962`, `NEWTON_MUON=0`, stop 2690, paired with the same-seed controls (s0 3.27980, s1 3.27942).

| Variant (env) | Seed | val @1000 | val @2000 | **val_ema @2690** | Δ vs control | ρ attn_v | ρ attn_o | W&B |
|---|---|---|---|---|---|---|---|---|
| control | 0 / 1 | 3.62924 / 3.62781 | 3.39517 / 3.39488 | 3.27980 / 3.27942 | – | −0.302 / −0.290 | −0.247 / −0.236 | [stsjiyx6](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/stsjiyx6) / [98ygxss6](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/98ygxss6) |
| `V_GATE=1` | 0 / 1 | 3.63120 / 3.62909 | 3.39474 / 3.39357 | 3.27783 / 3.27717 | −0.00197 / −0.00225 | −0.396 / −0.363 | −0.277 / −0.267 | [p0zgwh4z](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/p0zgwh4z) / [8c0b2qih](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/8c0b2qih) |
| `VO_HEAD=1` | 0 / 1 | 3.62838 / 3.62694 | 3.39390 / 3.39378 | 3.27849 / 3.27846 | −0.00131 / −0.00096 | −0.303 / −0.315 | −0.266 / −0.268 | [ebxv80f6](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/ebxv80f6) / [0d74vmza](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/0d74vmza) |
| `VO_DENOM_POWER=0.25` | 0 / 1 | 3.63161 / 3.63291 | 3.39547 / 3.39580 | 3.27887 / 3.27894 | −0.00093 / −0.00048 | −0.388 / −0.396 | −0.269 / −0.280 | [h3la4wa7](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/h3la4wa7) / [0ood5853](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/0ood5853) |

- All six runs are below their controls. The ranking matches the probe screen's V∪O ranking (gate > head ≈ p25).
- `V_GATE`: mean −0.0021 ± 0.0012 (SE from the paired SD 0.0017), about 1.8σ. Both runs are below 4 of the 5 controls. Promising; needs seeds 2–4.
- `VO_HEAD`: mean −0.0011; `VO_DENOM_POWER=0.25`: mean −0.0007. Both inside the noise with 2 seeds.
- ρ for attn_v got *more* negative with the gate and with V^¼ (−0.30 → −0.38) while the loss improved, and gate/V^¼ are behind at step 1000 and ahead by 2690. More bounce, better result: further evidence that ρ is not a target to minimize. The probe D picked the right ordering; ρ would have picked the wrong one.

### V_GATE on seeds 2–4, and V_GATE + VO_HEAD (5 runs)

Commit `257e962`, paired with the same-seed controls.

| Seed | control | `V_GATE=1` | Δ | `V_GATE=1 VO_HEAD=1` | Δ |
|---|---|---|---|---|---|
| 0 | 3.27980 | 3.27783 | −0.00197 | 3.27976 ([cfti692z](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/cfti692z)) | −0.00004 |
| 1 | 3.27942 | 3.27717 | −0.00225 | 3.27844 ([2uy7yhs1](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/2uy7yhs1)) | −0.00098 |
| 2 | 3.27863 | 3.27794 ([4fhoxffa](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/4fhoxffa)) | −0.00069 | | |
| 3 | 3.27698 | 3.27744 ([oyranqda](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/oyranqda)) | +0.00046 | | |
| 4 | 3.27937 | 3.27951 ([mm322vc9](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/mm322vc9)) | +0.00014 | | |
| mean | 3.27884 | 3.27798 | **−0.00086 ± 0.00054 (SE)** | | −0.00051 (2 seeds) |

**Conclusion.**
- `V_GATE`: the 2-seed −0.0021 shrank to −0.0009 at 5 seeds (paired t ≈ 1.6, p ≈ 0.19). Possibly a small real gain, not established. Same pattern as gain β₂: the first seeds overstated it.
- `V_GATE + VO_HEAD` is worse than `V_GATE` alone on the same seeds (−0.0005 vs −0.0021): the two do not add.
- Defaults stay at rec-46.

## 2026-10-29 — Fork protocol: checkpoints + same-checkpoint control forks

**Why.** Seed-to-seed SD of val@2690 is ≈ 0.0010, while the gains left in this stack are 15–40 steps (0.0007–0.0018). Detecting 0.0009 at 80% power needs ≈ 20 seeds per arm, so every 2–5 seed V/O result above is unresolved, not refuted. Most of a seed's luck is fixed early (corr(val@2000, val@2690) = 0.98 across 13 control runs).

**Setup** (commit `3bc552a` on `main`).
- `CKPT_STEPS=1000,1500,2000` saves the full training state at the top of those steps: weights, Muon momentum, SOAP statistics/bases/second moments, AdamW and Adam moments, EMA-Nesterov buffers and counter, Tail-EMA buffer, Muon step count (trust-gate schedule), data position, GNS held-out position, RNG. Rank 0 writes the replicated state; each rank writes its Muon shard, keyed by parameter name. 4.9 GB per checkpoint, under `/work/$USER/nanogpt-ckpt/<run>-seed<s>/step<k>/`.
- `RESUME_FROM=<dir>` starts at step k; env knobs given to the fork apply from step k on. Not covered: Newton-Muon state (asserted off) and the `VO_VARIANTS` shadow states.
- `DENSE_VAL=2650:2690:5` (default) adds val every 5 steps near the end.
- Parents: `ckpt-rec46`, rec-46 settings, seeds 0–2 (val_ema 3.27876 / 3.27949 / 3.27771; first step ≤ 3.28: 2675 / 2685 / 2655).
- Scoring: `fork_compare.py` compares each fork with the parent and the control forks of the same checkpoint; steps = −Δ / (controls' val_ema slope over 2650–2690 ≈ 0.000067/step).

**Restore check.** Every fork reproduces the parent's val at its fork step to the last digit (smoke test at 125; 1000/1500/2000 in the real runs).

**Fork noise** (identical control forks, val_ema@2690, parent + 2 forks per checkpoint, seeds 0–1):

| Fork step | SD across replicas |
|---|---|
| 1000 | 0.000075 |
| 1500 | 0.000042 |
| 2000 | 0.000012 |
| (independent seeds) | ≈ 0.0010 |

A variant–control fork difference has SD ≈ 0.0001 (≈ 1.5 steps) from step 1000, so one fork per seed resolves a 0.0009 effect at ≈ 9σ. Three seeds per arm test whether an effect is consistent across seeds, not noise. Caveat: a fork measures only the gain over the segment after the fork, plus any cost of switching mid-run.

### Fork results, round 1 (seeds 0–2, Δ val_ema@2690 vs same-checkpoint controls; positive = worse)

| Variant (env) | Fork at | Δ mean ± SE | Steps | Verdict |
|---|---|---|---|---|
| `V_GATE=1` | 1000 | +0.00086 ± 0.00007 | −12.9 | worse, reject |
| `V_GATE=1` | 1500 | +0.00066 ± 0.00008 | −9.8 | worse, reject |
| `VO_DENOM_POWER=0.25` | 1000 | +0.00073 ± 0.00004 | −10.9 | worse, reject |
| `VO_HEAD=1` | 1000 | +0.00028 ± 0.00002 | −4.2 | worse, reject |
| `GAIN_BETA2=0.997` | 1000 | +0.00000 ± 0.00012 | 0 | no effect, reject |
| `ROW_PIN=1` (per-row radius pin) | 1000 | +0.00001 ± 0.00007 | −0.2 | no effect, reject |
| `ROW_GAIN_LR=0.001` (Muown row gains) | 1000 | +0.00107 ± 0.00002 | −15.9 | worse |
| `ROW_GAIN_LR=0.003` | 1000 | +0.00045 ± 0.00013 | −6.7 | worse; improving with lr, larger lr queued |

- The V/O variants that looked like −0.001 to −0.002 in the 2–5 seed full runs are clearly worse after step 1000. The full-run "gains" were seed noise (or come only from steps < 1000, which the forks cannot see; the full-run evidence for that was p ≈ 0.19).
- The probe screen's D ranking (gate > head ≈ V^¼ > baseline on V∪O) did not predict the fork outcome. One-step greedy D on a held-out batch rewards directions that descend faster now; the fork measures the whole remaining trajectory.

### Fork results, round 2 (seeds 0–2)

| Variant (env) | Fork at | Δ mean ± SE | Steps | Verdict |
|---|---|---|---|---|
| `SOAP_SINK=cur` (SinkSOAP balancing of current P²) | 1000 | +0.00159 ± 0.00005 | −23.7 | much worse, reject |
| `SOAP_SINK=ema` (Sinkhorn-factored V) | 1000 | +0.00150 ± 0.00002 | −22.5 | much worse, reject |
| `SOAP_KL=1` (KL-SOAP coupled covariance) | 1000 | +0.00010 ± 0.00007 | −1.5 | no gain, reject |
| `ROW_GAIN_LR=0.01` | 1000 | **−0.00047 ± 0.00009** | **+7.0** | better on all 3 seeds → sweep |
| `ROW_GAIN_LR=0.03` | 1000 | −0.00003 ± 0.00006 | +0.4 | no effect |
| `ENA_REST=2300` (EMA-Nesterov on longer) | 1500 | +0.00023 ± 0.00001 | −3.5 | worse, reject |
| `ENA_REST=2690` | 1500 | +0.00044 ± 0.00007 | −6.5 | worse, reject |

- Row gains (Muown #23 port: angular-only Muon step per row + Adam on row norms) have an optimum: lr 0.001 −16 steps, 0.003 −7, **0.01 +7**, 0.03 0.
- Outer acceleration after 1950 hurts: EMA-Nesterov's lookahead is switched off at the right time. MuLoCo / SODA (same slot) not ported.

### Readout lab (fork from 2000, seeds 0–2; Δ vs rec-46 Tail-EMA readout, val = selection set, held-out = 10.5M tokens of the last train shard)

| Readout | Δ val | Δ held-out | Steps |
|---|---|---|---|
| EMA start 2200, τ 150, λ 0.6 | **−0.00041 ± 0.00001** | **−0.00039** | **+6.2** |
| EMA 2200/150, λ 0.7 | −0.00039 | −0.00036 | +5.8 |
| Tail-EMA + 5% toward θ₂₂₀₀ (#42 style, γ = −0.05) | −0.00029 | −0.00021 | +4.3 |
| Tail-EMA + #38 velocity readout (any γ, β) | ≥ +0.00003 | – | worse (removed from the lab) |
| Tail-EMA + RRE (#34), capped | +0.00014 | +0.00014 | −2.1 |
| any reference *extrapolation* (γ > 0) | ≥ +0.00001 | – | worse |

- #42's γ = −0.075 is an interpolation toward the reference (θ_eval = 0.925θ + 0.075θ_ref), not an extrapolation; extrapolating away from any reference was worse in every setting.
- The gain comes from a longer averaging window: starting the Tail-EMA at 2200 instead of 2400 (same τ, λ). Confirmed on held-out data (−0.00039 vs −0.00041 on val), so it is not selection on the val set.
- Refinement (start 2100–2300 × τ 100–200) running as `lab3`.

### Round 3: row-gain lr sweep, readout refinement, first mechanism statistics

**Row-gain lr** (`ROW_GAIN_LR`, fork at 1000, seeds 0–2; steps vs same-checkpoint controls):

| lr | 0.001 | 0.003 | 0.005 | 0.007 | 0.01 | **0.015** | 0.02 | 0.03 |
|---|---|---|---|---|---|---|---|---|
| Δ val_ema | +0.00107 | +0.00045 | −0.00004 | −0.00020 | −0.00047 | **−0.00065 ± 0.00005** | −0.00051 | −0.00003 |
| steps | −15.9 | −6.7 | +0.6 | +3.0 | +7.0 | **+9.7** | +7.7 | +0.4 |

- A smooth optimum at 0.015, consistent on every seed. lr 0.01 forked at 1500 gives only +2.6 (vs +7.0 from 1000): the gain accumulates over training, so a full run (on from step 0) may gain more.

**Readout refinement** (lab3, fork from 2000, seeds 0–2; vs Tail-EMA 2400/150/0.6):

| EMA start : τ, λ | Δ val | Δ held-out | steps |
|---|---|---|---|
| **2100 : 200, 0.6** | **−0.00056** | **−0.00045** | **+8.3** |
| 2100 : 150, 0.6 | −0.00044 | −0.00041 | +6.6 |
| 2200 : 200, 0.6 | −0.00046 | −0.00035 | +6.8 |
| 2200 : 150, 0.6 | −0.00039 | −0.00037 | +5.9 |
| 2300 : 150, 0.6 | −0.00028 | −0.00028 | +4.2 |

- Flat optimum around an earlier start (2100) and slightly longer τ (150–200), λ 0.6 throughout. Chosen: 2100 / 200 / 0.6 (best on both sets). Now env knobs `TAILEMA_START`, `TAILEMA_TAU`, `TAILEMA_LAMBDA` (defaults = rec-46).

**Row-norm statistics** (`<log>.rowstat.jsonl`, control fork vs `ROW_GAIN_LR=0.01` fork, seed 0, at 2690):

| type | mean row norm ctrl → gain | CV of row norms ctrl → gain |
|---|---|---|
| mlp.fc | 6.88 → 5.20 | 0.187 → 0.191 |
| mlp.proj | 7.40 → 10.63 | 0.182 → 0.348 |
| attn.q | 4.28 → 3.91 | 0.173 → 0.204 |
| attn.k | 4.37 → 4.20 | 0.251 → 0.334 |
| attn.v | 5.90 → 5.23 | 0.410 → 0.479 |
| attn.proj | 4.93 → 4.56 | 0.271 → 0.342 |

Two effects: (1) rows of a matrix spread out in norm (CV up, most in mlp.proj), (2) scale moves between matrices (mlp.fc shrinks, mlp.proj grows). Ablations to separate them (lr 0.015, fork at 1000): row gains on mlp only / attention only / mlp.proj only / mlp.fc only, and one gain per matrix (`ROW_GAIN_MODE=matrix`, effect 2 without effect 1).

**Full runs** (commit `243e148`, seeds 0–2): `ROW_GAIN_LR=0.015` + Tail-EMA 2100/200/0.6, with the row gains on from step 0 (`full-rg15-s0start`, also saving checkpoints) and from step 1000 (`full-rg15-1000start`).

### Round 4: what row gains act through

**Ablations** (lr 0.015, fork at 1000, seeds 0–2, commit `243e148`):

| Row gains on | Δ val_ema | steps |
|---|---|---|
| all six types (reference, same commit) | −0.00050 ± 0.00006 | +7.4 |
| attention only (q, k, v, o) | −0.00046 ± 0.00005 | +6.9 |
| mlp.fc only | −0.00045 ± 0.00006 | +6.7 |
| mlp (fc + proj) | −0.00017 ± 0.00004 | +2.5 |
| mlp.proj only | +0.00025 ± 0.00001 | −3.7 |
| one gain per matrix (`ROW_GAIN_MODE=matrix`), all types | +0.00214 ± 0.00016 | −31.9 |

- The benefit is per-row freedom, not moving scale between matrices: a single gain per matrix is far worse. (Caveat: the matrix-level gain reuses the per-row lr 0.015, which is large for one scalar per matrix.)
- Row gains help on attention and mlp.fc and hurt on mlp.proj — the type whose row norms grew most (7.4 → 10.6) and spread most (CV 0.18 → 0.35). Next: all types except mlp.proj.
- The same configuration (all types, lr 0.015) gave −0.00065 in round 3 and −0.00050 here (code identical in function). That 0.00015 gap is about 2 SE; read single-config differences under ~0.0002 with care.

**Full runs, row gains switched on at step 1000** (`full-rg15-1000start`, + Tail-EMA 2100/200/0.6, seeds 0–2):

| seed | rec-46 (`ckpt-rec46`) val_ema@2690 / first step ≤ 3.28 | row gains + new Tail-EMA |
|---|---|---|
| 0 | 3.27876 / 2675 | 3.27782 / 2655 |
| 1 | 3.27949 / 2685 | 3.27733 / 2650 |
| 2 | 3.27771 / 2655 | 3.27745 / 2650 |
| mean | 3.27865 | **3.27753 (−0.00112)** |

**Row gains from step 0 diverge** (loss 5.86 at step 125 vs 4.44, NaN at 440): attn.proj and mlp.proj are zero-initialized, and with the angular-only step a zero row cannot grow. Row gains need a start step after the projections have grown (`ROW_GAIN_START`).

### Round 5: scope and lr without mlp.proj; first mechanism test

| Row gains on (fork at 1000) | lr | Δ val_ema | steps |
|---|---|---|---|
| q, k, v, o, fc (no mlp.proj) | 0.01 | −0.00047 ± 0.00006 | +7.0 |
| q, k, v, o, fc | 0.015 | −0.00070 ± 0.00008 | +10.5 |
| q, k, v, o, fc | **0.02** | **−0.00081 ± 0.00010** | **+12.1** |
| q, k only | 0.015 | −0.00005 ± 0.00007 | +0.7 |
| v, o only | 0.015 | −0.00032 ± 0.00002 | +4.8 |

- Inside attention the gain is in V/O, not Q/K. Contributions: v/o ≈ +5, fc ≈ +7, q/k ≈ 0, mlp.proj < 0.
- The optimal lr moves up once mlp.proj is excluded; 0.025 and 0.03 queued.

**Full runs** (Tail-EMA 2100/200/0.6 in all; seeds 0 / 1 / 2; val_ema@2690 and first step ≤ 3.28):

| Run | seed 0 | seed 1 | seed 2 | mean | Δ vs rec-46 |
|---|---|---|---|---|---|
| rec-46 (`ckpt-rec46`) | 3.27876 / 2675 | 3.27949 / 2685 | 3.27771 / 2655 | 3.27865 | – |
| row gains all types, on at 1000 | 3.27782 / 2655 | 3.27733 / 2650 | 3.27745 / 2650 | 3.27753 | −0.00112 |
| no mlp.proj, on at 500 | 3.27730 / 2650 | 3.27809 / 2660 | 3.27799 / 2655 | 3.27779 | −0.00086 |
| no mlp.proj, on at 1000 | 3.27731 / 2650 | 3.27845 / 2665 | 3.27757 / 2650 | 3.27778 | −0.00087 |

All nine full runs are below the mean of rec-46 on the same seeds; with the seed SD ≈ 0.001 the differences between these three variants are not resolved by 3 seeds.

**Mechanism test 1: is there a persistent radial pull the pin suppresses?** `track_radial` logs, per output row, the radial gradient g_i = ⟨∇W_i, W_i/‖W_i‖⟩ and its signal-to-noise |E g| / std g over an EMA of ≈ 50 steps. In control and row-gain forks alike (seeds 0–1), the median SNR is 0.02–0.09 for every type, and no row has SNR > 1. Per-step radial gradients are noise-dominated everywhere, and row gains do not change that. So "a strong persistent pull held back by the pin" is not what distinguishes the types. Next: the whole-run t-statistic Σg / √Σg² per row (pure noise gives a median ≈ 0.67) and its correlation with each row's norm change, to see whether a weak long-horizon drift exists and whether row gains follow it.

### Round 6: lr plateau, readout on the row-gain trajectory, mechanism

**lr without mlp.proj** (fork at 1000): 0.015 +10.5, **0.02 +12.1**, 0.025 +11.4, 0.03 +11.4 steps; v, o, fc only at 0.02: +10.7. Plateau 0.02–0.03; q/k add ≈ 0. Chosen: `ROW_GAIN_LR=0.02 ROW_GAIN_SCOPE=q,k,v,o,fc`.

**Readout on the row-gain trajectory** (lab4: forks from step 2000 of `full-noproj-start1000`, which already uses Tail-EMA 2100/200/0.6): Tail-EMA 2100/200/0.6 is still the optimum (ema 2000/200 ties at −0.00001). rec-46's Tail-EMA 2400/150/0.6 on this trajectory is +0.00055 val / +0.00045 held-out worse (−9 steps). The readout change and the row gains stack.

**Mechanism.** What the data supports:
1. *Per-row, not per-matrix.* One gain per matrix is −32 steps; per-row gains are +12. Moving scale between matrices is not the effect.
2. *Only where row scale is visible to the function.* The model applies RMSNorm to each q and k head (`q, k = norm(q), norm(k)`), so scaling all rows of a head is invisible and only within-head ratios matter. Row gains on q/k: +0.7 steps. On v (value coordinates), attn.proj (attention-output coordinates of the residual stream) and mlp.fc (pre-activation scale of each relu² unit, entering squared) the row norm sets an output scale directly: +5 (v, o) and +7 (fc).
3. *Rows sit closer to their norm equilibrium.* Whole-run radial-gradient t-statistic per row, |Σg_i| / √Σg_i² from the fork at 1000 to 2690 (pure noise: median 0.67), control → row gains (lr 0.02, no mlp.proj), seeds 0–1 agree:

| type | median \|t\| ctrl → gain | rows with \|t\| > 3 ctrl → gain | corr(t_i, Δlog‖W_i‖) ctrl → gain |
|---|---|---|---|
| attn.q | 0.69 → 0.45 | 7% → 1–2% | −0.60 → −0.61 |
| attn.k | 0.91 → 0.44 | 9% → 1% | −0.62 → −0.61 |
| attn.v | 0.61 → 0.42 | 1% → 0% | −0.62 → −0.36 |
| attn.proj | 0.43 → 0.30 | 0–1% → 0% | −0.64 → −0.20 |
| mlp.fc | 0.32 → 0.24 | 0% | −0.31 → −0.10 |
| mlp.proj (gains off in both) | 0.90 → 0.84 | 3% | −0.09 → −0.15 |

   - In the control, v / attn.proj / mlp.fc have |t| *below* the noise level: their radial gradients are mean-reverting (rows oscillate around a norm equilibrium). q/k rows have a persistent drift (7–9% of rows at |t| > 3).
   - Row gains lower |t| in every type they act on: each row's radial gradient cancels more completely over the run, i.e. each row's norm stays nearer its own equilibrium. In rec-46 the per-row norms can only move through the whole-matrix radius pin, radial damping (outward radial motion halved) and the RowFloor, so individual rows lag their equilibrium.
   - The correlation between a row's accumulated radial gradient and its norm change weakens for v / o / fc (rows no longer need large drifts to cancel a pull, because the pull never builds up), and stays for q/k.
   - Consistent picture: row gains equilibrate every row's norm quickly; that matters where the row norm is visible to the loss (v, o, fc), not behind QK-norm (q, k). Not explained: why mlp.proj, whose row norm is also visible, is hurt (its rows grew 7.4 → 10.6 and spread CV 0.18 → 0.35 with gains on; one guess is that it lets MLP outputs dominate the residual stream).
   - Per-step SNR (EMA ≈ 50 steps) is 0.02–0.09 everywhere, so the equilibrium is only visible over hundreds of steps; this rules out "a strong persistent pull held back by the pin".

**Candidate for full-run validation** (commit `b7ae8c3`): `ROW_GAIN_LR=0.02 ROW_GAIN_SCOPE=q,k,v,o,fc ROW_GAIN_START=1000 TAILEMA_START=2100 TAILEMA_TAU=200` (λ 0.6), seeds 0–7 (`cand-rg20`), plus rec-46 controls for seeds 5–7 (`ctrl-rec46`). Scored by `fork_compare.py --full`: Track 3 rule at each dense step, earliest passing step, and seed-paired difference from the rec-46 controls.

### Full-run validation of the candidate (8 seeds)

Commit `b7ae8c3`. Reproduce one seed:

```
NEWTON_MUON=0 ROW_GAIN_LR=0.02 ROW_GAIN_SCOPE=q,k,v,o,fc ROW_GAIN_START=1000 TAILEMA_START=2100 TAILEMA_TAU=200 \
WANDB_RUN_NAME=cand-rg20 sbatch --job-name=cand-rg20-s<seed> run_record.sbatch b7ae8c3 --seed <seed>
```

Track 3 rule (3.28 − mean)·√n ≥ 0.004, n = 8 (`fork_compare.py --full cand-rg20`):

| step | 2650 | 2655 | 2660 | **2665** | 2670 | 2680 | 2690 |
|---|---|---|---|---|---|---|---|
| mean val_ema | 3.27953 | 3.27919 | 3.27888 | **3.27858** | 3.27829 | 3.27771 | 3.27717 |
| (3.28 − mean)·√8 | 0.00132 | 0.00231 | 0.00315 | **0.00403** | 0.00484 | 0.00647 | 0.00802 |

- **Earliest passing step: 2665** (rec-46: 2690). Borderline (0.00403 vs 0.004); seeds 8–15 queued.
- Seed-paired vs rec-46 controls at 2690 (controls: `ckpt-rec46`, `aux-control`, `ctrl-rec46`; mean where a seed has two): −0.00103, −0.00197, −0.00029, −0.00024, −0.00114, −0.00195, −0.00240, −0.00333; **mean −0.00154 ± 0.00038 (SE)**, all 8 negative, ≈ 23 steps at the end-of-run slope.
- rec-46 controls at this commit, seeds 5–7, alone: mean 3.27898 (does not pass by 2690 with n = 3; for reference only).
- W&B: [s0](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/vd8eol60) [s1](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/xgq882m1) [s2](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/wm78t2y9) [s3](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/uw2c8g4r) [s4](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/vw9eh8gb) [s5](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/dstii5eg) [s6](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/aysiph0t) [s7](https://forge.coreweave.com/wandb/bendoris0509-national-yang-ming-chiao-tung-university/nano-gpt/runs/1epj5o1f) 

**16 seeds** (seeds 8–15 added): Track 3 value (3.28 − mean)·√16 = 0.00288 at 2655, 0.00407 at 2660, **0.00530 at 2665**, 0.01097 at 2690. Earliest passing step 2660 (by 0.00007; it failed with the first 8 seeds); **settled step: 2665**, which passes with both n = 8 (0.00403) and n = 16 (0.00530). Mean val_ema at 2665 = 3.27867 (n = 16).

**Leaderboard submission** (user-authorized 2026-10-29): seed 0 (fixed in advance), `STOP_STEP=2665`, commit `b7ae8c3`, candidate env above, `OPTIM_DYNAMICS=0`, W&B `cerulean-labs/midterm-nanogpt` run `rowgain-tailema2100-2665-s0` ([wcs41r46](https://flock.cerulean.works/cerulean-labs/midterm-nanogpt/runs/wcs41r46)), Slurm 633829. Summary: `total_train_steps` 2665, `val_loss` (val_ema) 3.27759, raw val 3.28348, peak memory 34.8 GiB, no dynamics keys.

**Defaults on `main` = this record** (2026-10-29): `NEWTON_MUON=0 ROW_GAIN_LR=0.02 ROW_GAIN_SCOPE=q,k,v,o,fc ROW_GAIN_START=1000 TAILEMA_START=2100 TAILEMA_TAU=200`, and `record.env` sets `STOP_STEP=2665`. Commands above that rely on "defaults = rec-46" now need `NEWTON_MUON=0 ROW_GAIN_LR=0 TAILEMA_START=2400 TAILEMA_TAU=150 STOP_STEP=2690` (or the `rec-46` tag).

## Anatomy of the row-gain effect (2026-10-29)

**Setup.** Commit `bccead1` (transplants `15a1118`). Forks from `ckpt-rec46` seeds 0–1 at step 1000 (`anat-Gon2000`: at 2000), all with `TAILEMA_START=2100 TAILEMA_TAU=200`, stop 2690, `ANATOMY=1 VAL_EVERY=50 SNAP_STEPS=1500,2000,2690`. Analysis: `python anatomy.py --control anat-C --variants anat-G,anat-G0,anat-Goff2000,anat-Gon2000,anat-Gproj --seeds 0,1` and `--transplant tp-CG` / `tp-CGproj` / `tp-CC2`.

| fork | setting | Δ val_ema@2690 vs C, s0 / s1 |
|---|---|---|
| `anat-C` | rec-46 (`ROW_GAIN_LR=0`) | 3.27817 / 3.27888 |
| `anat-G` | row gains q,k,v,o,fc from 1000 | −0.00057 / −0.00088 |
| `anat-G0` | `ROW_GAIN_FIXED=1`: direction-only step per row, lengths moved only by CWD | +0.00219 / +0.00201 |
| `anat-Goff2000` | gains 1000→2000, then the rec-46 pin | −0.00035 / −0.00077 |
| `anat-Gon2000` | rec-46 until 2000, gains from 2000 | −0.00017 / −0.00009 |
| `anat-Gproj` | gains also on mlp.proj | −0.00023 / −0.00049 |

**Findings** (means of seeds 0–1; per-type tables in the `anatomy.py` output):
1. *Not a per-row learning-rate effect.* RowFloor lifts 94–100 % of rows to u/w = 0.3825 in every run, and the angle a row turns per unit lr is 0.390 in every run and type (q/k/v/o/fc/proj, C/G/G0). Row gains do not change how far rows rotate.
2. *Lengths have to move, and row gains move them on their own gradient.* With lengths left to CWD alone (G0), matrices end 28–39 % smaller than in C (q,k,v,o,fc; mlp.proj unchanged), layer-0 attention/MLP output budgets drop by 25–46 %, and the loss is 30 steps worse. Row gains: per-step |Δlog r| 2–6× larger than C (fc 2×); length deviations from their 201-step trend decay faster (lifetime τ_d 38 → 24 steps over 1000–1500, 31–33 → 22–23 over 2000–2690 for q/k/v/o; G0: 45); corr(γ_t, d_t), the restoring push on a row longer than its trend, rises 2–5× (q/k/v/o); end state: q,k,v,o matrices 10–37 % larger than C, fc 5 % smaller, row-norm CV within a matrix 25–65 % larger.
3. *Autocorrelation of the radial gradient γ barely changes.* γ is anti-persistent in every run, including G0: lag-1 autocorrelation −0.4 to −0.6 (1000–1500) and −0.07 to −0.23 (2000–2690), variance ratio VR(200) 0.01–0.2. G lowers VR(50/200) by 0–40 % in q/k/v and |t| by 25–45 %. So the earlier |t| statistic mostly reflected this generic anti-persistence; the length dynamics (τ_d, corr(γ,d), |Δlog r|) are where the runs differ.
4. *Built during 1000–2000 and kept.* Gains switched off at 2000 keep ~77 % of the 2690 advantage; gains switched on at 2000 get ~18 %. The advantage appears around step 2150 in both seeds. Right after switching off (2050), Goff2000 is 0.0006–0.0007 better than G in both seeds, i.e. while active the gains also add some length jitter that costs loss at the moment (consistent with the longer Tail-EMA window helping on top).
5. *Not a portable length profile.* Two control forks from the same checkpoint already diverge to row cosines 0.82–0.89, and transplanting their row lengths costs only +0.002 (matrix norms: ~0). Transplanting between C and G costs +0.28 to +0.51 (row) and +0.14 to +0.21 (matrix norm only), in both directions: G's lengths (log-ratio SD 0.06–0.17 per row vs 0.02 between controls) are co-adapted with its own directions. The transplant cannot attribute the gain to the length configuration alone.
6. *Function.* G: layer-0 MLP output / residual +12–14 %, last-layer attention +27–58 %, last-layer MLP −12–16 %, fc units fire slightly more often (+3–6 % relative). Gproj: MLP outputs of layers 0 and 3–4 grow 25–50 % and mid-layer attention shrinks ~15 %: mlp.proj gains let the MLP crowd attention out of the residual stream, which fits Gproj being worse than G.
7. *Where the loss improves.* At positions ≥ 128 in the context (88 % / 90 % of the gain) and on the rarest 40 % of target tokens (52 % / 98 %). G0 loses at the same long positions.

**Reading.** Row gains help through the length dynamics, not the rotation: each row's length follows its own (anti-persistent, low-SNR) radial gradient through Adam, which lets q/k/v/o grow and spread across rows while length errors are damped faster. The gain accumulates in 1000–2000 into a different, co-adapted solution that uses long context better. Removing the radial update without a replacement (G0) shrinks the weights under CWD and costs 30 steps, so rec-46's radial part was doing real work.

**Metrics changed.** Removed the per-token "control-loss decile" split (regression to the mean; identical for every variant). Transplant results are printed as absolute loss changes, with a control-vs-control pair as the baseline.

### Frozen lengths without weight decay (2026-10-29)

Commit `97303c2`, `ROW_GAIN_NO_CWD=1` (no CWD on q,k,v,o,fc while their row gains are active), otherwise as the anatomy forks. Steps = −Δ / 0.000067.

| fork | setting | Δ val_ema@2690 vs C, s0 / s1 | ≈ steps | log(‖W‖/‖W_C‖) at 2690, q k v o fc |
|---|---|---|---|---|
| `anat-G` | row gains | −0.00057 / −0.00088 | +11 | +0.16 +0.31 +0.09 +0.10 −0.06 |
| `anat-Gnc` | row gains, no CWD | −0.00076 / −0.00103 | +13 | +0.50 +0.65 +0.41 +0.43 +0.28 |
| `anat-G0` | direction-only, CWD on | +0.00219 / +0.00201 | −31 | (−0.33 … −0.49) |
| `anat-G0nc` | direction-only, no CWD | +0.00117 / +0.00103 | −16 | +0.02 0.00 −0.09 −0.05 −0.14 |

- In G0nc the optimizer step leaves every row length unchanged (max |Δlog r| per step 2·10⁻⁷). Lengths still grow through the EMA-Nesterov lookahead until step 1950 (cumulative +0.20 in log, the same mechanism in every run), so G0nc ends with matrix norms within −14 … +2 % of C's.
- *Learning the lengths is worth ~30 steps:* Gnc vs G0nc differ only in the Adam length step (Δ −0.00193 / −0.00206).
- *Shrinking under CWD is worth ~15 steps on top:* G0 vs G0nc (Δ +0.00102 / +0.00098).
- *Overall size is not it:* G0nc has the control's matrix norms and is 16 steps worse than C, whose lengths move through Muon's (damped) radial part; Gnc is 30–90 % larger than C and the best. What matters is that each row's length moves with its own gradient.
- *CWD on the gained matrices does not help:* Gnc beats G by 0.00019 / 0.00015 (≈ 2.5 steps). Both seeds agree, but each is only ~1.5–2 fork-noise SDs of a difference (√2·0.000075), so this is a lead, not a result.

### Full runs: no CWD on the gained matrices (`nocwd`, 4 seeds)

Commit `bb1bc0b`, record settings + `ROW_GAIN_NO_CWD=1`, `STOP_STEP=2690`, seeds 0–3 (`nocwd-s<seed>`); compared with `cand-rg20` on the same seeds (`fork_compare.py --full nocwd --full-controls cand-rg20`).

- Track 3 rule (n = 4): 0.00373 at 2660, **0.00438 at 2665** (earliest pass), 0.00721 at 2690.
- Seed-paired vs `cand-rg20` at 2690: −0.00044, −0.00025, −0.00308, −0.00048; mean −0.00106 ± 0.00067 (SE). All 4 lower; without seed 2, −0.00039 (≈ 6 steps). At 2665: mean 3.27781 vs 3.27887.
- The forks predicted ≈ −0.00017. The full-run difference has the same sign in every seed but its size is not resolved by 4 seeds (seed 2 alone is −0.0031).

**Seeds 4–7 added** (`nocwd-s4`–`nocwd-s7`, same commit and settings), n = 8:
- Track 3 rule: 0.00366 at 2655, **0.00445 at 2660** (earliest pass), 0.00536 at 2665, 0.00937 at 2690. (`cand-rg20` with n = 8: 0.00315 at 2660, earliest pass 2665; with n = 16: 0.00407 at 2660.)
- Seed-paired vs `cand-rg20` at 2690: −0.00044, −0.00025, −0.00308, −0.00048, +0.00102, +0.00014, −0.00060, −0.00013; **mean −0.00048 ± 0.00041 (SE)**, median −0.00035, 6 of 8 lower. Same at 2655–2665 (−0.00046 to −0.00048).
- About 7 ± 6 steps: not resolved. The direction agrees with the forks (≈ −0.00017); the size does not separate from zero at n = 8.
