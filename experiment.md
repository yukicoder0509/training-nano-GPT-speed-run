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
