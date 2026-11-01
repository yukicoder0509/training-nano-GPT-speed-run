# Optimizer step (math)

This is the update that `train_gpt.py` and `train_gpt_record.py` perform on each training step: the current record, rec-46 plus four additions:
- row gains on the hidden matrices, with no CWD while they are active;
- the Tail-EMA readout at 2100/200;
- a 2.5× EMA-Nesterov jump on the norm gains;
- a length Adam at β = 0.99, 0.99.

It passes from step 2635 with n = 8 and with n = 16. `record.env` keeps `STOP_STEP=2640`.

The experimental knobs of the earlier record chain were removed from the script after commit `20cc71b`; their definitions and results are in that commit's version of this file and in experiment.md. These knobs stay in `train_gpt.py` (environment):
- `ROW_GAIN_LR`, `ROW_GAIN_START`, `ROW_GAIN_BETAS`;
- `ENA_G_MULT`;
- `TAILEMA_START`, `TAILEMA_TAU`, `TAILEMA_LAMBDA`.

Keep this file in sync with the code: any commit that changes the optimizer updates this file in the same commit.

Notation: t is the loop step; g = ∇L, the gradient summed over ranks. W is a parameter and η_t its learning rate. ‖X‖_S4 = ‖XᵀX‖_F^½ is the Schatten-4 norm, which is what `gram_frobenius_norm_estimate` computes. ⊙ is elementwise multiplication.

## 0. Schedules (`set_hparams`)

- **Learning rate (PowerCool), per parameter group:** η_t = min(η₀, c·(T_end − t)^1.2), with T_end = 2900 (`FINAL_SCHEDULE_STEPS`).
  - Each group has its own constant c (`*_POWER_C`). Muon starts decaying at about step 511.
- **Muon momentum μ_t:**
  - rises linearly 0.85 → 0.95 over steps 0–300;
  - holds at 0.95;
  - falls linearly 0.95 → 0.85 over steps 2700–2900.
- Training stops at `STOP_STEP` = 2640 (`record.env`; rec-46 used 2690), so at the last step η is still about 7 % of η₀ for Muon, and the μ cooldown never starts.

## 1. Outer wrapper: EMA-Nesterov (all parameters)

The wrapper keeps a buffer b and the previous weights W_prev for every parameter.

- **Before the forward pass, only when 300 ≤ t < 1950:** W ← W + β_t·b, with β_t = 0.3·η_{t−1}/η₀.
  - The LR ratio is taken from the embedding Adam group, using the previous step's LR.
  - The norm gains jump 2.5·β_t instead (`ENA_G_MULT`).
  - The shift is not undone afterwards.
- The forward and backward passes then run at the shifted point, and the inner optimizers step W → W′.
- **After the inner step:**
  - b ← 0.99·b + 0.01·(W′ − W_prev)
  - W_prev ← W′

## 2. Muon: hidden 2-D matrices (q, k, v, attn.proj, mlp.fc, mlp.proj)

Each parameter is updated on one rank and then all-gathered. η₀ = 0.0375.

1. **Momentum (Nesterov form):**
   - m ← μ·m + (1−μ)·g
   - U = (1−μ)·g + μ·m
2. **SOAP whitening in an eigenbasis.** L, R, Q_L, Q_R and V are SOAP's per-parameter state.
   - Rotate: P = Q_Lᵀ·U·Q_R
   - Second moment: V ← 0.9·V + 0.1·P²
   - Precondition and rotate back: S = Q_L·(P / √V)·Q_Rᵀ, then rescale so ‖S‖_S4 = ‖U‖_S4.
   - On step 0 there is no eigenbasis yet, so S = U.
3. **Trust gate (attn.proj only).** For q, k, v and the MLP matrices, U ← S directly.
   - a = clip((cos(S,U) − 0.2)/0.8, 0, 1)
   - b = clip(cos(S,g) / cos(U,g), 0, 1)
   - γ = clip(a·b, floor_t, cap_t)
     - floor_t is 0.45 for t < 1375, then falls linearly to 0 at t = 1625.
     - cap_t is 0.85 for t < 1625, then 1.
   - Blend: U ← (U + γ·(S − U)), rescaled to ‖U‖_S4.
4. **Orthogonalize:** O = NS₁₂(U / ‖U‖_S4) · √max(1, rows/cols).
   - NS₁₂ is 12 Newton–Schulz iterations in bf16: X ← 2X − 1.5(XXᵀ)X + 0.5(XXᵀ)²X.
5. **Damp the radial part:**
   - c = ⟨O,W⟩ / ‖W‖²
   - O ← (O − c·W) + s·c·W
   - s = 0.5 when the step would grow ‖W‖ (c < 0), and 1 otherwise.
6. **RowFloor:** for each output row i, O_i ← max(1, 0.3825·‖W_i‖ / ‖O_i‖)·O_i.
7. **Step and length control.**
   - **Before Muon step 1000, and always on mlp.proj: radius pin, then cautious weight decay.**
     - Target norm: r* = ‖W‖ − η·⟨O,W⟩ / ‖W‖
     - Step: W ← W − η·O, then W ← W · r* / ‖W‖
     - Decay: W ← W ⊙ (1 − 0.025·η·1[O ⊙ W_old > 0]). The mask is taken from the weights before the step.
   - **From Muon step 1000 on q, k, v, attn.proj, mlp.fc: row gains, no decay.**
     - Angular step per row: W ← W − η·O, then each row is rescaled back to its length before the step, ‖W_i‖.
     - Each row length then takes its own Adam step:
       - g_i = ⟨∇W_i, W_i/‖W_i‖⟩, from the raw gradient and the weights before the step;
       - Adam(β = 0.99, 0.99, bias-corrected);
       - W_i ← W_i·(1 − a·(η_t/η₀)·m̂_i/√v̂_i), a = 0.02.
     - β₁ ≤ √β₂ keeps m̂/√v̂ bounded.
8. **Update the SOAP statistics with the raw gradient** (used from the next step on):
   - L ← 0.9·L + 0.1·g·gᵀ
   - R ← 0.9·R + 0.1·gᵀ·g
   - Q_L ← qr(L·Q_L) and Q_R ← qr(R·Q_R): one power-iteration step every step, after sorting by eigenvalue.

There is no gradient clipping and no other weight decay.

## 3. Adam groups

- **Embedding and lm_head:** `torch.optim.AdamW` (fused).
  - β = (0.8, 0.99), ε = 1e-10, no weight decay, standard bias correction.
  - η₀ is 0.3 for the embedding and 1/320 for lm_head.
- **1-D parameters (norm gains and biases):** Adam without bias correction, η₀ = 0.01.
  - m ← β₁·m + (1−β₁)·g, with β₁ = 0.8
  - v ← β₂·v + (1−β₂)·g²
  - W ← W − η·m / (√v + 1e-10)
  - β₂ is 0.99 for gains, 0.9965 for the attn.proj bias, and 0.997 for everything else.

## 4. Evaluation only: Tail-EMA (training is unaffected)

- For 2100 ≤ t < 2900, every parameter except the token embedding keeps an average: ē ← ē + (W − ē)/200.
- Validation is scored on the blend (1 − 0.6)·W + 0.6·ē. This is the reported `val_ema_loss`.
- In `train_gpt.py`, validation runs:
  - every 125 steps;
  - at the stop step;
  - every 5 steps over 2650–2690 (`DENSE_VAL`, "lo:hi:stride"), so a run can be scored at the earliest step that passes.

  `train_gpt_record.py` validates only at the stop step.
- Checkpoint/fork (`CKPT_STEPS`, `RESUME_FROM`) restores the exact training state above (weights, Muon/SOAP, Adam, EMA-Nesterov, Tail-EMA, data position); it does not change the update.
