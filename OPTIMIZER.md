# Optimizer step (math)

This is the update that `train_gpt.py` at this commit performs on each training step. The defaults are the current record (rec-46 + row gains + Tail-EMA 2100/200, 2665 steps) plus no CWD on the row-gain matrices (`ROW_GAIN_NO_CWD=1`, 8 seeds, not yet a validated record; `ROW_GAIN_NO_CWD=0` gives the 2665 record); to get rec-46 set `NEWTON_MUON=0 ROW_GAIN_LR=0 TAILEMA_START=2400 TAILEMA_TAU=150 STOP_STEP=2690` or run the `rec-46` tag. Keep this file in sync with the code: any commit that changes the optimizer updates this file in the same commit.

Notation: t is the loop step; g = ∇L, the gradient summed over ranks. W is a parameter and η_t its learning rate. ‖X‖_S4 = ‖XᵀX‖_F^½ is the Schatten-4 norm, which is what `gram_frobenius_norm_estimate` computes. ⊙ is elementwise multiplication.

## 0. Schedules (`set_hparams`)

- **Learning rate (PowerCool), per parameter group:** η_t = min(η₀, c·(T_end − t)^1.2), with T_end = 2900 (`FINAL_SCHEDULE_STEPS`).
  - Each group has its own constant c (`*_POWER_C`). Muon starts decaying at about step 511.
- **Muon momentum μ_t:**
  - rises linearly 0.85 → 0.95 over steps 0–300;
  - holds at 0.95;
  - falls linearly 0.95 → 0.85 over steps 2700–2900.
- Training stops at `STOP_STEP` = 2665 (`record.env`; rec-46 used 2690), so at the last step η is still about 6 % of η₀ for Muon, and the μ cooldown never starts.

## 1. Outer wrapper: EMA-Nesterov (all parameters)

The wrapper keeps a buffer b and the previous weights W_prev for every parameter.

- **Before the forward pass, only when 300 ≤ t < 1950** (the end is `ENA_REST`, default 1950):
  W ← W + β_t·b, with β_t = 0.3·η_{t−1}/η₀.
  - The LR ratio is taken from the embedding Adam group, using the previous step's LR.
  - The shift is not undone afterwards.
- The forward and backward passes then run at the shifted point, and the inner optimizers step W → W′.
- **After the inner step:**
  - b ← 0.99·b + 0.01·(W′ − W_prev)
  - W_prev ← W′

- **Per-group jump (env, default off):** `ENA_GROUP` = norm | embed | head | bias | hidden gives one group its own jump W ← W + β_g·d, all others keep the one above.
  - β_g = `ENA_G_MULT`·β_t (default 1), active for `ENA_G_START` < t + 1 and t < `ENA_G_END` (defaults 300 and the global end).
  - `ENA_G_AUTO`=κ: s_t = ⟨g_t, d⟩ with g_t the gradient at the jumped-to point; s̄ ← 0.9·s̄ + 0.1·sign(s_t); `ENA_G_MULT` ← clip(`ENA_G_MULT`·e^(−κ·s̄), 0, `ENA_G_MAX`) on every active step.
  - `ENA_G_LS`=λ: d is the exponentially weighted least-squares velocity of the group's weights (weights λ^age; S0 ← θ + λS0, S1 ← λ(S1 + S0_old), d = (W1·S0 − W0·S1)/(W0·W2 − W1²)) instead of b.

## 2. Muon: hidden 2-D matrices (q, k, v, attn.proj, mlp.fc, mlp.proj)

Each parameter is updated on one rank and then all-gathered.

0. **Newton-Muon on the gradient, only when `NEWTON_MODE=grad`** (as in record #15):
   - g ← g·K_ℓ, with no rescale (§2a).
   - Every later use of g in this section (momentum, the trust gate's cos(·,g), and the L/R statistics) then sees g·K_ℓ.
1. **Momentum (Nesterov form):**
   - m ← μ·m + (1−μ)·g
   - U = (1−μ)·g + μ·m
2. **SOAP whitening in an eigenbasis.** L, R, Q_L, Q_R and V are SOAP's per-parameter state.
   - Rotate: P = Q_Lᵀ·U·Q_R
   - Second moment: V ← 0.9·V + 0.1·P²
   - Precondition and rotate back: S = Q_L·(P / √V)·Q_Rᵀ, then rescale so ‖S‖_S4 = ‖U‖_S4.
   - On step 0 there is no eigenbasis yet, so S = U.
   - **One-sided when `NEWTON_MODE=right`:** R and Q_R are not kept, and Q_R = I. So P = Q_Lᵀ·U and S = Q_L·(P / √V), and K_ℓ takes over the input (column) side.
3. **Newton-Muon right preconditioning, off by default** (`NEWTON_MUON=1` turns it on; it applies when `NEWTON_MODE` is `soap` (the default) or `right`; record #15 / PR #281):
   - Ŝ = S·K_ℓ, rescaled so ‖Ŝ‖_S4 = ‖S‖_S4.
   - K_ℓ is the preconditioner for this layer's input space (§2a).
   - For mlp.proj (input width 3072), K_ℓ is block-diagonal with 4 blocks of 768×768.
   - With `NEWTON_MODE=grad`, this step is skipped and Ŝ = S.
4. **Trust gate (attn.proj only).** For q, k, v and the MLP matrices, U ← Ŝ directly.
   - a = clip((cos(Ŝ,U) − 0.2)/0.8, 0, 1)
   - b = clip(cos(Ŝ,g) / cos(U,g), 0, 1)
   - γ = clip(a·b, floor_t, cap_t)
     - floor_t is 0.45 for t < 1375, then falls linearly to 0 at t = 1625.
     - cap_t is 0.85 for t < 1625, then 1.
   - Blend: U ← (U + γ·(Ŝ − U)), rescaled to ‖U‖_S4.
5. **Orthogonalize:** O = NS₁₂(U / ‖U‖_S4) · √max(1, rows/cols).
   - NS₁₂ is 12 Newton–Schulz iterations in bf16: X ← 2X − 1.5(XXᵀ)X + 0.5(XXᵀ)²X.
6. **Damp the radial part:**
   - c = ⟨O,W⟩ / ‖W‖²
   - O ← (O − c·W) + s·c·W
   - s = 0.5 when the step would grow ‖W‖ (c < 0), and 1 otherwise.
7. **RowFloor:** for each output row i, O_i ← max(1, 0.3825·‖W_i‖ / ‖O_i‖)·O_i.
8. **Radius pin, then cautious weight decay:**
   - Target norm: r* = ‖W‖ − η·⟨O,W⟩ / ‖W‖
   - Step: W ← W − η·O, then W ← W · r* / ‖W‖
   - Decay: W ← W ⊙ (1 − 0.025·η·1[O ⊙ W_old > 0]). The mask is taken from the weights before the step. Skipped on q, k, v, o, fc while their row gains are active (`ROW_GAIN_NO_CWD=1`, the default).
9. **Update the SOAP statistics with the raw gradient** (used from the next step on):
   - L ← 0.9·L + 0.1·g·gᵀ
   - R ← 0.9·R + 0.1·gᵀ·g (skipped in `right` mode)
   - Q_L ← qr(L·Q_L) and Q_R ← qr(R·Q_R): one power-iteration step every step, after sorting by eigenvalue.

There is no gradient clipping and no other weight decay.

**Muon learning rate:** η₀ = 0.0375 · m_type.
- The multipliers m_type are set by `MUON_LR_MULT` = "m_qkv, m_attn.proj, m_mlp", default "1,1,1". #15's ratios are 0.8, 1, 1.2.
- Each type's PowerCool constant is c·m_type, so η_t = m_type·η_t^(base) and every type starts decaying at the same step.
- V/O SOAP knobs (env, defaults = rec-46):
  - `V_GATE=1`: the trust gate of step 4 is also applied on attn.v (default: attn.proj only).
  - `VO_DENOM_POWER` = p: for attn.v and attn.proj, step 2 divides by V^p instead of V^½ (default 0.5).
  - `VO_HEAD=1`: step 9's new statistics are masked to the 6 head blocks (128×128) on V's output side (L) and O's input side (R), so those bases are block-diagonal per head.
- Per-row norm control (Muown #23; env). Default: row gains on q,k,v,o,fc from step 1000 with a = 0.02 (the current record); `ROW_GAIN_LR=0` restores rec-46.
  - `ROW_PIN=1`: step 8's pin is per row: r*_i = ‖W_i‖ − η⟨O_i,W_i⟩/‖W_i‖ and each row is rescaled to r*_i.
  - `ROW_GAIN_SCOPE` (types among q,k,v,o,fc,proj; default q,k,v,o,fc) limits the row gains to those matrices (mlp.proj keeps the rec-46 pin); `ROW_GAIN_MODE=matrix` uses one gain per matrix (pure angular step for the whole matrix, g = ⟨∇W, W/‖W‖⟩) instead of one per row.
  - `ROW_GAIN_START` = t₀ (default 1000): the row gains are used only from step t₀ on (before it, step 8 is the rec-46 pin). `ROW_GAIN_END` = t₁ (default 0 = never): from step t₁ on, step 8 is the rec-46 pin again.
  - `ROW_GAIN_FIXED=1`: step 8 is the direction-only step per row (r*_i = ‖W_i‖) without the Adam length step, so row lengths change only through CWD (ablation: "drop the radial part" vs "learn the lengths").
  - `ROW_GAIN_NO_CWD` (default 1): no CWD on those matrices while their row gains are active, i.e. from step 1000 on q, k, v, o, fc only mlp.proj keeps CWD; with `ROW_GAIN_FIXED=1` every row length stays exactly constant. `ROW_GAIN_NO_CWD=0` is the 2665 record.
  - `ROW_GAIN_LR` = a > 0 (default 0.02): step 8 is purely angular per row (r*_i = ‖W_i‖), then each row norm takes its own Adam step: g_i = ⟨∇W_i, W_i/‖W_i‖⟩ (raw gradient), Adam(β = 0.9, 0.95, bias-corrected), W_i ← W_i·(1 − a·(η_t/η₀)·m̂_i/√v̂_i). CWD still follows.
- SOAP variants (env, defaults = rec-46; scope `SOAP_*_SCOPE` = all | vo):
  - `SOAP_SINK=cur|ema`: step 2's P/√V is replaced by √r_i·P_ij·√c_j, with r, c the Sinkhorn scalings (10 iterations) that give r_i B_ij c_j uniform row and column marginals; B = P² (cur, as SinkSOAP #26) or B = V (ema, a factored second moment).
  - `SOAP_KL=1`: step 9's statistics become KL-SOAP's coupled estimates (#19/#25): L ← 0.9·L + 0.1·G·Q_R·diag(1/λ_R)·Q_Rᵀ·Gᵀ/n and R ← 0.9·R + 0.1·Gᵀ·Q_L·diag(1/λ_L)·Q_Lᵀ·G/m, where λ_L, λ_R are EMAs (0.9) of the row/column energies of Q_Lᵀ·G·Q_R whitened by the other side. The first KL step replaces L and R instead of mixing.
- Screening only: `VO_VARIANTS=1` also computes V/O SOAP variant directions (per-head statistics, a shared V–O basis, V^¼ denominator, trust gate on V; class `VOVariants`) and records them on probe steps. They are never applied, so the update above is unchanged.
- V/O ablation knobs (env, default 1 = rec-46): `VO_LR_MULT` multiplies m_type for attn.v and attn.proj (same PowerCool scaling), and `VO_FLOOR_MULT` multiplies their RowFloor target (0.3825 in step 7).

### 2a. Newton-Muon preconditioner K_ℓ

ℓ indexes the input spaces, 4 per block:
- the shared input of q/k/v (output of norm1);
- the input of attn.proj (attention output);
- the input of mlp.fc (output of norm2);
- the input of mlp.proj (relu² activations), split into 4 blocks of 768.

- **Collect statistics, only on refresh steps** (t + 1 ≡ 0 mod 64). Each layer's `Linear.forward` accumulates C_ℓ = mean over tokens of x·xᵀ for that step's batch. C_ℓ is averaged over ranks and microbatches.
- **Refresh, at the start of that same step's Muon update:**
  - Σ_ℓ ← 0.95·Σ_ℓ + 0.05·C_ℓ
  - K_ℓ = (Σ_ℓ + (δ·tr(Σ_ℓ)/d + 10⁻⁸)·I)⁻¹, computed by Cholesky.
  - The damping is δ = `NEWTON_DAMPING`, default 0.2 (#15's value). A larger δ makes K_ℓ closer to a scaled identity, i.e. weaker preconditioning.
  - If the Cholesky fails, K_ℓ = I.
- **Initial values:** Σ_ℓ = 0.001·I. K_ℓ = I until the first refresh at t = 63, so Newton-Muon has no effect on steps 0–62.

## 3. Adam groups

- **Embedding and lm_head:** `torch.optim.AdamW`.
  - β = (0.8, 0.99), ε = 1e-10, no weight decay, standard bias correction.
  - η₀ is 0.3 for the embedding and 1/320 for lm_head.
- **1-D parameters (norm gains and biases):** Adam without bias correction, η₀ = 0.01·a.
  - m ← β₁·m + (1−β₁)·g, with β₁ = 0.8
  - v ← β₂·v + (1−β₂)·g²
  - W ← W − η·m / (√v + 1e-10)
  - β₂ is 0.99 for gains, 0.9965 for the attn.proj bias, and 0.997 for everything else.
  - Ablation knobs (env), defaults = rec-46: `GAIN_BETA2` (gains' β₂, 0.99), `AUX_BETA1` (β₁ for all 1-D params, 0.8), `AUX_LR_MULT` = a (1.0). a scales the PowerCool constant too, so η_t = a·η_t^(base).

## 4. Evaluation only: Tail-EMA (training is unaffected)

- For 2100 ≤ t < 2900, every parameter except the token embedding keeps an average: ē ← ē + (W − ē)/200. (env `TAILEMA_START`, `TAILEMA_TAU`, `TAILEMA_LAMBDA`; defaults 2100 / 200 / 0.6; rec-46 used 2400 / 150.)
- Validation is scored on the blend (1 − 0.6)·W + 0.6·ē. This is the reported `val_ema_loss`.
- Validation runs every 125 steps, at the stop step, and densely every 5 steps over 2650–2690 (`DENSE_VAL`, "lo:hi:stride"), so a run can be scored at the earliest step that passes.
- Readout lab (`READOUT_LAB=1`, eval only): keeps weight snapshots after `LAB_REF_STEPS` and update-EMAs m_β (β = 0.9, 0.97) of Δθ from `LAB_VEL_START`; at the stop step scores θ_eval = base + γ·(base − θ_ref) and base + γ·m_β, for base = raw W or the Tail-EMA blend, plus a λ grid of the Tail-EMA blend. Results go to `<log>.readout.json`.
- Checkpoint/fork (`CKPT_STEPS`, `RESUME_FROM`) restores the exact training state above (weights, Muon/SOAP, Adam, EMA-Nesterov, Tail-EMA, data position); it does not change the update.
