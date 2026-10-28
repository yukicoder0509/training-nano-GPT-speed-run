"""
train_gpt_cwd_SOTA.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).

SOTA optimizer for a PR. It is the clean SOAP-Muon base from PR #321 (the 2750/2755
"aux-b2 + SOAP-f1 clean" record) with THREE additional levers stacked on top, each on a
different axis chosen to survive the radius pin (which re-pins global Frobenius magnitude
every step, so only directional / shape / readout changes persist):

  (A) Tail-EMA eval readout (PR #325): an eval-time-only weight blend. Over the cooldown
      tail [2400, 2900] maintain a slow EMA  ema += (w - ema) / TAU  (TAU=150) of every
      parameter except the token embedding; at validation, evaluate the partial blend
      w_eval = (1-L)*w + L*ema  (L=0.6). Training is untouched; only the weights used for
      the val forward change. The partial (not full) blend cancels the late cross-valley
      oscillation while still tracking the descending floor.

  (B) RowFloor (per-output-row u/w-floor): replace the SCALAR u/w-floor with a per-row
      floor on the orthogonalized update -- each output row whose update norm is below
      TARGET_UW * ||row|| is lifted to that target (RHO=1.0, no Frobenius renorm). A
      per-row SHAPE change, so it survives the radius pin.

  (C) Cautious Weight Decay (CWD=0.025, POST-pin; Cautious Optimizers arXiv:2411.16085):
      anisotropic per-coordinate decay  p *= 1 - lr*CWD*mask  with mask = 1[update*p > 0]
      (only the coords the optimizer step already shrinks; never fights coords it wants to
      grow), applied AFTER rescale_to_radius so the per-coord SHAPE change is not
      re-normalized away.

The clean PR #321 base retains:
  - SOAP preconditioning on all hidden matrices (all_hidden, freq=1)
  - u/w-floor hyperball constraint (TARGET_UW=0.3825); here per-row (RowFloor)
  - radial scaling + rescale-to-radius (the radius pin)
  - EMA-Nesterov outer wrapper
  - PowerCool LR schedule, mu schedule, val schedule
  - aux-Adam beta2 split, depth-scaled mlp.fc init

All hyperparameters are hardcoded; only --seed is a command-line argument. Set CWD=0.0,
ROWFLOOR=False and TAILEMA_TAU=0 below to recover the clean PR #321 base.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import argparse
import json
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist
import harness


# cuDNN SDPA can fail to build an execution plan for this compiled causal-attention
# layout on some PyTorch/CUDA/cuDNN combinations. Leave Flash/mem-efficient/math SDPA
# enabled and only remove the cuDNN backend from consideration.
torch.backends.cuda.enable_cudnn_sdp(False)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


SEED = args.seed
# Hardcoded final schedule constants
FINAL_TRAIN_STEPS = 2900
# Cooldown horizon for the LR anneal. SOTA uses 2900.
FINAL_SCHEDULE_STEPS = 2900   # t_end win (the +25-step SOTA lever; original public was 2960)
FINAL_LR_POWER = 1.2
ADAM_EMBED_POWER_C = 4.976805410800738e-05
ADAM_PROJ_POWER_C = 5.184172302917436e-07
ADAM_OTHER_POWER_C = 1.6589351369335795e-06
MUON_POWER_C = 3.3169534699576625e-06
EXPERIMENT_NAME = "clean-simplified-soaph"
EXPERIMENT_INTUITION = "SOTA SOAP-all_hidden (freq=1) + u/w-floor + radial scale + EMA-Nesterov, with Soft-Muon / Contra-Muon / Circuit-Muon / CenterShrink / Aurora removed."
MU = 0.95
MUON_LR = 0.0375
TARGET_UW = 0.3825
SOAP_TARGET_UW = TARGET_UW
NONSOAP_TARGET_UW = TARGET_UW
SOAP_BETA2 = 0.90
SOAP_PRECONDITION_FREQUENCY = 1
SOAP_DENOM_POWER = 0.50
ATTN_EARLY_TRUST_FLOOR = 0.45
ATTN_EARLY_TRUST_CAP = 0.85
ATTN_TRUST_FLOOR_END_STEP = 1375
ATTN_TRUST_FLOOR_FADE_END_STEP = 1625
ATTN_TRUST_MIN_AGREE = 0.20
ATTN_TRUST_MIN_GRAD_ALIGN = 0.00
ATTN_TRUST_POWER = 1.00
TRAIN_PROGRESS_INTERVAL = 0
LOG_DIR = Path("logs")
# Checkpoint / fork. CKPT_STEPS="1000,1500,2000" saves the full training state at the top of those steps (after their
# validation) to CKPT_DIR/step<k>/: common.pt (replicated state, rank 0) + muon_rank<r>.pt (each rank's Muon shard).
# RESUME_FROM=<CKPT_DIR>/step<k> starts the loop at step k from that state; env knobs set for the resumed run take
# effect from step k on (a fork). Not covered: the Newton-Muon K/covariance and the VO_VARIANTS shadow states.
CKPT_STEPS = {int(x) for x in os.environ.get("CKPT_STEPS", "").split(",") if x}
CKPT_DIR = os.environ.get("CKPT_DIR", f"/work/{os.environ.get('USER', 'user')}/nanogpt-ckpt/"
                                      f"{os.environ.get('WANDB_RUN_NAME', 'run')}-seed{args.seed}")
RESUME_FROM = os.environ.get("RESUME_FROM", "")
RESUME_STEP = json.load(open(f"{RESUME_FROM}/meta.json"))["step"] if RESUME_FROM else 0
# Readout lab (eval only): READOUT_LAB=1 keeps weight snapshots after LAB_REF_STEPS, extra weight EMAs (LAB_EMAS,
# "start:tau" pairs, non-embedding params like Tail-EMA) and the last RRE iterates, then at STOP_STEP scores a grid of
# late-trajectory readouts on the val set AND on a held-out set (the last train shard, past the probe/GNS batches),
# so a readout picked on val can be confirmed on data it was not picked on. Training is unchanged.
# (The update-velocity readout of #38 was removed: every setting was worse than Tail-EMA alone on 3 seeds.)
READOUT_LAB = os.environ.get("READOUT_LAB", "0") == "1"
LAB_REF_STEPS = [int(x) for x in os.environ.get("LAB_REF_STEPS", "2000,2200,2375").split(",") if x]
LAB_EMAS = [tuple(int(v) for v in x.split(":")) for x in
            os.environ.get("LAB_EMAS", "2000:300,2200:150,2200:250,2400:100,2400:250").split(",") if x]
LAB_RRE_K, LAB_RRE_EVERY = 4, 5   # RRE (#34) from the iterates at STOP_STEP - k*every, ..., STOP_STEP
# Dense validation near the end, "lo:hi:stride" (inclusive), to score a run at the earliest step that passes.
DENSE_VAL = os.environ.get("DENSE_VAL", "2650:2690:5")
RADIAL_OUTWARD_SCALE = 0.5
RADIAL_INWARD_SCALE = 1.0
HEAD_DIM = 128

# --- The three SOTA levers stacked on the clean PR #321 base (all hardcoded). ---
# (B) RowFloor: per-output-row u/w-floor on the orthogonalized update (replaces the scalar
#     floor for 2-D Muon params). RHO=1.0, no Frobenius renorm (magnitude is then re-pinned
#     by rescale_to_radius, so only the per-row SHAPE survives). ROWFLOOR=False -> scalar floor.
ROWFLOOR = True
ROWFLOOR_RHO = 1.0
# (C) Cautious Weight Decay: anisotropic per-coordinate decay on the 2-D Muon params, applied
#     AFTER the radius pin (POST). mask = 1[update*p > 0]. CWD=0.0 -> off.
CWD = 0.025
# (A) Tail-EMA eval readout: slow weight EMA over [START,END] (token embed excluded), eval the
#     partial blend (1-LAMBDA)*w + LAMBDA*ema. TAILEMA_TAU=0 -> off (eval raw weights only).
TAILEMA_TAU = 150.0
TAILEMA_START = 2400
TAILEMA_END = 2900
TAILEMA_LAMBDA = 0.6
# (D) Newton-Muon (record #15, PR #281, arXiv:2604.01472): right-precondition every hidden matrix's
#     SOAP-whitened update by K = (Sigma + DAMPING*tr(Sigma)/d*I)^-1, Sigma = EMA of the layer-input
#     covariance x x^T (collected in Linear.forward on refresh steps only). K is refreshed every REFRESH
#     steps (K = I before the first refresh). mlp.proj (in=3072) uses a 4-block-diagonal K of 768^2
#     blocks. NEWTON_MODE="soap": U <- SOAP(U) K, before the attn.proj trust gate and Newton-Schulz.
#     NEWTON_MODE="grad": g <- g K before momentum (as in #15), so momentum, SOAP's statistics and the
#     trust gate all see the preconditioned gradient.
#     NEWTON_MODE="right": one-sided SOAP (output/row side only) then U <- SOAP_L(U) K, i.e. K replaces
#     SOAP's input-side (column) whitening instead of stacking on it.
NEWTON_MUON = os.environ.get("NEWTON_MUON", "1") == "1"
NEWTON_MODE = os.environ.get("NEWTON_MODE", "soap")
assert NEWTON_MODE in ("soap", "grad", "right")
SOAP_ONE_SIDED = NEWTON_MUON and NEWTON_MODE == "right"
NEWTON_REFRESH = 64
NEWTON_COV_EMA = 0.05
NEWTON_DAMPING = float(os.environ.get("NEWTON_DAMPING", 0.2))   # larger -> K closer to a scaled identity
# Per-type Muon lr multipliers (q/k/v, attn.proj, mlp.fc+mlp.proj); #15 used 0.8 : 1 : 1.2. The PowerCool
# constant is scaled by the same factor so every type starts decaying at the same step.
MUON_LR_MULT = dict(zip(("qkv", "attn_proj", "mlp"),
                        (float(x) for x in os.environ.get("MUON_LR_MULT", "1,1,1").split(","))))
NEWTON_BLOCK = 768
# attn.v + attn.proj (V/O) ablation knobs, on top of MUON_LR_MULT; defaults are rec-46. VO_LR_MULT scales their
# Muon lr (and PowerCool constant); VO_FLOOR_MULT scales their RowFloor target u/w.
VO_LR_MULT = float(os.environ.get("VO_LR_MULT", 1.0))
VO_FLOOR_MULT = float(os.environ.get("VO_FLOOR_MULT", 1.0))
# Screening only: compute V/O SOAP variant directions (class VOVariants) and record them on probe steps for
# harness.ProbeQuadratic. They are never applied, so training is unchanged; costs extra compute every step.
VO_VARIANTS = os.environ.get("VO_VARIANTS", "0") == "1"
# Applied V/O SOAP changes (defaults = rec-46): V_GATE=1 puts the attn.proj trust gate on attn.v too;
# VO_DENOM_POWER is the SOAP denominator power for attn.v / attn.proj (rec-46: SOAP_DENOM_POWER = 0.5);
# VO_HEAD=1 uses per-head (block-diagonal) SOAP statistics on V's output side and O's input side.
V_GATE = os.environ.get("V_GATE", "0") == "1"
VO_DENOM_POWER = float(os.environ.get("VO_DENOM_POWER", 0.5))
VO_HEAD = os.environ.get("VO_HEAD", "0") == "1"
# Per-row norm control (Muown #23 ideas; defaults = rec-46). ROW_PIN=1 replaces the whole-matrix radius pin by a
# per-row pin (each row keeps its own first-order radial change). ROW_GAIN_LR=a > 0 makes the Muon step purely
# angular per row and moves each row norm with its own bias-corrected Adam(0.9, 0.95) on g_i = <grad_i, W_i/|W_i|>:
# |W_i| *= 1 - a * (lr/lr0) * adam_i.
# SOAP variants (defaults = rec-46). SOAP_SINK = "cur" | "ema": replace the elementwise Adam denominator in the
# eigenbasis by a rank-1 Sinkhorn balancing (SinkSOAP #26) of the current projected energy P^2 ("cur") or of the EMA
# second moment V ("ema", a factored V). SOAP_KL=1: KL-SOAP (#19/#25) coupled covariance estimate for the bases,
# L ~ E[G R^-1 G^T]/n, R ~ E[G^T L^-1 G]/m, with R^-1, L^-1 from running eigenvalue estimates in the current basis.
# SOAP_SINK_SCOPE / SOAP_KL_SCOPE: "all" or "vo" (attn.v + attn.proj only).
SOAP_SINK = os.environ.get("SOAP_SINK", "")
SOAP_KL = os.environ.get("SOAP_KL", "0") == "1"
SOAP_SINK_SCOPE = os.environ.get("SOAP_SINK_SCOPE", "all")
SOAP_KL_SCOPE = os.environ.get("SOAP_KL_SCOPE", "all")
ENA_REST = int(os.environ.get("ENA_REST", 0))   # EMA-Nesterov lookahead end step (0 = rec-46: train_steps - 950 = 1950)
ROW_PIN = os.environ.get("ROW_PIN", "0") == "1"
ROW_GAIN_LR = float(os.environ.get("ROW_GAIN_LR", 0.0))
# ROW_GAIN_SCOPE: comma list of matrix types it applies to (q,k,v,o,fc,proj; default all). ROW_GAIN_MODE=matrix: one
# gain per matrix (pure angular step for the whole matrix, then W *= 1 - a*(lr/lr0)*adam(<grad, W/|W|>)) instead of
# one per row - separates "rows of a matrix can differ in norm" from "matrices can change scale".
ROW_GAIN_SCOPE = set(os.environ.get("ROW_GAIN_SCOPE", "q,k,v,o,fc,proj").split(","))
ROW_GAIN_MODE = os.environ.get("ROW_GAIN_MODE", "row")
# 1-D Adam (norm gains and biases) ablation knobs; defaults are rec-46. AUX_LR_MULT scales both eta0 and
# the PowerCool constant, i.e. the whole LR curve.
GAIN_BETA2 = float(os.environ.get("GAIN_BETA2", 0.99))
AUX_BETA1 = float(os.environ.get("AUX_BETA1", 0.8))
AUX_LR_MULT = float(os.environ.get("AUX_LR_MULT", 1.0))


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024, skip=0):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        if skip > 0:  # resume: fast-forward past the batches already trained on
            skip, pos = skip - 1, pos + batch_size
            continue
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (norm(x.float()) * self.gains).type_as(x)

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)
        # Newton-Muon input statistics, set only on refresh steps: (nb, d, d) accumulator + count.
        # Collected in forward (not a hook) because forward hooks added after model.compile never fire.
        self.newton_accum = None
        self.newton_count = None

    def forward(self, x):
        if self.newton_accum is not None:
            xf = x.detach().flatten(0, -2).float()
            n, nb, d = xf.size(0), self.newton_accum.size(0), self.newton_accum.size(-1)
            xf = xf.reshape(n, nb, d).permute(1, 2, 0)
            self.newton_accum.add_(torch.bmm(xf, xf.mT), alpha=1 / n)
            self.newton_count.add_(1)
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def gram_frobenius_norm_estimate(G: Tensor, keepdim: bool = False, eps: float = 1e-10) -> Tensor:
    X = G.float()
    gram = X.mT @ X if X.size(-2) > X.size(-1) else X @ X.mT
    return gram.norm(dim=(-2, -1), keepdim=keepdim).sqrt().clamp_min(eps)

def _ns_inner(X: Tensor) -> Tensor:
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    # Standard Muon Newton-Schulz orthogonalization. (Aurora wide-matrix row-rescale
    # removed -- ablated neutral; this is exactly the AURORA_K=0 path.)
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / gram_frobenius_norm_estimate(X, keepdim=True, eps=1e-7).to(X.dtype)
    X = _ns_inner(X)
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def should_soap_param(name: str) -> bool:
    is_mlp_fc = name.endswith(".mlp.fc.weight")
    is_mlp_proj = name.endswith(".mlp.proj.weight")
    is_attn_proj = name.endswith(".attn.proj.weight")
    is_qkv = (
        name.endswith(".attn.q.weight")
        or name.endswith(".attn.k.weight")
        or name.endswith(".attn.v.weight")
    )
    is_q = name.endswith(".attn.q.weight")
    is_k = name.endswith(".attn.k.weight")
    is_v = name.endswith(".attn.v.weight")
    return is_mlp_fc or is_mlp_proj or is_attn_proj or is_qkv  # SOAP on all hidden 2D matrices

def is_attn_proj_param(name: str) -> bool:
    return name.endswith(".attn.proj.weight")

def is_attn_param(name: str) -> bool:
    return (
        name.endswith(".attn.q.weight")
        or name.endswith(".attn.k.weight")
        or name.endswith(".attn.v.weight")
        or name.endswith(".attn.proj.weight")
    )

def tensor_cosine(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    a_f, b_f = a.float(), b.float()
    return (a_f * b_f).sum() / (a_f.norm() * b_f.norm()).clamp_min(eps)

def trust_gate(raw: Tensor, soap: Tensor, grad: Tensor, eps: float = 1e-8) -> Tensor:
    # SOAP is trusted when it still points with raw momentum and is at least as
    # gradient-aligned as raw momentum. This catches stale whitening bases.
    raw_grad = tensor_cosine(raw, grad, eps)
    soap_grad = tensor_cosine(soap, grad, eps)
    soap_raw = tensor_cosine(soap, raw, eps)

    agree_gate = ((soap_raw - ATTN_TRUST_MIN_AGREE) / (1 - ATTN_TRUST_MIN_AGREE)).clamp(0, 1)
    denom = (raw_grad - ATTN_TRUST_MIN_GRAD_ALIGN).clamp_min(eps)
    grad_gate = ((soap_grad - ATTN_TRUST_MIN_GRAD_ALIGN) / denom).clamp(0, 1)
    gate = (agree_gate * grad_gate).clamp(0, 1)
    if ATTN_TRUST_POWER != 1.0:
        gate = gate.pow(ATTN_TRUST_POWER)
    return gate

def early_trust_floor_for_step(step: int) -> float:
    if ATTN_TRUST_FLOOR_FADE_END_STEP <= ATTN_TRUST_FLOOR_END_STEP:
        return 0.0 if step >= ATTN_TRUST_FLOOR_FADE_END_STEP else ATTN_EARLY_TRUST_FLOOR
    if step < ATTN_TRUST_FLOOR_END_STEP:
        return ATTN_EARLY_TRUST_FLOOR
    if step >= ATTN_TRUST_FLOOR_FADE_END_STEP:
        return 0.0
    return ATTN_EARLY_TRUST_FLOOR * (
        ATTN_TRUST_FLOOR_FADE_END_STEP - step
    ) / (ATTN_TRUST_FLOOR_FADE_END_STEP - ATTN_TRUST_FLOOR_END_STEP)

def bounded_trust_gate(gate: Tensor, step: int) -> Tensor:
    floor = early_trust_floor_for_step(step)
    cap = ATTN_EARLY_TRUST_CAP if step < ATTN_TRUST_FLOOR_FADE_END_STEP else 1.0
    return gate.clamp(min=floor, max=cap)

def norm_preserving_blend(raw: Tensor, soap: Tensor, gate: Tensor, eps: float = 1e-8) -> Tensor:
    blended = raw + (soap - raw) * gate.to(raw.dtype)
    raw_norm = gram_frobenius_norm_estimate(raw, eps=eps)
    blended_norm = gram_frobenius_norm_estimate(blended, eps=eps)
    return (blended * (raw_norm / blended_norm).to(blended.dtype)).to(raw.dtype)

def scale_radial_update(update: Tensor, param: Tensor, eps: float = 1e-12) -> Tensor:
    update_f = update.float()
    param_f = param.float()
    denom = (param_f * param_f).sum().clamp_min(eps)
    coeff = (update_f * param_f).sum() / denom
    radial = coeff * param_f
    tangential = update_f - radial
    # p.add_(update, alpha=-lr), so actual movement is -update.
    # Outward movement means (-update) is aligned with p, i.e. coeff < 0.
    radial_scale = torch.where(
        coeff < 0,
        update_f.new_tensor(RADIAL_OUTWARD_SCALE),
        update_f.new_tensor(RADIAL_INWARD_SCALE),
    )
    return (tangential + radial_scale * radial).to(update.dtype)

def target_radius_after_update(param: Tensor, update: Tensor, lr: float, eps: float = 1e-8) -> Tensor:
    param_f = param.float()
    update_f = update.float()
    before_norm = param_f.norm().clamp_min(eps)
    # Use only the radial component's first-order radius change as the intended
    # radius change; the post-step rescale below removes finite tangent drift.
    radial_delta = -lr * (update_f * param_f).sum() / before_norm
    return (before_norm + radial_delta).clamp_min(eps)

def rescale_to_radius(param: Tensor, target_norm: Tensor, eps: float = 1e-8):
    after_norm = param.float().norm().clamp_min(eps)
    param.mul_((target_norm / after_norm).to(param.dtype))

def soap_eigenbasis(mat: Tensor) -> Tensor:
    try:
        _, q = torch.linalg.eigh(mat + 1e-30 * torch.eye(mat.size(0), device=mat.device))
    except RuntimeError:
        _, q = torch.linalg.eigh(mat.double() + 1e-30 * torch.eye(mat.size(0), device=mat.device))
        q = q.float()
    return torch.flip(q, [1])

def soap_basis_qr(row_gg, col_gg, q_row, q_col, exp_avg_sq):
    # col_gg is None for one-sided (row-only) SOAP
    row_eig = torch.diag(q_row.T @ row_gg @ q_row)
    row_sort = torch.argsort(row_eig, descending=True)
    q_row = q_row[:, row_sort]
    exp_avg_sq = exp_avg_sq.index_select(0, row_sort)
    q_row, _ = torch.linalg.qr(row_gg @ q_row)
    if col_gg is None:
        return q_row, q_col, exp_avg_sq

    col_eig = torch.diag(q_col.T @ col_gg @ q_col)
    col_sort = torch.argsort(col_eig, descending=True)
    q_col = q_col[:, col_sort]
    exp_avg_sq = exp_avg_sq.index_select(1, col_sort)
    q_col, _ = torch.linalg.qr(col_gg @ q_col)
    return q_row, q_col, exp_avg_sq

def sinkhorn_sqrt_scalings(B, steps=10):
    """Row/col scalings r, c with r_i B_ij c_j having uniform marginals (SinkSOAP); returns sqrt(r)[:, None], sqrt(c)."""
    m, n = B.shape
    B = B + B.mean() * 1e-6
    c = torch.ones(n, device=B.device, dtype=B.dtype)
    for _ in range(steps):
        r = (1.0 / m) / (B @ c)
        c = (1.0 / n) / (B.T @ r)
    return r.sqrt()[:, None], c.sqrt()[None, :]

def soap_precondition_momentum(update, state, beta2=SOAP_BETA2, eps=1e-8, denom_power=SOAP_DENOM_POWER, sink=""):
    update_f = update.float()
    if state["q_row"] is None:
        return update
    q_row, q_col = state["q_row"], state["q_col"]
    projected = q_row.T @ update_f if q_col is None else q_row.T @ update_f @ q_col
    state["exp_avg_sq"].mul_(beta2).add_(projected.square(), alpha=1 - beta2)
    if sink and q_col is not None:
        sr, sc = sinkhorn_sqrt_scalings(state["exp_avg_sq"] if sink == "ema" else projected.square())
        precond = q_row @ (projected * sr * sc) @ q_col.T
    else:
        denom = state["exp_avg_sq"].clamp_min(eps * eps).pow(denom_power)
        precond = q_row @ (projected / denom) if q_col is None else q_row @ (projected / denom) @ q_col.T
    precond.mul_(gram_frobenius_norm_estimate(update_f, eps=eps) / gram_frobenius_norm_estimate(precond, eps=eps))
    return precond.to(update.dtype)

def kl_soap_targets(grad_f, state, beta):
    """KL-SOAP coupled covariance targets: G R^-1 G^T / n and G^T L^-1 G / m, with L^-1, R^-1 diagonal in the current
    bases (running eigenvalue estimates). Returns None until a basis exists."""
    q_row, q_col = state["q_row"], state["q_col"]
    if q_row is None or q_col is None:
        return None
    m, n = grad_f.shape
    if "kl_eig_row" not in state:
        state["kl_eig_row"] = torch.ones(m, device=grad_f.device)
        state["kl_eig_col"] = torch.ones(n, device=grad_f.device)
    inv_row = state["kl_eig_row"].clamp_min(state["kl_eig_row"].max() * 1e-8).rsqrt()
    inv_col = state["kl_eig_col"].clamp_min(state["kl_eig_col"].max() * 1e-8).rsqrt()
    rw = (q_col.T @ grad_f.T) * inv_col[:, None]
    lw = (q_row.T @ grad_f) * inv_row[:, None]
    proj = q_row.T @ grad_f @ q_col
    fresh = "kl_started" not in state
    for key, d in (("kl_eig_row", (proj * inv_col[None, :]).square().mean(dim=1)),
                   ("kl_eig_col", (proj * inv_row[:, None]).square().mean(dim=0))):
        state[key] = d if fresh else state[key].lerp_(d, 1 - beta)
    state["kl_started"] = True
    return rw.T @ rw / n, lw.T @ lw / m, fresh

def soap_update_preconditioner(grad, state, shampoo_beta=SOAP_BETA2, precondition_frequency=SOAP_PRECONDITION_FREQUENCY,
                               row_mask=None, col_mask=None, kl=False):
    # row_mask / col_mask: elementwise masks on the new statistics (per-head block-diagonal SOAP, VO_HEAD)
    grad_f = grad.float()
    klt = kl_soap_targets(grad_f, state, shampoo_beta) if kl else None
    if klt is not None:  # the first KL step replaces the plain statistics (different scale) instead of mixing in
        row_new, col_new, fresh = klt
        for key, new in (("row_gg", row_new), ("col_gg", col_new)):
            state[key].copy_(new) if fresh else state[key].lerp_(new, 1 - shampoo_beta)
    else:
        row_new = grad_f @ grad_f.T
        state["row_gg"].lerp_(row_new if row_mask is None else row_new * row_mask, 1 - shampoo_beta)
        if state["col_gg"] is not None:
            col_new = grad_f.T @ grad_f
            state["col_gg"].lerp_(col_new if col_mask is None else col_new * col_mask, 1 - shampoo_beta)
    if state["q_row"] is None:
        state["q_row"] = soap_eigenbasis(state["row_gg"])
        state["q_col"] = None if state["col_gg"] is None else soap_eigenbasis(state["col_gg"])
    elif state["soap_step"] > 0 and state["soap_step"] % precondition_frequency == 0:
        state["q_row"], state["q_col"], state["exp_avg_sq"] = soap_basis_qr(
            state["row_gg"], state["col_gg"], state["q_row"], state["q_col"], state["exp_avg_sq"]
        )
    state["soap_step"] += 1

class VOVariants:
    """V/O SOAP variant directions for screening (VO_VARIANTS=1); never applied to the weights.

    Kept up to date every step for the attn.v / attn.proj params this rank owns; on probe steps Muon records each
    variant, after the same attn.proj trust gate and Newton-Schulz as the real path, as a candidate direction:
      vo_ns      the real direction, restricted to V/O (the baseline the others are compared with)
      vo_head    per-head SOAP: V's output side and O's input side use block-diagonal (heads x HEAD_DIM) statistics
      vo_shared  one basis for V's output side and O's input side (the same space), from the sum of the two
                 trace-normalized statistics g_V g_V^T / |g_V|^2 + g_O^T g_O / |g_O|^2
      vo_p25     the real SOAP state with denominator V^(1/4) instead of V^(1/2) (partial whitening)
      vo_gate    the attn.proj trust gate also applied on attn.v (O = the real direction)
    The shadow states (vo_head, vo_shared) run the same SOAP recurrences as the real one, from step 0.
    """
    names = ["vo_ns", "vo_head", "vo_shared", "vo_p25", "vo_gate"]

    def __init__(self, named_params):
        self.kind, self.partner, self.state = {}, {}, {}
        pairs = {}
        for n, p in named_params:
            if n.endswith(".attn.v.weight") or n.endswith(".attn.proj.weight"):
                k = "v" if n.endswith(".attn.v.weight") else "o"
                self.kind[p] = k
                pairs.setdefault(n.split(".attn.")[0], {})[k] = p
        for pr in pairs.values():
            self.partner[pr["v"]], self.partner[pr["o"]] = pr["o"], pr["v"]
        self.mask = None

    @staticmethod
    def _new_state(p):
        return dict(exp_avg_sq=torch.zeros_like(p, dtype=torch.float32),
                    row_gg=torch.zeros(p.size(0), p.size(0), dtype=torch.float32, device=p.device),
                    col_gg=torch.zeros(p.size(1), p.size(1), dtype=torch.float32, device=p.device),
                    q_row=None, q_col=None, soap_step=0)

    @staticmethod
    def _update_stats(st, row_new, col_new):  # soap_update_preconditioner with given statistics
        st["row_gg"].lerp_(row_new, 1 - SOAP_BETA2)
        st["col_gg"].lerp_(col_new, 1 - SOAP_BETA2)
        if st["q_row"] is None:
            st["q_row"], st["q_col"] = soap_eigenbasis(st["row_gg"]), soap_eigenbasis(st["col_gg"])
        elif st["soap_step"] > 0 and st["soap_step"] % SOAP_PRECONDITION_FREQUENCY == 0:
            st["q_row"], st["q_col"], st["exp_avg_sq"] = soap_basis_qr(
                st["row_gg"], st["col_gg"], st["q_row"], st["q_col"], st["exp_avg_sq"])
        st["soap_step"] += 1

    def step(self, p, grad, mom, soap_out, real_state, step_count, capture):
        """Advance the shadow states; on capture steps return {name: pre-Newton-Schulz direction} (vo_ns excluded)."""
        kind = self.kind[p]
        if p not in self.state:
            self.state[p] = dict(head=self._new_state(p), shared=self._new_state(p))
        if self.mask is None:
            h = p.size(0) // HEAD_DIM
            self.mask = torch.block_diag(*[torch.ones(HEAD_DIM, HEAD_DIM, device=p.device)] * h)
        sts = self.state[p]
        dirs = {}
        if capture:
            dirs["vo_head"] = soap_precondition_momentum(mom, sts["head"])
            dirs["vo_shared"] = soap_precondition_momentum(mom, sts["shared"])
            if real_state["q_row"] is None:
                dirs["vo_p25"] = mom
            else:
                q_row, q_col, m = real_state["q_row"], real_state["q_col"], mom.float()
                proj = q_row.T @ m @ q_col
                pre = q_row @ (proj / real_state["exp_avg_sq"].clamp_min(1e-16).pow(0.25)) @ q_col.T
                pre.mul_(gram_frobenius_norm_estimate(m) / gram_frobenius_norm_estimate(pre))
                dirs["vo_p25"] = pre.to(mom.dtype)
            if kind == "o":  # O keeps the real trust gate, applied to each variant
                for k, d in dirs.items():
                    dirs[k] = norm_preserving_blend(mom, d, bounded_trust_gate(trust_gate(mom, d, grad), step_count))
            else:
                dirs["vo_gate"] = norm_preserving_blend(
                    mom, soap_out, bounded_trust_gate(trust_gate(mom, soap_out, grad), step_count))
        else:  # keep the shadow second moments advancing exactly as they would on capture steps
            soap_precondition_momentum(mom, sts["head"])
            soap_precondition_momentum(mom, sts["shared"])
        g = grad.float()
        gp = self.partner[p].grad.float()
        gg_row, gg_col = g @ g.T, g.T @ g
        if kind == "v":  # heads live on V's output side (rows)
            self._update_stats(sts["head"], gg_row * self.mask, gg_col)
            pg = gp.T @ gp  # O's input side = V's output space
            self._update_stats(sts["shared"], gg_row / gg_row.trace().clamp_min(1e-30)
                               + pg / pg.trace().clamp_min(1e-30), gg_col)
        else:  # heads live on O's input side (columns)
            self._update_stats(sts["head"], gg_row, gg_col * self.mask)
            pg = gp @ gp.T  # V's output side
            self._update_stats(sts["shared"], gg_row, gg_col / gg_col.trace().clamp_min(1e-30)
                               + pg / pg.trace().clamp_min(1e-30))
        return dirs


class NewtonPreconditioner:
    """Newton-Muon right preconditioner, one K per layer-input space (q/k/v share theirs)."""
    def __init__(self, model):
        dev, d = next(model.parameters()).device, NEWTON_BLOCK
        self.stats, self.of_param = [], {}
        for block in model.blocks:
            a, m = block.attn, block.mlp
            for src, users in ((a.q, (a.q, a.k, a.v)), (a.proj, (a.proj,)), (m.fc, (m.fc,)), (m.proj, (m.proj,))):
                nb = src.weight.size(1) // d
                st = dict(module=src, n=nb,
                          accum=torch.zeros(nb, d, d, device=dev), count=torch.zeros((), device=dev))
                self.stats.append(st)
                for u in users:
                    self.of_param[u.weight] = st
        n = sum(st["n"] for st in self.stats)
        eye = torch.eye(d, device=dev).repeat(n, 1, 1)
        self.cov = 0.001 * eye
        self.K = eye.clone()
        self.counts = torch.zeros(n, device=dev)
        i = 0
        for st in self.stats:
            st["inv"] = self.K[i:i + st["n"]]
            st["slice"] = slice(i, i + st["n"])
            i += st["n"]
        self.has_inv = False

    def is_refresh_step(self, step):
        return (step + 1) % NEWTON_REFRESH == 0

    def enable_capture(self):
        for st in self.stats:
            st["module"].newton_accum, st["module"].newton_count = st["accum"], st["count"]

    def disable_capture(self):
        for st in self.stats:
            st["module"].newton_accum = st["module"].newton_count = None

    @torch.no_grad()
    def refresh(self):
        # all ranks: Sigma <- lerp(Sigma, mean_batch x x^T, COV_EMA); K = (Sigma + damping)^-1
        batch = torch.cat([st["accum"] for st in self.stats])
        for st in self.stats:
            self.counts[st["slice"]] = st["count"]
        dist.all_reduce(batch, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.counts, op=dist.ReduceOp.SUM)
        batch.div_(self.counts.clamp_min(1).view(-1, 1, 1))
        self.cov.lerp_(batch, (self.counts > 0).float().mul(NEWTON_COV_EMA).view(-1, 1, 1))
        d = self.cov.size(-1)
        damped = self.cov.clone()
        diag = damped.diagonal(dim1=-2, dim2=-1)
        diag.add_((diag.sum(-1) / d * NEWTON_DAMPING + 1e-8).unsqueeze(-1))
        L, info = torch.linalg.cholesky_ex(damped, upper=False, check_errors=False)
        torch.cholesky_inverse(L, upper=False, out=self.K)
        if info.any():
            self.K[info != 0] = torch.eye(d, device=self.K.device)
        # diagnostics: capture counts and how far K is from a scaled identity
        Kn = self.K / self.K.diagonal(dim1=-2, dim2=-1).mean(-1).view(-1, 1, 1)
        off = (Kn - torch.eye(d, device=Kn.device)).norm(dim=(-2, -1)) / d**0.5
        self.last_diag = dict(count_min=self.counts.min().item(), count_max=self.counts.max().item(),
                              K_offident=off.mean().item(), cholesky_fail=int((info != 0).sum().item()))
        for st in self.stats:
            st["accum"].zero_()
            st["count"].zero_()
        self.has_inv = True

    def apply(self, update, p, rescale=True, eps=1e-8):
        # U K (block-diagonal K for wide inputs), optionally rescaled to keep U's Schatten-4 norm
        if not self.has_inv:
            return update
        inv = self.of_param[p]["inv"]
        u = update.float()
        nb, d = inv.size(0), inv.size(-1)
        out = torch.bmm(u.view(u.size(0), nb, d).permute(1, 0, 2), inv).permute(1, 0, 2).reshape_as(u)
        if rescale:
            out.mul_(gram_frobenius_norm_estimate(u, eps=eps) / gram_frobenius_norm_estimate(out, eps=eps))
        return out.to(update.dtype)

def dirs_or_real(pre_ns, real, names):
    """Variant directions after Newton-Schulz; a variant without its own direction here is the real one."""
    return {k: (muon_update(pre_ns[k]) if k in pre_ns else real) for k in names}

def muon_update(update):
    # Newton-Schulz orthogonalization + aspect-ratio scale.
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, named_params, lr=0.02, weight_decay=0, mu=0.95, newton=None):
        assert isinstance(named_params, list) and len(named_params) >= 1
        self.newton = newton
        self.soap_params = {p for n, p in named_params if should_soap_param(n)}
        self.attn_soap_params = {p for n, p in named_params if should_soap_param(n) and is_attn_param(n)}
        self.attn_proj_soap_params = {p for n, p in named_params if should_soap_param(n) and is_attn_proj_param(n)}
        self.v_params = {p for n, p in named_params if n.endswith(".attn.v.weight")}
        self.gated_params = self.attn_proj_soap_params | (self.v_params if V_GATE else set())
        self.vo_params = self.v_params | {p for n, p in named_params if is_attn_proj_param(n)}
        self.head_mask = None
        _kind = lambda n: {"attn.q": "q", "attn.k": "k", "attn.v": "v", "attn.proj": "o", "mlp.fc": "fc",
                           "mlp.proj": "proj"}.get(n.split(".", 1)[1].rsplit(".", 1)[0])
        self.row_gain_params = {p for n, p in named_params if _kind(n) in ROW_GAIN_SCOPE} if ROW_GAIN_LR > 0 else set()
        self.step_count = 0
        # Probe diagnostics (harness.ProbeQuadratic): on probe steps the harness sets probe_capture and this optimizer
        # records, per param it owns, the candidate update directions below (descent convention, W -= lr*u).
        # A new direction idea ("variant") is screened by adding its name here and a self._capture call.
        self.probe_capture, self.probe_dirs = False, {}
        self.probe_dir_names = ["momentum", "soap"] + (["newton"] if newton is not None and NEWTON_MODE != "grad" else []) + ["ns"]
        self.vo = VOVariants(named_params) if VO_VARIANTS else None
        if self.vo is not None:
            self.probe_dir_names += VOVariants.names
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        if all(m == 1.0 for m in MUON_LR_MULT.values()) and VO_LR_MULT == 1.0 and VO_FLOOR_MULT == 1.0:
            params = sorted([p for _, p in named_params], key=lambda x: x.size(), reverse=True)
        else:
            by_type = {}
            for n, p in named_params:
                kind = "attn_proj" if is_attn_proj_param(n) else "qkv" if is_attn_param(n) else "mlp"
                is_vo = is_attn_proj_param(n) or n.endswith(".attn.v.weight")
                key = (MUON_LR_MULT[kind] * (VO_LR_MULT if is_vo else 1.0), VO_FLOOR_MULT if is_vo else 1.0)
                by_type.setdefault(key, []).append(p)
            params = [dict(params=sorted(ps, key=lambda x: x.size(), reverse=True),
                           lr=lr * lr_mult, lr_mult=lr_mult, floor_mult=floor_mult)
                      for (lr_mult, floor_mult), ps in by_type.items()]
        super().__init__(params, defaults)

    def in_scope(self, p, scope):
        return scope == "all" or (scope == "vo" and p in self.vo_params)

    def sink_of(self, p):
        return SOAP_SINK if SOAP_SINK and self.in_scope(p, SOAP_SINK_SCOPE) else ""

    def _capture(self, p, name, u):
        if self.probe_capture:
            self.probe_dirs.setdefault(p, {})[name] = u.detach().float().clone()

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if self.newton is not None and self.newton.is_refresh_step(self.step_count):
            self.newton.refresh()
            print0(f"newton refresh step:{self.step_count} " + " ".join(f"{k}:{v:.4g}" for k, v in self.newton.last_diag.items()), console=True)
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                        if p in self.soap_params:
                            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                            state["row_gg"] = torch.zeros(p.size(0), p.size(0), dtype=torch.float32, device=p.device)
                            state["col_gg"] = (None if SOAP_ONE_SIDED else
                                               torch.zeros(p.size(1), p.size(1), dtype=torch.float32, device=p.device))
                            state["q_row"] = None
                            state["q_col"] = None
                            state["soap_step"] = 0
                    grad = p.grad
                    if self.newton is not None and NEWTON_MODE == "grad":
                        grad = self.newton.apply(grad, p, rescale=False)
                    state["momentum"].lerp_(grad, 1 - group["mu"])
                    momentum_update = grad.lerp(state["momentum"], group["mu"])
                    self._capture(p, "momentum", momentum_update)
                    is_attn_soap = p in self.attn_soap_params
                    vo_dirs = None
                    use_soap = p in self.soap_params
                    if use_soap:
                        if is_attn_soap:
                            soap_update = soap_precondition_momentum(
                                momentum_update, state,
                                denom_power=VO_DENOM_POWER if p in self.vo_params else SOAP_DENOM_POWER,
                                sink=self.sink_of(p))
                            self._capture(p, "soap", soap_update)
                            if self.newton is not None and NEWTON_MODE in ("soap", "right"):
                                soap_update = self.newton.apply(soap_update, p)
                                self._capture(p, "newton", soap_update)
                            if p in self.gated_params:
                                gate = bounded_trust_gate(
                                    trust_gate(momentum_update, soap_update, grad),
                                    self.step_count
                                )
                            else:
                                gate = torch.ones((), dtype=torch.float32, device=p.device)
                            vo_dirs = (self.vo.step(p, grad, momentum_update, soap_update, state, self.step_count,
                                                    self.probe_capture) if self.vo is not None and p in self.vo.kind else None)
                            momentum_update = norm_preserving_blend(momentum_update, soap_update, gate)
                        else:
                            momentum_update = soap_precondition_momentum(momentum_update, state, sink=self.sink_of(p))
                            self._capture(p, "soap", momentum_update)
                            if self.newton is not None and NEWTON_MODE in ("soap", "right"):
                                momentum_update = self.newton.apply(momentum_update, p)
                                self._capture(p, "newton", momentum_update)
                    update = muon_update(momentum_update)
                    self._capture(p, "ns", update)
                    if self.probe_capture and vo_dirs is not None:
                        self._capture(p, "vo_ns", update)
                        for k, d in dirs_or_real(vo_dirs, update, VOVariants.names[1:]).items():
                            self._capture(p, k, d)
                    update = scale_radial_update(update, p)
                    # u/w-floor. SOAP and non-SOAP params can use different floors.
                    p_fro = p.float().norm().clamp_min(1e-8)
                    u_fro = update.float().norm().clamp_min(1e-8)
                    cur_uw = u_fro / p_fro
                    target_uw = (SOAP_TARGET_UW if use_soap else NONSOAP_TARGET_UW) * group.get("floor_mult", 1.0)
                    if ROWFLOOR and p.ndim == 2:
                        # (B) RowFloor: boost each under-updated OUTPUT ROW to its target update/weight
                        # ratio. Per-row SHAPE change; magnitude is re-pinned below, so only the shape
                        # survives the radius pin. RHO=1.0 -> the clamp is the whole floor.
                        r_row = p.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
                        s_row = update.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
                        f_row = torch.clamp(target_uw * r_row / s_row, min=1.0).pow(ROWFLOOR_RHO)
                        update = (update.float() * f_row).to(update.dtype)
                    else:
                        scale = torch.where(cur_uw < target_uw, target_uw * p_fro / u_fro, torch.ones_like(p_fro))
                        update = update * scale.to(update.dtype)
                    gain = p in self.row_gain_params
                    row_ctl = (ROW_PIN or gain) and p.ndim == 2
                    if row_ctl:
                        p_before = p.float()
                        r_rows = p_before.norm(dim=1, keepdim=True).clamp_min(1e-8)
                        if gain and ROW_GAIN_MODE == "matrix":
                            r_rows = p_before.norm().clamp_min(1e-8).reshape(1, 1)
                        target_rows = r_rows if gain else (
                            r_rows - group["lr"] * (update.float() * p_before).sum(dim=1, keepdim=True) / r_rows)
                        target_rows = target_rows.clamp_min(1e-8)
                    else:
                        target_radius = target_radius_after_update(p, update, group["lr"])
                    # WD set to 0 — u/w target replaces wd's role (smaller updates as p grows).
                    if CWD > 0.0 and p.ndim == 2:
                        # (C) Cautious Weight Decay: mask the coords where -lr*update already shrinks |p|
                        # (update*p>0); decay only those (never fight a coord the optimizer wants to grow).
                        cwd_mask = (update.float() * p.float() > 0).to(p.dtype)
                    p.add_(update, alpha=-group["lr"])
                    if row_ctl:
                        cur = (p.float().norm().reshape(1, 1) if gain and ROW_GAIN_MODE == "matrix"
                               else p.float().norm(dim=1, keepdim=True))
                        p.mul_((target_rows / cur.clamp_min(1e-8)).to(p.dtype))
                        if gain:
                            g_row = ((grad.float() * p_before).sum().reshape(1, 1) if ROW_GAIN_MODE == "matrix"
                                     else (grad.float() * p_before).sum(dim=1, keepdim=True)) / r_rows
                            if "row_m" not in state:
                                state["row_m"], state["row_v"], state["row_t"] = torch.zeros_like(g_row), torch.zeros_like(g_row), 0
                            state["row_t"] += 1
                            state["row_m"].lerp_(g_row, 0.1)
                            state["row_v"].lerp_(g_row.square(), 0.05)
                            adam = (state["row_m"] / (1 - 0.9 ** state["row_t"])) / (
                                (state["row_v"] / (1 - 0.95 ** state["row_t"])).sqrt() + 1e-12)
                            p.mul_((1 - ROW_GAIN_LR * group["lr"] / group["initial_lr"] * adam).to(p.dtype))
                    else:
                        rescale_to_radius(p, target_radius)
                    if CWD > 0.0 and p.ndim == 2:
                        # apply AFTER the radius pin (POST) so the per-coord shape change is preserved.
                        p.mul_(1.0 - (group["lr"] * CWD) * cwd_mask)
                    if use_soap:
                        if VO_HEAD and p in self.vo_params:
                            if self.head_mask is None:
                                h = p.size(0) // HEAD_DIM
                                self.head_mask = torch.block_diag(*[torch.ones(HEAD_DIM, HEAD_DIM, device=p.device)] * h)
                            # heads live on V's output side (rows) and O's input side (columns)
                            soap_update_preconditioner(grad, state, row_mask=self.head_mask if p in self.v_params else None,
                                                       col_mask=None if p in self.v_params else self.head_mask)
                        else:
                            soap_update_preconditioner(grad, state, kl=SOAP_KL and self.in_scope(p, SOAP_KL_SCOPE))
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self.step_count += 1


class Adam(torch.optim.Optimizer):
    """Bias-correction-free Adam for the aux (<2D) param groups.

    This is exactly the SOTA's CenterShrinkAdam with center_shrink (rho) = 1.0:
    at rho=1.0 the center-shrink collapses to the identity (mixed == adam_dir and
    the rms-rescale ratio is 1), leaving plain Adam with raw (uncorrected) EMAs.
    NOTE: this intentionally does NOT apply the (1-beta^t) bias correction, so it
    is NOT the same as torch.optim.AdamW. It matches the source under
    AUX_CENTER_SHRINK=1.0 bit-for-bit.
    """
    def __init__(self, params, lr=0.01, betas=(0.8, 0.99), eps=1e-10):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.lerp_(grad.float().square(), 1 - beta2)
                adam_dir = exp_avg.float() / (exp_avg_sq.sqrt() + eps)
                p.add_(adam_dir.to(p.dtype), alpha=-group["lr"])


class EMA_Nesterov(torch.optim.Optimizer):
    def __init__(self, params, inner_optimizer, lookahead_stepsize=0, use_scheduled_lookahead_stepsize=True, lookahead_ema=0.9, prefill_steps=0, rest_steps=0):
        if lookahead_stepsize < 0.0:
            raise ValueError("Invalid momentum value: {}".format(lookahead_stepsize))

        super(EMA_Nesterov, self).__init__(params, {})
        self.inner_optimizer = inner_optimizer
        self.use_scheduled_lookahead_stepsize = use_scheduled_lookahead_stepsize
        self.lookahead_stepsize = lookahead_stepsize
        self.lookahead_ema = lookahead_ema
        self.prefill_steps = prefill_steps
        self.rest_steps = rest_steps
        self.it = 0
        self.lookahead_status = False
        self.current_lookahead_stepsize = 0
        self.initialize_buffers()

    def __setstate__(self, state):
        super(EMA_Nesterov, self).__setstate__(state)

    @torch.no_grad()
    def initialize_buffers(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'prev_params' not in param_state:
                    param_state['prev_params'] = (p.clone(), self.it)

                if 'lookahead_buffer' not in param_state:
                    param_state['lookahead_buffer'] = (torch.zeros_like(p), -1)

    def get_lr_lambda(self):
        if isinstance(self.inner_optimizer, list):
            lr_lambda = self.inner_optimizer[0].param_groups[0]["lr"] / self.inner_optimizer[0].param_groups[0]["initial_lr"]
        else:
            lr_lambda = self.inner_optimizer.param_groups[0]["lr"] / self.inner_optimizer.param_groups[0]["initial_lr"]
        return lr_lambda

    @torch.no_grad()
    def lookahead_step(self):
        """Performs nesterov's lookahead."""
        if self.use_scheduled_lookahead_stepsize:
            lookahead_stepsize = self.lookahead_stepsize * self.get_lr_lambda()
        else:
            lookahead_stepsize = self.lookahead_stepsize
        self.current_lookahead_stepsize = lookahead_stepsize

        for group in self.param_groups:
            for p in group['params']:

                param_state = self.state[p]

                lookahead = param_state['lookahead_buffer'][0]

                p.add_(lookahead, alpha=lookahead_stepsize)



    @torch.no_grad()
    def accum_lookahead(self):
        """Update nesterov's lookahead direction."""
        lookahead_ema = self.lookahead_ema
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                look = p.add(param_state['prev_params'][0], alpha=-1)

                # update lookahead buffer
                buf = param_state['lookahead_buffer'][0]

                param_state['lookahead_buffer'] = (buf.lerp_(look, 1 - lookahead_ema), self.it) # m^{t+1} = beta * m^t + (1-beta) * look

                # update prev_params buffer
                param_state['prev_params'] = (param_state['prev_params'][0].copy_(p), self.it)


    @torch.no_grad()
    def nesterov_step(self):
        if self.it + 1 > self.prefill_steps and self.it < self.rest_steps and not self.lookahead_status:
            self.lookahead_step()
        else:
            self.current_lookahead_stepsize = 0
        self.lookahead_status = True

    @torch.no_grad()
    def step(self):
        if not self.lookahead_status:
            raise ValueError("optimizer.nesterov_step() should be invoked before model forward pass.")
        if isinstance(self.inner_optimizer, list):
            for opt in self.inner_optimizer:
                opt.step()
        else:
            self.inner_optimizer.step()

        self.accum_lookahead()

        self.lookahead_status = False
        self.it += 1


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
harness.setup(device)
torch.manual_seed(SEED)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
if dist.get_rank() == 0:
    log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = str(log_dir / f"{uuid.uuid4()}.txt")
    print(logfile, flush=True)

def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s, flush=True)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running on device_name={torch.cuda.get_device_name(device)} with world_size={dist.get_world_size()}")
print0(f"Run UTC timestamp={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
print0(f"Using seed={SEED}")
print0(f"Experiment={EXPERIMENT_NAME}")
print0(f"Intuition={EXPERIMENT_INTUITION}")
print0("SOAP runs on all all_hidden matrices for the whole run at frequency 1.")
print0(f"Schedule uses hardcoded PR 287 cooldown constants with train_steps={FINAL_TRAIN_STEPS}, schedule_steps={FINAL_SCHEDULE_STEPS}.")
print0(f"Using newton_muon={NEWTON_MUON} mode={NEWTON_MODE} refresh={NEWTON_REFRESH} cov_ema={NEWTON_COV_EMA} damping={NEWTON_DAMPING} block={NEWTON_BLOCK} muon_lr_mult={MUON_LR_MULT}")
print0(f"Using vo_lr_mult={VO_LR_MULT} vo_floor_mult={VO_FLOOR_MULT} vo_variants={VO_VARIANTS} "
       f"v_gate={V_GATE} vo_denom_power={VO_DENOM_POWER} vo_head={VO_HEAD} row_pin={ROW_PIN} row_gain_lr={ROW_GAIN_LR} "
       f"row_gain_scope={sorted(ROW_GAIN_SCOPE)} row_gain_mode={ROW_GAIN_MODE} "
       f"soap_sink={SOAP_SINK}/{SOAP_SINK_SCOPE} soap_kl={SOAP_KL}/{SOAP_KL_SCOPE} ena_rest={ENA_REST}")
print0(f"Using 1-D Adam gain_beta2={GAIN_BETA2} aux_beta1={AUX_BETA1} aux_lr_mult={AUX_LR_MULT}")
print0(f"Using mu={MU}")
print0(f"Using muon_lr={MUON_LR}")
print0(f"Using target_uw={TARGET_UW}")
print0(f"Using soap_target_uw={SOAP_TARGET_UW}")
print0(f"Using nonsoap_target_uw={NONSOAP_TARGET_UW}")
print0(f"Using soap_beta2={SOAP_BETA2}")
print0(f"Using soap_precondition_frequency={SOAP_PRECONDITION_FREQUENCY}")
print0(f"Using soap_denom_power={SOAP_DENOM_POWER}")
print0(f"Using attn_early_trust_floor={ATTN_EARLY_TRUST_FLOOR}")
print0(f"Using attn_early_trust_cap={ATTN_EARLY_TRUST_CAP}")
print0(f"Using attn_trust_floor_end_step={ATTN_TRUST_FLOOR_END_STEP}")
print0(f"Using attn_trust_floor_fade_end_step={ATTN_TRUST_FLOOR_FADE_END_STEP}")
print0(f"Using attn_trust_min_agree={ATTN_TRUST_MIN_AGREE}")
print0(f"Using attn_trust_min_grad_align={ATTN_TRUST_MIN_GRAD_ALIGN}")
print0(f"Using attn_trust_power={ATTN_TRUST_POWER}")
print0(f"Using radial_outward_scale={RADIAL_OUTWARD_SCALE}")
print0(f"Using radial_inward_scale={RADIAL_INWARD_SCALE}")
print0("Dampened radial gradient component is applied before the u/w floor; post-step radius is corrected to remove tangent drift.")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size, skip=RESUME_STEP)
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))
# Held-out training data for the diagnostics (harness: ProbeQuadratic, GradNoiseScale): the last train shard, which
# training never reaches. Its first batch is the FIXED probe batch; later batches are fresh GNS batches.
_train_files = sorted(Path.cwd().glob("data/fineweb10B/fineweb_train_*.bin"))
assert FINAL_TRAIN_STEPS * batch_size < sum((f.stat().st_size - 1024) // 2 for f in _train_files[:-1])
heldout_loader = distributed_data_generator(str(_train_files[-1].relative_to(Path.cwd())), batch_size)
probe_inputs, probe_targets = next(heldout_loader)

def probe_loss_fn():
    return model(probe_inputs, probe_targets) / probe_targets.numel()

_heldout_n = 0   # GNS batches drawn so far (restored on resume, so a fork sees the same batches)

def gns_loss_fn(i):
    global _heldout_n
    _heldout_n += 1
    inputs, targets = next(heldout_loader)
    return model(inputs, targets) / targets.numel(), targets.numel()

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


########################################
#       Init & Optim Hyperparams       #
########################################

# we want to minimize this while still reaching 3.28 val loss
train_steps = FINAL_TRAIN_STEPS
# Speed: stop the loop early (trajectory-preserving -- schedule constants stay at FINAL_TRAIN_STEPS,
# so evals at steps <= STOP_STEP are IDENTICAL to the full run). Target boundary ~2775 only needs evals
# up to ~2800-2850, so STOP_STEP=2850 saves the unused 2850-2900 tail. Default = full run.
STOP_STEP = int(os.environ.get("STOP_STEP", FINAL_TRAIN_STEPS))   # default: full schedule; set env to early-stop
val_regular_interval = 125
_dv = [int(x) for x in DENSE_VAL.split(":")] if DENSE_VAL else None
extra_val_steps = set(range(_dv[0], _dv[1] + 1, _dv[2])) if _dv else set()
extra_val_steps |= {2700, 2705, 2710, 2715, 2720, 2725, 2730, 2735, 2740, 2745, 2750, 2755, 2760, 2765, 2770, 2775, 2780, 2785, 2790, 2795, 2800, 2805, 2810, 2820, 2830, 2840, 2850, 2860, 2870, 2880, 2890, 2895, 2900, 2910, 2920, 2930, 2940, 2950, 2960, 2965, 2970, 2975, 2980, 2985, 2990, 2995, 2999, 3000, 3010, 3020}

# initialize model parameters
for name, p in model.named_parameters():
    if "proj" in name:
        p.data.zero_()

# v44: v13 depth-scaled mlp.fc init alpha=0.30 (ported from opus v15)
_DI_FC_ALPHA = 0.30
_NUM_BLOCKS = len(model.blocks)
with torch.no_grad():
    for l_idx, block in enumerate(model.blocks):
        ramp = l_idx / (_NUM_BLOCKS - 1) if _NUM_BLOCKS > 1 else 0.0
        s_l = 1.0 - _DI_FC_ALPHA * ramp
        block.mlp.fc.weight.data.mul_(s_l)

# v44: v14 CGI Rademacher channel-gain split alpha=0.14 (ported from opus v15)
_CGI_ALPHA = 0.125
_CGI_PAIR_FROM_LAYER = 6
_CGI_HEAD_MEAN_SHRINK = 0.50

def headmean_antithetic_pair(shape, *, device, layer_idx: int, seed: int, head_dim: int = HEAD_DIM):
    width = shape[-1]
    if width % head_dim != 0:
        s0 = (torch.randint(0, 2, shape, device=device, dtype=torch.float32) * 2 - 1)
        return s0, -s0
    gen = torch.Generator(device=device)
    gen.manual_seed(0xA9170000 + seed * 1009 + layer_idx * 9176)
    heads0 = []
    heads1 = []
    for _ in range(width // head_dim):
        h0 = (torch.randint(0, 2, (head_dim,), device=device, generator=gen, dtype=torch.int64) * 2 - 1).float()
        plus0 = int((h0 > 0).sum().item())
        plus0 = int(round(head_dim / 2 + _CGI_HEAD_MEAN_SHRINK * (plus0 - head_dim / 2)))
        plus0 = max(0, min(head_dim, plus0))
        h0 = torch.cat([
            torch.ones(plus0, device=device, dtype=torch.float32),
            -torch.ones(head_dim - plus0, device=device, dtype=torch.float32),
        ])
        h0 = h0[torch.randperm(head_dim, device=device, generator=gen)]
        plus1 = head_dim - plus0
        h1 = torch.cat([
            torch.ones(plus1, device=device, dtype=torch.float32),
            -torch.ones(head_dim - plus1, device=device, dtype=torch.float32),
        ])
        h1 = h1[torch.randperm(head_dim, device=device, generator=gen)]
        heads0.append(h0)
        heads1.append(h1)
    return torch.cat(heads0).reshape(shape), torch.cat(heads1).reshape(shape)

with torch.no_grad():
    pair_next = None
    for l_idx, block in enumerate(model.blocks):
        if l_idx >= _CGI_PAIR_FROM_LAYER and (l_idx - _CGI_PAIR_FROM_LAYER) % 2 == 1:
            s = pair_next
            pair_next = None
        elif l_idx >= _CGI_PAIR_FROM_LAYER:
            s, pair_next = headmean_antithetic_pair(
                block.norm1.gains.shape,
                device=block.norm1.gains.device,
                layer_idx=l_idx,
                seed=SEED,
            )
        else:
            s = (torch.randint(0, 2, block.norm1.gains.shape,
                               device=block.norm1.gains.device, dtype=torch.float32) * 2 - 1)
        block.norm1.gains.data.copy_((1.0 - _CGI_ALPHA * s).to(block.norm1.gains.dtype))
        block.norm2.gains.data.copy_((1.0 + _CGI_ALPHA * s).to(block.norm2.gains.dtype))

# create the optimizer(s)
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight], lr=1/320)],
                   betas=(0.8, 0.99), eps=1e-10, weight_decay=0, fused=True)
gain_aux_params = [p for n, p in model.named_parameters() if p.ndim < 2 and n.endswith(".gains")]
attn_proj_bias_params = [p for n, p in model.named_parameters() if n.endswith(".attn.proj.bias")]
other_aux_params = [p for n, p in model.named_parameters()
                    if p.ndim < 2 and not n.endswith(".gains")
                    and not n.endswith(".attn.proj.bias")]
# Aux (<2D) param groups. CenterShrink with rho=1.0 is plain (bias-correction-free)
# Adam, so this is the minimal `Adam` class above with the per-group betas preserved.
# NOTE: we do NOT use torch.optim.AdamW here because AdamW applies (1-beta^t) bias
# correction, which CenterShrinkAdam does not -- they are not equivalent.
optimizer3 = Adam([
        dict(params=gain_aux_params, lr=0.01 * AUX_LR_MULT, betas=(AUX_BETA1, GAIN_BETA2)),
        dict(params=other_aux_params, lr=0.01 * AUX_LR_MULT, betas=(AUX_BETA1, 0.997)),
        dict(params=attn_proj_bias_params, lr=0.01 * AUX_LR_MULT, betas=(AUX_BETA1, 0.9965)),
    ],
    lr=0.01, betas=(0.8, 0.99), eps=1e-10)
# Skylight-001: NorMuon-lite (per-row variance) + u/w-floor + lr=0.0375.
newton = NewtonPreconditioner(model) if NEWTON_MUON else None
optimizer2 = Muon([(n, p) for n, p in model.blocks.named_parameters() if p.ndim >= 2],
                  lr=MUON_LR, mu=MU, newton=newton)
optimizers = [optimizer1, optimizer2, optimizer3]
optimizers = [EMA_Nesterov(
        [p for p in model.parameters()],
        optimizers,
        lookahead_stepsize=0.3,
        use_scheduled_lookahead_stepsize=True,
        lookahead_ema=0.99,
        prefill_steps=300,
        rest_steps=ENA_REST or train_steps - 950,
    )]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers[0].inner_optimizer:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
optimizer1.param_groups[0]["power_c"] = ADAM_EMBED_POWER_C
optimizer1.param_groups[1]["power_c"] = ADAM_PROJ_POWER_C
for group in optimizer3.param_groups:
    group["power_c"] = ADAM_OTHER_POWER_C * AUX_LR_MULT
for group in optimizer2.param_groups:
    group["power_c"] = MUON_POWER_C * group.get("lr_mult", 1.0)

# learning rate schedule: stable then decay
def _lr(step, initial_lr, power_c, power=1.0):
    t_end = FINAL_SCHEDULE_STEPS
    flat_lr = initial_lr
    downward_lr = power_c * max(0.0, t_end - step) ** power
    return min(flat_lr, downward_lr)

# v49: v15 Muon mu schedule (warmup 0.85->0.95 over 300 steps, cooldown 0.95->0.85 over last 50)
_MU_MIN = 0.85
_MU_MAX = 0.95
_MU_WARMUP_STEPS = 300
# Muon-momentum cooldown horizon (mu 0.95->0.85 over the last N steps).
_MU_COOLDOWN_STEPS = 200   # tuned (mu-cooldown extended; original 100)

def _muon_mu_at_step(step, train_steps):
    cd_start = train_steps - _MU_COOLDOWN_STEPS
    if step < _MU_WARMUP_STEPS:
        frac = step / max(_MU_WARMUP_STEPS, 1)
        return _MU_MIN + frac * (_MU_MAX - _MU_MIN)
    elif step > cd_start:
        frac = (step - cd_start) / max(_MU_COOLDOWN_STEPS, 1)
        return _MU_MAX - frac * (_MU_MAX - _MU_MIN)
    else:
        return _MU_MAX

def set_hparams(step):
    progress = step / FINAL_SCHEDULE_STEPS
    # progress >= 1 is allowed when t_end (FINAL_SCHEDULE_STEPS) < train end: LR is clamped to 0
    # by _lr's max(0, t_end-step) for those tail steps (the harmless frozen coast).
    assert 0 <= progress
    mu = _muon_mu_at_step(step, FINAL_TRAIN_STEPS)
    for opt in optimizers[0].inner_optimizer:
        for group in opt.param_groups:
            group["lr"] = _lr(step, group["initial_lr"], group["power_c"], FINAL_LR_POWER)
    for group in optimizer2.param_groups:
        group["mu"] = mu




########################################
#        Training and Validation       #
########################################

for p in model.parameters():
    dist.broadcast(p.detach(), 0)
# start the clock
training_time = 0
_tailema = None   # (A) Tail-EMA buffer: one fp32 tensor per non-embed param; lazy-init at TAILEMA_START
dist.barrier()
t0 = time.perf_counter()
H = harness.Harness(model, optimizers, logfile if dist.get_rank() == 0 else None,
                    probe_loss_fn=probe_loss_fn, gns_loss_fn=gns_loss_fn, config=dict(train_steps=train_steps, newton_muon=NEWTON_MUON, newton_mode=NEWTON_MODE,
                                   newton_refresh=NEWTON_REFRESH, newton_damping=NEWTON_DAMPING,
                                   muon_lr_mult=MUON_LR_MULT, gain_beta2=GAIN_BETA2,
                                   aux_beta1=AUX_BETA1, aux_lr_mult=AUX_LR_MULT,
                                   vo_lr_mult=VO_LR_MULT, vo_floor_mult=VO_FLOOR_MULT,
                                   v_gate=V_GATE, vo_denom_power=VO_DENOM_POWER, vo_head=VO_HEAD))

pname = {p: n for n, p in model.named_parameters()}
p_of = {n: p for n, p in model.named_parameters()}
_ckpt_env = ("ROW_GAIN_SCOPE", "ROW_GAIN_MODE", "SOAP_SINK", "SOAP_KL", "ENA_REST", "ROW_PIN", "ROW_GAIN_LR", "NEWTON_MUON", "GAIN_BETA2", "AUX_BETA1", "AUX_LR_MULT", "MUON_LR_MULT", "VO_LR_MULT", "VO_FLOOR_MULT",
             "V_GATE", "VO_DENOM_POWER", "VO_HEAD")

def save_checkpoint(step):
    assert not NEWTON_MUON, "checkpoints do not cover the Newton-Muon state"
    d = Path(CKPT_DIR) / f"step{step}"
    d.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    by_name = lambda opt: {pname[p]: st for p, st in opt.state.items()}
    torch.save({"muon": by_name(optimizer2)}, d / f"muon_rank{rank}.pt.tmp")
    os.replace(d / f"muon_rank{rank}.pt.tmp", d / f"muon_rank{rank}.pt")
    if rank == 0:
        ema = optimizers[0]
        torch.save(dict(step=step, model=model.state_dict(), adamw=by_name(optimizer1), adam=by_name(optimizer3),
                        ema_nesterov=dict(state=by_name(ema), it=ema.it), muon_step_count=optimizer2.step_count,
                        tailema=_tailema, training_time=training_time, heldout_n=_heldout_n,
                        rng=dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())), d / "common.pt.tmp")
        os.replace(d / "common.pt.tmp", d / "common.pt")
        json.dump(dict(step=step, seed=SEED, world_size=dist.get_world_size(), commit=os.environ.get("RECORD_COMMIT"),
                       run=os.environ.get("WANDB_RUN_NAME"), env={k: os.environ.get(k) for k in _ckpt_env}),
                  open(d / "meta.json", "w"), indent=1)
    dist.barrier()
    print0(f"checkpoint step:{step} -> {d}", console=True)

def load_checkpoint(path):
    global _tailema, training_time, _heldout_n
    assert not NEWTON_MUON, "checkpoints do not cover the Newton-Muon state"
    meta = json.load(open(f"{path}/meta.json"))
    assert meta["world_size"] == dist.get_world_size()
    ck = torch.load(f"{path}/common.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    for opt, key in ((optimizer1, "adamw"), (optimizer3, "adam"), (optimizers[0], "ema_nesterov")):
        states = ck[key]["state"] if key == "ema_nesterov" else ck[key]
        assert set(states) == {pname[p] for g in opt.param_groups for p in g["params"]}, key
        for n, st in states.items():
            opt.state[p_of[n]] = st
    optimizers[0].it = ck["ema_nesterov"]["it"]
    optimizer2.step_count = ck["muon_step_count"]
    # Muon states are sharded by rank; read every shard and keep the params this rank owns now (ownership can change
    # when a knob regroups the params)
    ws, rank = dist.get_world_size(), dist.get_rank()
    owned = {p for g in optimizer2.param_groups for i, p in enumerate(g["params"]) if i % ws == rank}
    n_loaded = 0
    for r in range(meta["world_size"]):
        for n, st in torch.load(f"{path}/muon_rank{r}.pt", map_location=device, weights_only=False)["muon"].items():
            if p_of[n] in owned:
                optimizer2.state[p_of[n]] = st
                n_loaded += 1
    assert n_loaded == len(owned), (n_loaded, len(owned))
    _tailema, training_time = ck["tailema"], ck["training_time"]
    for _ in range(ck["heldout_n"]):
        next(heldout_loader)
    _heldout_n = ck["heldout_n"]
    torch.set_rng_state(ck["rng"]["cpu"].cpu())
    torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
    H.t = meta["step"]
    if H.dyn is not None:
        H.dyn.t = meta["step"]
    print0(f"resumed from {path} (step {meta['step']}, parent run {meta['run']}, parent env {meta['env']})", console=True)

_lab_ref, _lab_ema, _lab_rre = {}, {}, {}
_lab_ho = None   # held-out eval set (inputs, targets), built lazily

@torch.no_grad()
def lab_track(t):
    """After step t-1 (weights = after t steps): reference snapshots, extra EMAs, RRE iterates."""
    params = list(model.parameters())
    if t in LAB_REF_STEPS:
        _lab_ref[t] = [p.detach().float().clone() for p in params]
    for start, tau in LAB_EMAS:
        if start <= t:
            e = _lab_ema.get((start, tau))
            if e is None:
                _lab_ema[(start, tau)] = [None if p is model.embed.weight else p.detach().float().clone() for p in params]
            else:
                for ej, p in zip(e, params):
                    if ej is not None:
                        ej.add_(p.detach().float() - ej, alpha=1.0 / tau)
    if STOP_STEP - t >= 0 and (STOP_STEP - t) % LAB_RRE_EVERY == 0 and (STOP_STEP - t) // LAB_RRE_EVERY <= LAB_RRE_K:
        _lab_rre[t] = [p.detach().float().clone() for p in params]

@torch.no_grad()
def lab_eval(inputs, targets):
    loss = 0
    for i in range(len(inputs) // mbs):
        loss += model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    return (loss / val_tokens).item()

def rre_extrapolate(xs, damping, max_rel):
    """Reduced-rank extrapolation from iterates xs[0..k] (lists of tensors): c = argmin |sum c_i u_i|, sum c = 1,
    u_i = x_{i+1} - x_i; x* = sum c_i x_i; returns x_k + damping * (x* - x_k), the step capped at max_rel * |x_k|."""
    k = len(xs) - 1
    us = [[b - a for a, b in zip(xs[i], xs[i + 1])] for i in range(k)]
    G = torch.tensor([[sum((a * b).sum().item() for a, b in zip(us[i], us[j])) for j in range(k)] for i in range(k)],
                     dtype=torch.float64)
    G += torch.eye(k, dtype=torch.float64) * G.diagonal().mean() * 1e-8
    c = torch.linalg.solve(G, torch.ones(k, dtype=torch.float64))
    c = (c / c.sum()).tolist()
    xstar = [sum(ci * xs[i][j] for i, ci in enumerate(c)) for j in range(len(xs[0]))]
    delta = [damping * (a - b) for a, b in zip(xstar, xs[-1])]
    dn = sum(d.square().sum().item() for d in delta) ** 0.5
    xn = sum(x.square().sum().item() for x in xs[-1]) ** 0.5
    scale = min(1.0, max_rel * xn / max(dn, 1e-30)) if max_rel else 1.0
    return [d * scale for d in delta], c

@torch.no_grad()
def run_readout_lab(step):
    """Score readouts theta_eval = f(theta, EMAs, references, RRE) on val and held-out data; weights are restored."""
    global _lab_ho
    if _lab_ho is None:
        _lab_ho = next(distributed_data_generator(str(_train_files[-1].relative_to(Path.cwd())), val_tokens, skip=5))
    model.eval()
    params = list(model.parameters())
    theta = [p.detach().float().clone() for p in params]
    mix = lambda e, lam: [t if ej is None else (1 - lam) * t + lam * ej for t, ej in zip(theta, e)]
    results, heldout = [], []

    def score(name, ws):
        for p, w in zip(params, ws):
            p.copy_(w.to(p.dtype))
        results.append((name, lab_eval(val_inputs, val_targets)))
        heldout.append((name, lab_eval(*_lab_ho)))
        print0(f"readout step:{step} {name} val:{results[-1][1]:.5f} heldout:{heldout[-1][1]:.5f}", console=True)

    tail = _tailema if _tailema is not None else [None] * len(theta)
    for lam in (0.0, 0.5, 0.6, 0.7):
        score(f"tailema lam={lam}", mix(tail, lam))
    base_t = mix(tail, TAILEMA_LAMBDA)
    for (start, tau), e in sorted(_lab_ema.items()):
        for lam in (0.5, 0.6, 0.7, 0.8):
            score(f"ema{start}:{tau} lam={lam}", mix(e, lam))
    for r, ref in sorted(_lab_ref.items()):
        for g in (-0.1, -0.075, -0.05, -0.025):
            score(f"tailema{TAILEMA_LAMBDA}+ref{r} gamma={g}", [b + g * (b - q) for b, q in zip(base_t, ref)])
    if len(_lab_rre) == LAB_RRE_K + 1:
        xs = [v for _, v in sorted(_lab_rre.items())]
        for damping, cap in ((0.875, 0.001), (0.5, 0.001), (0.875, 0.0)):
            delta, c = rre_extrapolate(xs, damping, cap)
            print0(f"readout rre damping={damping} cap={cap} coeffs={[round(x, 3) for x in c]}", console=True)
            for base_name, base in (("raw", theta), (f"tailema{TAILEMA_LAMBDA}", base_t)):
                score(f"{base_name}+rre d={damping} cap={cap}", [b + d for b, d in zip(base, delta)])
    for p, t in zip(params, theta):
        p.copy_(t.to(p.dtype))
    if dist.get_rank() == 0:
        with open(logfile.replace(".txt", ".readout.json"), "w") as f:
            json.dump(dict(step=step, results=results, heldout=heldout), f)
    model.train()

_rowstat_prev = {}

@torch.no_grad()
def log_rowstats(step):
    """Row-norm statistics of the hidden matrices per type (rank 0, at validation steps) -> <log>.rowstat.jsonl:
    mean and CV (std/mean) of the output-row norms, and the mean |d log row norm| per step since the last log."""
    if dist.get_rank() != 0:
        return
    rec = {"step": step}
    by_type = {}
    for n, p in model.blocks.named_parameters():
        if p.ndim != 2:
            continue
        typ = n.split(".", 1)[1].rsplit(".", 1)[0].replace(".", "_")   # attn_q, attn_proj, mlp_fc, ...
        r = p.float().norm(dim=1)
        by_type.setdefault(typ, []).append(r)
        prev = _rowstat_prev.get(n)
        if prev is not None:
            rec.setdefault(f"{typ}/dlog_per_step", []).append(
                ((r.log() - prev[1].log()).abs().mean() / max(step - prev[0], 1)).item())
        _rowstat_prev[n] = (step, r)
    for typ, rs in by_type.items():
        r = torch.cat(rs)
        rec[f"{typ}/mean"], rec[f"{typ}/cv"] = r.mean().item(), (r.std() / r.mean()).item()
        if f"{typ}/dlog_per_step" in rec:
            rec[f"{typ}/dlog_per_step"] = sum(rec[f"{typ}/dlog_per_step"]) / len(rec[f"{typ}/dlog_per_step"])
    with open(logfile.replace(".txt", ".rowstat.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")

if RESUME_FROM:
    load_checkpoint(RESUME_FROM)
if READOUT_LAB:
    lab_track(RESUME_STEP)
for step in range(RESUME_STEP, train_steps + 1):
    H.start_step()

    # --------------- VALIDATION SECTION -----------------
    should_validate = (
        step == train_steps
        or step == STOP_STEP
        or (step > 0 and step % val_regular_interval == 0)
        or step in extra_val_steps
    )
    if should_validate:
        # stop the clock
        dist.barrier()
        training_time += time.perf_counter() - t0
        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        # (A) Tail-EMA readout: eval the partial blend (1-LAMBDA)*w + LAMBDA*ema (stash -> blend ->
        # eval -> restore). val_ema_loss defaults to the raw val_loss until the EMA exists.
        val_ema_loss = val_loss
        if _tailema is not None:
            with torch.no_grad():
                stash = [p.detach().clone() for p in model.parameters()]
                for j, p in enumerate(model.parameters()):
                    if _tailema[j] is not None:
                        p.copy_(((1.0 - TAILEMA_LAMBDA) * p.float() + TAILEMA_LAMBDA * _tailema[j]).to(p.dtype))
                val_ema_loss = 0
                for i in range(len(val_inputs) // mbs):
                    val_ema_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
                dist.all_reduce(val_ema_loss, op=dist.ReduceOp.SUM)
                val_ema_loss /= val_tokens
                for p, s in zip(model.parameters(), stash):
                    p.copy_(s)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} val_ema_loss:{val_ema_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*training_time/max(step, 1):.2f}ms", console=True)
        H.log_val(step, val_ema_loss, val_raw_loss=val_loss)
        log_rowstats(step)
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if READOUT_LAB and step == STOP_STEP:
        run_readout_lab(step)
    if step in CKPT_STEPS and step != RESUME_STEP:
        save_checkpoint(step)
    if step >= STOP_STEP:
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    # accumulate across microbatches in case we are running with fewer than 8 gpus
    assert len(inputs) % mbs == 0
    optimizers[0].nesterov_step()
    newton_capture = newton is not None and newton.is_refresh_step(step)
    if newton_capture:
        newton.enable_capture()
    train_loss = 0
    for i in range(len(inputs) // mbs):
        loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
        # NaN guard: catch divergence the step it happens, not 750 steps later.
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"non-finite train loss at step {step} mb {i}: {loss.item()}")
        loss.backward()
        train_loss = train_loss + loss.detach()
    if newton_capture:
        newton.disable_capture()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    # set optimization hyperparameters and take a step
    set_hparams(step)
    H.before_step()
    for opt in optimizers:
        opt.step()
    H.after_step(step, train_loss / inputs.numel())
    model.zero_grad(set_to_none=True)
    # (A) Tail-EMA buffer maintenance over the cooldown tail: ema += (w - ema)/TAU (exclude the
    # token embedding). Lazy-init at TAILEMA_START from the live post-step weights.
    if TAILEMA_TAU > 0 and TAILEMA_START <= (step + 1) < TAILEMA_END:
        if _tailema is None:
            _tailema = [None if p is model.embed.weight else p.detach().float().clone()
                        for p in model.parameters()]
        else:
            for j, p in enumerate(model.parameters()):
                if _tailema[j] is not None:
                    _tailema[j].add_(p.detach().float() - _tailema[j], alpha=1.0 / TAILEMA_TAU)
    if READOUT_LAB:
        lab_track(step + 1)
    if TRAIN_PROGRESS_INTERVAL > 0 and (step + 1) % TRAIN_PROGRESS_INTERVAL == 0:
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

H.finish()
dist.destroy_process_group()
