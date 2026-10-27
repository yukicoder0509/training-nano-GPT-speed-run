"""
harness.py - shared run plumbing for the record scripts (train_gpt.py).

Kept out of the record scripts so each record's diff is only its optimization change.
  - VRAM cap at 80GB (competition targets H100-80GB, we run on H200-141GB)
  - W&B logging of train/val metrics + leaderboard summary (total_train_steps, val_loss)
  - optimization dynamics (optim_dynamics.py), rank 0 only, never to a cerulean-labs entity
  - early stop at the record step (STOP_STEP) with the LR schedule horizon unchanged

Usage inside a record script:
    harness.setup(device)                         # right after torch.cuda.set_device
    H = harness.Harness(model, optimizers, logfile, config=dict(...))
    ... each val step:   H.log_val(step, val_loss=..., **extra)
                         if H.should_stop(step): break   (and add `or H.should_stop(step)` to the val condition)
    ... each train step: H.before_step(); <optimizer steps>; H.after_step(step, train_loss)
    H.finish()
"""

import os
import subprocess

import torch
import torch.distributed as dist

from optim_dynamics import OptimDynamicsLogger, make_wandb_log_fn

# nanogpt parameter names -> optim_dynamics block types (the default rules target HF names)
NANOGPT_RULES = [
    ("embed", r"^embed\.weight$"),
    ("lm_head", r"^proj\.weight$"),
    ("attn_q", r"\.attn\.q\."),
    ("attn_k", r"\.attn\.k\."),
    ("attn_v", r"\.attn\.v\."),
    ("attn_o", r"\.attn\.proj\."),
    ("mlp_up", r"\.mlp\.fc\."),
    ("mlp_down", r"\.mlp\.proj\."),
]


def setup(device):
    # cap the caching allocator; the CUDA context and NCCL buffers live outside it, hence the headroom
    max_alloc_bytes = float(os.environ.get("MAX_ALLOC_GIB", 72)) * 1024**3
    torch.cuda.set_per_process_memory_fraction(max_alloc_bytes / torch.cuda.get_device_properties(device).total_memory, device)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


class Harness:
    def __init__(self, model, optimizers, logfile, config=None):
        self.model = model
        # wrappers (EMA-Nesterov) hold the lr-carrying optimizers in .inner_optimizer
        self.optimizers = []
        for opt in optimizers:
            inner = getattr(opt, "inner_optimizer", None)
            self.optimizers += inner if isinstance(inner, list) else [inner] if inner is not None else [opt]
        self.device = next(model.parameters()).device
        self.rank0 = dist.get_rank() == 0
        self.stop_step = int(os.environ["STOP_STEP"]) if os.environ.get("STOP_STEP") else None
        self.last_step = None
        self.last_val = None
        self.dyn = None
        self.run = None
        if not self.rank0:
            return
        import wandb
        entity = os.environ.get("WANDB_ENTITY")
        dynamics = os.environ.get("OPTIM_DYNAMICS", "1") == "1"
        if dynamics and entity and "cerulean-labs" in entity:
            raise RuntimeError("optimization dynamics must never be logged to a cerulean-labs entity; "
                               "set OPTIM_DYNAMICS=0 or use the personal project")
        self.run = wandb.init(entity=entity, project=os.environ.get("WANDB_PROJECT"),
                              name=os.environ.get("WANDB_RUN_NAME"),
                              config=dict(config or {}, stop_step=self.stop_step, git_commit=os.environ.get("RECORD_COMMIT") or _git_commit(),
                                          record=os.environ.get("RECORD"), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                                          world_size=dist.get_world_size(), torch_version=torch.version.__version__,
                                          gpu=torch.cuda.get_device_name(self.device), logfile=logfile))
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")
        self.logfile = logfile
        if dynamics:
            log_every = int(os.environ.get("DYN_EVERY", 100))
            self.dyn = OptimDynamicsLogger(model, log_every=log_every,
                                           spectral_every=int(os.environ.get("DYN_SPECTRAL_EVERY", 5 * log_every)),
                                           out_path=logfile.replace(".txt", ".dyn.jsonl"), rules=NANOGPT_RULES,
                                           log_fn=make_wandb_log_fn(step_key="step"))

    # -- training step hooks (call on every rank; work happens on rank 0) --------------
    def start_step(self):
        """Optional, at the top of the iteration: for optimizers that move the weights before the
        forward pass (EMA-Nesterov lookahead), so the measured update dW spans the whole iteration."""
        self._w0 = None
        if self.dyn is not None and self.dyn._phase() is not None:
            self._w0 = {name: p.detach().clone() for name, p, _, _ in self.dyn.params}

    def before_step(self):
        if self.dyn is not None:
            # power iteration draws random vectors: keep the training RNG untouched
            with torch.random.fork_rng(devices=[self.device]):
                self.dyn.before_step()
            if getattr(self, "_w0", None) is not None:
                for name, b in self.dyn._buf.items():
                    if "w_before" in b:
                        b["w_before"] = self._w0[name]
                self._w0 = None

    def after_step(self, step, train_loss=None):
        """train_loss: this rank's mean per-token loss (tensor); averaged over ranks here."""
        if self.dyn is not None:
            with torch.random.fork_rng(devices=[self.device]):
                self.dyn.after_step()
        if train_loss is not None:
            train_loss = train_loss.detach().float().clone()
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        if self.rank0:
            import wandb
            log = {"step": step + 1}
            if train_loss is not None:
                log["train/loss"] = train_loss.item()
            for i, opt in enumerate(self.optimizers):
                for j, g in enumerate(opt.param_groups):
                    if "lr" in g:
                        log[f"lr/opt{i}_group{j}"] = float(g["lr"])
            wandb.log(log)

    # -- validation ---------------------------------------------------------------------
    def log_val(self, step, val_loss, **extra):
        """val_loss is the record's reported metric (e.g. val_ema_loss for the Tail-EMA records)."""
        self.last_step, self.last_val = step, float(val_loss)
        peak = torch.tensor(torch.cuda.max_memory_reserved(self.device) / 1024**3, device=self.device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if self.rank0:
            import wandb
            wandb.log({"step": step, "val/loss": float(val_loss), "peak_mem_reserved_gib": peak.item(),
                       **{f"val/{k}": float(v) for k, v in extra.items()}})

    def should_stop(self, step):
        return self.stop_step is not None and step >= self.stop_step

    def finish(self):
        if self.rank0:
            import wandb
            self.run.summary["total_train_steps"] = self.last_step
            self.run.summary["val_loss"] = self.last_val
            wandb.save(self.logfile, policy="now")
            wandb.finish()
