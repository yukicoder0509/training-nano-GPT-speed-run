"""Insert harness hooks into an official record script. usage: python tools/patch_record.py official.py train_gpt.py [val_metric]"""
import re, sys

src, dst = sys.argv[1], sys.argv[2]
metric = sys.argv[3] if len(sys.argv) > 3 else "val_loss"
s = open(src).read()
errors = []

def sub(pat, repl, desc, count=1, flags=re.M):
    global s
    n = len(re.findall(pat, s, flags))
    if n != count:
        errors.append(f"{desc}: {n} matches")
        return
    s = re.sub(pat, repl, s, flags=flags)

sub(r"^import torch\.distributed as dist\n", r"\g<0>import harness\n", "import")
sub(r"^torch\.cuda\.set_device\(device\)\n", r"\g<0>harness.setup(device)\n", "setup")
sub(r"^for step in range\(.*\):\n",
    "H = harness.Harness(model, optimizers, logfile if dist.get_rank() == 0 else None, config=dict(train_steps=train_steps))\n"
    r"\g<0>    H.start_step()\n", "loop")
# validation condition / stop (scripts with their own STOP_STEP already handle both)
if "STOP_STEP" not in s:
    if "should_validate = (" in s:
        sub(r"^(    should_validate = \(\n        step == train_steps\n)", r"\1        or H.should_stop(step)\n", "val cond")
    else:
        sub(r"^(    if step == train_steps or [^\n]*?):\n", r"\1 or H.should_stop(step):\n", "val cond")
    sub(r"^    if step == train_steps:\n        break\n", "    if step == train_steps or H.should_stop(step):\n        break\n", "break")
extras = {"val_loss": "", "val_ema_loss": ", val_raw_loss=val_loss"}[metric]
sub(r"^(        model\.train\(\)\n        # start the clock again\n)", f"        H.log_val(step, {metric}{extras})\n" + r"\1", "log_val")
# training loss
sub(r"^(    for i in range\(len\(inputs\) // mbs\):\n)", r"    train_loss = 0\n\1", "train_loss init")
sub(r"^(        loss\.backward\(\)\n)", r"\1        train_loss = train_loss + loss.detach()\n", "train_loss acc")
sub(r"^(    for opt in optimizers:\n        opt\.step\(\)\n)",
    r"    H.before_step()\n\1    H.after_step(step, train_loss / inputs.numel())\n", "opt step")
sub(r"^dist\.destroy_process_group\(\)\n", r"H.finish()\n\g<0>", "finish")

if errors:
    print("FAIL", src, *errors, sep="\n  ")
    sys.exit(1)
open(dst, "w").write(s)
print("ok", src)
