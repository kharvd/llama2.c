import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from autoenc import Autoencoder, Task

out_dir = "out"
dataset_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "llama_autoenc"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
input_dim = 64
dim_multiplier = 8
hidden_dim = input_dim * dim_multiplier
l1 = 0.01
# adam optimizer
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_iters = 100000  # total number of training iterations

# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = "bfloat16"  # float32|bfloat16|float16
compile = True  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

samples_per_iter = gradient_accumulation_steps * batch_size

print(f"samples per iteration will be: {samples_per_iter:,}")
print(
    f"breaks down as: {gradient_accumulation_steps} grad accum steps * {batch_size} batch size"
)
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    dtype=ptdtype,
    batch_size=batch_size,
    dataset_dir=dataset_dir,
    device=device,
)

iter_num = 0
best_val_loss = 1e9

model_args = dict(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    l1=l1,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = Autoencoder(**model_args)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["input_dim", "hidden_dim", "l1"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    model = Autoencoder(**model_args)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    assert False, f"Unknown init_from: {init_from}"

model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free memory

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0


@torch.no_grad()
def estimate_loss():
    out = {
        "loss": {},
        "l0_norm": {},
    }
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        l0_norms = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X = next(batch_iter)
            with ctx:
                metrics = model.metrics(X)
                loss = metrics["loss"]
                l0_norm = metrics["l0_norm"]

            losses[k] = loss.item()
            l0_norms[k] = l0_norm.item()

        out["l0_norm"][split] = l0_norms.mean()
        out["loss"][split] = losses.mean()
    model.train()
    return out


if wandb_log:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

train_batch_iter = iter_batches(split="train")
X = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
while True:
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        metrics = estimate_loss()
        losses = metrics["loss"]
        l0_norms = metrics["l0_norm"]
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val l0 norm {l0_norms['val']:.4f}"
        )
        if wandb_log:
            try:
                wandb.log(
                    {
                        "iter": iter_num,
                        "samples": iter_num * samples_per_iter,
                        "loss/train": losses["train"],
                        "loss/val": losses["val"],
                        "l0_norm/val": l0_norms["val"],
                        "lr": learning_rate,
                    },
                    step=iter_num,
                )
            except Exception as e:
                print(f"logging to wandb failed: {e}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            _ = model(X)
            loss = model.last_loss
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {learning_rate:e} | {dt*1000:.2f}ms "
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
