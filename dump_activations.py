# python dump_activations.py --ckpt_path="/Users/demihalf/sources/tinyllamas/stories260K/stories260K.pt" --device=cpu --compile=False --vocab_source="custom" --vocab_size=512 --batch_size=256 --device="mps"
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
import tqdm
import numpy as np

import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import Task
from export import model_export

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
ckpt_path = "out/ckpt.pt"
# data
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 256
vocab_source = (
    "llama2"  # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
)
vocab_size = 32000  # the Llama 2 tokenizer has 32K tokens
# model
layer_idx = 4
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0
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
# -----------------------------------------------------------------------------

# validating checks
assert vocab_source in ["llama2", "custom"]
assert (
    vocab_source == "custom" or vocab_size == 32000
), "The vocab from Meta has 32K tokens"
assert ckpt_path is not None, "Please specify a checkpoint path to dump activations"

# various inits, derived attributes, I/O setup
seed_offset = 0

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
)

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)  # start with model_args from command line

# resume training from a checkpoint.
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint["model_args"]
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
for k in [
    "dim",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "vocab_size",
    "multiple_of",
    "max_seq_len",
]:
    model_args[k] = checkpoint_model_args[k]
# create the model
gptconf = ModelArgs(**model_args)
print(gptconf)
model = Transformer(gptconf)
state_dict = checkpoint["model"]
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.to(device)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

train_batch_iter = iter_batches(split="train")

file_size = 1024 * 1024 * 1024  # 1GB
dtype_size = 4 if dtype == "float32" else 2
batch_disk_size = max_seq_len * batch_size * model_args["dim"] * dtype_size
n_batches = math.ceil(file_size / batch_disk_size)
print(f"n_batches: {n_batches}")

out_batch = np.zeros(
    (n_batches * batch_size * max_seq_len, model_args["dim"]),
    dtype=np.float32 if dtype == "float32" else np.float16,
)

out_batch_i = 0
out_batch_idx = 0

model.eval()
with torch.no_grad():
    for X, _ in tqdm.tqdm(train_batch_iter):
        acts = model.collect_activations(X, layer_idx=layer_idx)
        # assert acts.shape[0] == batch_size * max_seq_len
        out_batch[out_batch_i : out_batch_i + acts.shape[0]] = (
            acts.detach().cpu().numpy()
        )
        out_batch_i += acts.shape[0]

        if out_batch_i >= out_batch.shape[0]:
            # write to disk to out_dir
            print(f"Writing batch {out_batch_idx} to disk")
            with open(os.path.join(out_dir, f"batch_{out_batch_idx}.npy"), "wb") as f:
                np.save(f, out_batch)
            out_batch_idx += 1
            out_batch_i = 0

# write the last batch
print(f"Writing batch {out_batch_idx} to disk")
with open(os.path.join(out_dir, f"batch_{out_batch_idx}.npy"), "wb") as f:
    np.save(f, out_batch[:out_batch_i])
out_batch_idx += 1
