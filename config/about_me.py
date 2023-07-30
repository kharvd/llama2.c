wandb_log = True
wandb_project = "llamac_about_me"
wandb_run_name = "run_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

eval_interval = 100

# data
batch_size = 112  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 256

# model
dim = 512
n_layers = 8
n_heads = 8
multiple_of = 32

compile = False

# adamw optimizer
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
learning_rate = 3e-4  # max learning rate
max_iters = 100000  # total number of training iterations