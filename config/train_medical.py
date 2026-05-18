# Model architecture
n_layer = 12        # 12 transformer layers stacked on top of each other
n_head = 12         # 12 attention heads per layer
n_embd = 768        # each token is represented as a 768-dimensional vector
block_size = 512    # the model sees 512 tokens at a time (context window)
dropout = 0.1       # randomly drops 10% of connections to prevent memorisation

# Data
dataset = 'pubmed'
batch_size = 32     # processes 32 chunks of 512 tokens simultaneously

# Training
max_iters = 10000   # number of training steps
learning_rate = 3e-4
lr_decay_iters = 10000
eval_interval = 500  # check val loss every 500 steps
eval_iters = 50

# System
device = 'mps'      # use your Mac M5 GPU
compile = False     # metal doesn't support torch.compile yet
dtype = 'float32'   # mps works best with float32

# Logging
wandb_log = False
out_dir = 'out-medical'

# Checkpointing - save progress automatically
always_save_checkpoint = True   # save after every eval
out_dir = 'out-medical'         # folder where checkpoints are saved
