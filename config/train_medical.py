# Model architecture
n_layer = 12
n_head = 12
n_embd = 768
block_size = 256
dropout = 0.1

# Data
dataset = 'pubmed'
batch_size = 8
gradient_accumulation_steps = 8

# Training
max_iters = 10000
learning_rate = 3e-4
lr_decay_iters = 10000
eval_interval = 500
eval_iters = 20

# System
device = 'mps'
compile = False
dtype = 'float32'

# Logging
wandb_log = False
out_dir = 'out-medical'
always_save_checkpoint = True
