# SpaceCode for small ideas

## Methods for Allocating GPUs

### 1. Interactive Allocation with `salloc`

Request GPU resources interactively using `salloc` with the `--gres` (Generic Resources) flag:

```bash
salloc --time=1:00:00 --partition=gpu_a100 --gres=gpu:2
```

**Parameters:**
- `--time=1:00:00`: Allocate resources for 1 hour
- `--partition=gpu_a100`: Use the GPU A100 partition
- `--gres=gpu:2`: Request 2 GPUs


### 2. Background Job Submission with `nohup`

Submit jobs to run in the background without SLURM:

```bash
nohup python xxx.py > output.log 2>&1 &
```

