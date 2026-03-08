# SpaceCode for small ideas

## Project Structure
- `main.py` - Main entry point to choose training framework or run RoPE test
- `models/model.py` - Shared SimpleCNN model definition and data loading utilities
- `multi_cards/ddp.py` - PyTorch DDP training implementation
- `multi_cards/deepseed.py` - DeepSpeed training implementation
- `multi_cards/hf_accelerate.py` - Hugging Face Accelerate training implementation
- `pos_encoding/` - RoPE (Rotary Position Embedding) implementation
- `requirements.txt` - Python dependencies



## Allocating GPUs

Request GPU resources interactively using `salloc` with the `--gres` (Generic Resources) flag:

```bash
salloc --time=1:00:00 --partition=gpu_a100 --gres=gpu:2
conda activate space
```

**Parameters:**
- `--time=1:00:00`: Allocate resources for 1 hour
- `--partition=gpu_a100`: Use the GPU A100 partition
- `--gres=gpu:2`: Request 2 GPUs



## Multi-Cards Processing

**PyTorch DDP:**
```bash
python multi_cards/ddp.py --batch_size 32 --num_epochs 10
torchrun --nprocs_per_node=2 multi_cards/ddp.py --batch_size 32 --num_epochs 10
```

**DeepSpeed:**
```bash
deepspeed multi_cards/deepseed.py --batch_size 32 --num_epochs 10
deepspeed --num_gpus=2 multi_cards/deepseed.py --batch_size 32 --num_epochs 10
```

**Hugging Face Accelerate:**
```bash
python multi_cards/hf_accelerate.py --batch_size 32 --num_epochs 10
accelerate launch multi_cards/hf_accelerate.py --batch_size 32 --num_epochs 10
accelerate launch --multi_gpu --num_processes=2 multi_cards/hf_accelerate.py --batch_size 32 --num_epochs 10
```

### How to run

**RoPE positional embedding test:**
```bash
conda activate space
python main.py --framework rope
# Or run directly: python pos_encoding/test.py
```

**Interactive mode:**
```bash
conda activate space
python main.py --batch_size 32 --num_epochs 10
# Then select: 1 (DDP), 2 (DeepSpeed), 3 (Accelerate), or 4 (RoPE test)
```

**Direct framework selection:**
```bash
# DDP
python main.py --framework ddp --batch_size 32 --num_epochs 10

# DeepSpeed with 2 GPUs
python main.py --framework deepseed --num_gpus 2 --batch_size 32 --num_epochs 10

# Accelerate
python main.py --framework accelerate --batch_size 32 --num_epochs 10

# RoPE test
python main.py --framework rope
```


**Background Job Submission with `nohup`**

Submit jobs to run in the background without SLURM:

```bash
nohup python xxx.py > output.log 2>&1 &
```