"""
Main entry point for training with different distributed frameworks
"""
import argparse
import subprocess
import sys
from pathlib import Path

def run_ddp(args):
    """Run training with PyTorch DDP"""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "multi_cards" / "ddp.py"),
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.num_epochs),
    ]
    print(f"🚀 Starting DDP training with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_deepseed(args):
    """Run training with DeepSpeed"""
    cmd = [
        "deepspeed",
        str(Path(__file__).parent / "multi_cards" / "deepseed.py"),
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.num_epochs),
    ]
    
    if args.num_gpus:
        cmd.insert(2, "--num_gpus=" + str(args.num_gpus))
    
    print(f"🚀 Starting DeepSpeed training with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_accelerate(args):
    """Run training with Hugging Face Accelerate"""
    cmd = [
        "accelerate", "launch",
        str(Path(__file__).parent / "multi_cards" / "hf_accelerate.py"),
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.num_epochs),
    ]
    
    if args.num_gpus:
        cmd.insert(2, f"--num_processes={args.num_gpus}")
    
    print(f"🚀 Starting Accelerate training with command: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_rope_test(args):
    """Run RoPE positional embedding test"""
    cmd = [sys.executable, str(Path(__file__).parent / "pos_encoding" / "test.py")]
    print(f"🧪 Running RoPE positional embedding test...")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="SpaceCode Training Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run DDP training
  python main.py --framework ddp --batch_size 32 --num_epochs 10
  
  # Run DeepSpeed training with 2 GPUs
  python main.py --framework deepseed --num_gpus 2 --batch_size 32 --num_epochs 10
  
  # Run Accelerate training
  python main.py --framework accelerate --batch_size 32 --num_epochs 10
  
  # Run RoPE positional embedding test
  python main.py --framework rope
  
  # Interactive mode (no framework specified)
  python main.py --batch_size 32 --num_epochs 10
        """
    )
    
    parser.add_argument(
        "--framework",
        type=str,
        choices=["ddp", "deepseed", "accelerate", "rope"],
        help="Choose training framework (ddp, deepseed, accelerate) or rope for RoPE test"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs (default: 10)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (for DeepSpeed and Accelerate)"
    )
    
    args = parser.parse_args()
    
    # If no framework specified, show interactive menu
    if not args.framework:
        print("\n" + "="*60)
        print("🎯 SpaceCode - Distributed Training Framework Selector")
        print("="*60)
        print("\nAvailable options:")
        print("  1. DDP       - PyTorch Distributed Data Parallel")
        print("  2. DeepSpeed - DeepSpeed with ZeRO optimization")
        print("  3. Accelerate - Hugging Face Accelerate")
        print("  4. RoPE      - RoPE positional embedding test")
        print("="*60 + "\n")
        
        choice = input("Select option (1/2/3/4) or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            print("Exiting...")
            sys.exit(0)
        
        choice_map = {
            '1': 'ddp',
            '2': 'deepseed',
            '3': 'accelerate',
            '4': 'rope'
        }
        
        if choice not in choice_map:
            print("❌ Invalid choice. Please select 1, 2, 3, or 4.")
            sys.exit(1)
        
        args.framework = choice_map[choice]
        print(f"\n✅ Selected framework: {args.framework.upper()}\n")
    
    # Route to appropriate function
    if args.framework == "ddp":
        run_ddp(args)
    elif args.framework == "deepseed":
        run_deepseed(args)
    elif args.framework == "accelerate":
        run_accelerate(args)
    elif args.framework == "rope":
        run_rope_test(args)


if __name__ == "__main__":
    main()
