import argparse
import os
import shutil
import subprocess
import time


def run_command(command):
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        return completed.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as exc:
        return f"[unavailable] {' '.join(command)}: {exc}"


def get_memory_pressure():
    output = run_command(["memory_pressure"])
    return "\n".join(output.splitlines()[:12])


def get_top_processes(limit):
    output = run_command(["ps", "-axo", "pid,ppid,%mem,rss,command"])
    lines = output.splitlines()
    if len(lines) <= 1:
        return output

    header = lines[0]
    rows = sorted(lines[1:], key=lambda line: float(line.split(None, 4)[2]), reverse=True)
    return "\n".join([header] + rows[:limit])


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def get_torch_mps_memory():
    try:
        import torch
    except ImportError:
        return "PyTorch not installed"

    if not torch.backends.mps.is_available():
        return "MPS not available"

    allocated = torch.mps.current_allocated_memory()
    driver = torch.mps.driver_allocated_memory()
    return (
        f"torch.mps.current_allocated_memory(): {format_bytes(allocated)}\n"
        f"torch.mps.driver_allocated_memory(): {format_bytes(driver)}"
    )


def get_powermetrics():
    if shutil.which("powermetrics") is None:
        return "powermetrics not available"

    output = run_command(["powermetrics", "--show-process-gpu", "-n", "1"])
    tail = output.splitlines()[-30:]
    return "\n".join(tail)


def clear_screen():
    os.system("clear")


def render_snapshot(limit, include_powermetrics):
    print("=== Memory Pressure ===")
    print(get_memory_pressure())
    print()
    print(f"=== Top {limit} Processes By %MEM ===")
    print(get_top_processes(limit))
    print()
    print("=== PyTorch MPS Memory ===")
    print(get_torch_mps_memory())
    if include_powermetrics:
        print()
        print("=== powermetrics GPU Activity ===")
        print(get_powermetrics())


def main():
    parser = argparse.ArgumentParser(description="Monitor RAM and PyTorch MPS memory on macOS.")
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds.")
    parser.add_argument("--top", type=int, default=8, help="Number of processes to show.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    parser.add_argument(
        "--powermetrics",
        action="store_true",
        help="Include `powermetrics --show-process-gpu -n 1` output. This may require sudo.",
    )
    args = parser.parse_args()

    while True:
        clear_screen()
        render_snapshot(args.top, args.powermetrics)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
