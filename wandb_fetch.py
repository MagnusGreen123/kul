"""
Fetch wandb run data for inspection.

Usage:
    python wandb_fetch.py                    # list all runs
    python wandb_fetch.py <run_id>           # full summary of a specific run
    python wandb_fetch.py <run_id> --tail N  # last N logged steps
    python wandb_fetch.py --latest           # latest run full summary
"""

import argparse
import math
import os
import sys

from dotenv import load_dotenv
load_dotenv()

import wandb


def get_api():
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        print("ERROR: WANDB_API_KEY not set in .env")
        sys.exit(1)
    return wandb.Api(api_key=key)


def get_entity_project():
    entity = os.environ.get("WANDB_ENTITY", "magnus100304-universitetet-i-agder")
    project = os.environ.get("WANDB_PROJECT", "dreamer")
    return entity, project


def list_runs():
    api = get_api()
    entity, project = get_entity_project()
    runs = api.runs(f"{entity}/{project}", order="-created_at")

    if not runs:
        print("No runs found.")
        return

    print(f"{'ID':<12} {'State':<10} {'Name':<30} {'Steps':>8} {'Runtime':>10}")
    print("-" * 75)
    for run in runs:
        runtime = run.summary.get("_runtime", 0)
        h, m = int(runtime // 3600), int((runtime % 3600) // 60)
        step = run.summary.get("_step", 0)
        print(f"{run.id:<12} {run.state:<10} {run.name or '(unnamed)':<30} {step:>8} {h:>3}h {m:>2}m")


def show_run(run_id, tail=None):
    api = get_api()
    entity, project = get_entity_project()
    run = api.run(f"{entity}/{project}/{run_id}")

    print(f"Run: {run.name or run.id}")
    print(f"State: {run.state}")
    print(f"Config:")
    for k, v in sorted(run.config.items()):
        print(f"  {k}: {v}")

    # Summary
    print(f"\n--- Summary ---")
    summary = dict(run.summary)

    runtime = summary.get("_runtime", 0)
    step = summary.get("_step", 0)
    print(f"  Global step: {step}")
    print(f"  Runtime: {runtime/3600:.1f}h")

    # Group metrics
    groups = {"episode": {}, "wm": {}, "ac": {}, "other": {}}
    for k, v in sorted(summary.items()):
        if k.startswith("_") or k.startswith("system"):
            continue
        if isinstance(v, (int, float)):
            if "episode" in k or "reward" in k or "length" in k:
                groups["episode"][k] = v
            elif "wm/" in k or "recon" in k or "kl" in k:
                groups["wm"][k] = v
            elif "ac/" in k or "actor" in k or "critic" in k or "imagined" in k:
                groups["ac"][k] = v
            else:
                groups["other"][k] = v

    for group_name, metrics in groups.items():
        if metrics:
            print(f"\n--- {group_name.upper()} ---")
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")

    # System metrics (GPU, CPU, memory, disk, network)
    print(f"\n--- SYSTEM METRICS ---")
    try:
        sys_history = run.history(stream="events").to_dict("records")
        if sys_history:
            # Collect all system keys and compute averages
            def _finite(v):
                return isinstance(v, (int, float)) and math.isfinite(v)

            sys_keys = set()
            for row in sys_history:
                sys_keys.update(k for k in row.keys()
                                if not k.startswith("_") and _finite(row.get(k)))

            # Compute min/avg/max for each metric
            sys_stats = {}
            for k in sorted(sys_keys):
                vals = [row[k] for row in sys_history if k in row and _finite(row[k])]
                if vals:
                    sys_stats[k] = {"min": min(vals), "avg": sum(vals)/len(vals), "max": max(vals)}

            # Group and display (exclusive: first match wins)
            gpu_keys, cpu_keys, mem_keys, disk_keys, net_keys, other_sys = [], [], [], [], [], []
            for k in sys_stats:
                kl = k.lower()
                if "gpu" in kl:
                    gpu_keys.append(k)
                elif "cpu" in kl:
                    cpu_keys.append(k)
                elif "disk" in kl:
                    disk_keys.append(k)
                elif "network" in kl:
                    net_keys.append(k)
                elif "memory" in kl or "mem" in kl:
                    mem_keys.append(k)
                else:
                    other_sys.append(k)

            def print_group(label, keys):
                if not keys:
                    return
                print(f"  [{label}]")
                for k in sorted(keys):
                    s = sys_stats[k]
                    print(f"    {k:<40s}  min={s['min']:>8.1f}  avg={s['avg']:>8.1f}  max={s['max']:>8.1f}")

            print_group("GPU", gpu_keys)
            print_group("CPU", cpu_keys)
            print_group("Memory", mem_keys)
            print_group("Disk", disk_keys)
            print_group("Network", net_keys)
            print_group("Other", other_sys)

            print(f"  ({len(sys_history)} system samples)")
        else:
            print("  (no system metrics logged)")
    except Exception as e:
        print(f"  (could not fetch system metrics: {e})")

    # History (tail)
    if tail:
        print(f"\n--- Last {tail} logged steps ---")
        history = list(run.scan_history())
        if not history:
            print("  (no history)")
            return

        recent = history[-tail:]
        # Find all keys present
        all_keys = set()
        for row in recent:
            all_keys.update(k for k in row.keys() if not k.startswith("_") and isinstance(row[k], (int, float)))

        # Print as table
        sorted_keys = sorted(all_keys)
        # Truncate key names for display
        short_keys = [k[-25:] for k in sorted_keys]

        header = f"{'step':>8} | " + " | ".join(f"{k:>12}" for k in short_keys)
        print(header)
        print("-" * len(header))
        for row in recent:
            step = row.get("_step", "?")
            vals = []
            for k in sorted_keys:
                v = row.get(k)
                if v is None:
                    vals.append(f"{'':>12}")
                elif isinstance(v, float):
                    vals.append(f"{v:>12.4f}")
                else:
                    vals.append(f"{v:>12}")
            print(f"{step:>8} | " + " | ".join(vals))


def show_latest(tail=None):
    api = get_api()
    entity, project = get_entity_project()
    runs = api.runs(f"{entity}/{project}", order="-created_at", per_page=1)
    if not runs:
        print("No runs found.")
        return
    show_run(runs[0].id, tail=tail)


def main():
    parser = argparse.ArgumentParser(description="Fetch wandb run data")
    parser.add_argument("run_id", nargs="?", help="Run ID to inspect")
    parser.add_argument("--latest", action="store_true", help="Show latest run")
    parser.add_argument("--tail", type=int, default=None, help="Show last N history rows")
    args = parser.parse_args()

    if args.latest:
        show_latest(tail=args.tail)
    elif args.run_id:
        show_run(args.run_id, tail=args.tail)
    else:
        list_runs()


if __name__ == "__main__":
    main()
