"""
Dump all wandb run data in a Claude-friendly format.

Usage:
    python wandb_dump.py                          # latest run, full dump
    python wandb_dump.py <run_id>                 # specific run
    python wandb_dump.py --compare <id1> <id2>    # side-by-side comparison
    python wandb_dump.py --history <run_id>        # full history CSV to stdout
    python wandb_dump.py --fields <run_id>         # list all available fields
    python wandb_dump.py --tail N <run_id>         # last N steps with all metrics
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


ENTITY = os.environ.get("WANDB_ENTITY", "magnus100304-universitetet-i-agder")
PROJECT = os.environ.get("WANDB_PROJECT", "dreamer")


def run_path(run_id):
    return f"{ENTITY}/{PROJECT}/{run_id}"


def get_latest_id():
    api = get_api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", order="-created_at", per_page=1)
    if not runs:
        print("No runs found.")
        sys.exit(1)
    return runs[0].id


def _finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def list_fields(run_id):
    """List every available field across summary, history, config, and system events."""
    api = get_api()
    run = api.run(run_path(run_id))

    print(f"=== Run: {run.name} ({run_id}) ===\n")

    # Config
    print("CONFIG KEYS:")
    for k in sorted(run.config.keys()):
        v = run.config[k]
        print(f"  {k} = {v}")

    # Summary
    print("\nSUMMARY KEYS (latest values):")
    for k, v in sorted(run.summary.items()):
        if isinstance(v, (int, float)):
            print(f"  {k} = {v}")
        elif isinstance(v, str):
            print(f"  {k} = {v!r}")
        else:
            print(f"  {k} = ({type(v).__name__})")

    # History keys (sampled from first + last rows)
    print("\nHISTORY KEYS (logged per step):")
    rows = list(run.scan_history(min_step=0, max_step=8001))
    all_keys = set()
    for row in rows:
        all_keys.update(k for k in row.keys() if _finite(row.get(k)))
    for k in sorted(all_keys):
        print(f"  {k}")

    # System event keys
    print("\nSYSTEM EVENT KEYS:")
    try:
        sys_history = run.history(stream="events").to_dict("records")
        sys_keys = set()
        for row in sys_history[:10]:
            sys_keys.update(k for k in row.keys() if not k.startswith("_") and _finite(row.get(k)))
        for k in sorted(sys_keys):
            print(f"  {k}")
    except Exception as e:
        print(f"  (error: {e})")


def dump_run(run_id, tail=None):
    """Full dump of a single run: config, summary, and optional tail history."""
    api = get_api()
    run = api.run(run_path(run_id))

    print(f"{'='*60}")
    print(f"RUN: {run.name}  |  ID: {run_id}  |  State: {run.state}")
    print(f"{'='*60}")

    # Config diff-friendly
    print("\n--- CONFIG ---")
    for k, v in sorted(run.config.items()):
        print(f"  {k}: {v}")

    # Summary metrics grouped
    summary = dict(run.summary)
    step = summary.get("_step", 0)
    runtime = summary.get("_runtime", 0)
    print(f"\n--- SUMMARY (step {step}, {runtime/3600:.1f}h) ---")

    groups = {
        "episode": [], "wm": [], "ac": [], "other": []
    }
    for k, v in sorted(summary.items()):
        if k.startswith("_") or k.startswith("system") or not isinstance(v, (int, float)):
            continue
        if "episode" in k or k in ("wm/reward",):
            groups["episode"].append((k, v))
        elif k.startswith("wm/"):
            groups["wm"].append((k, v))
        elif k.startswith("ac/"):
            groups["ac"].append((k, v))
        else:
            groups["other"].append((k, v))

    for gname, metrics in groups.items():
        if not metrics:
            continue
        print(f"  [{gname.upper()}]")
        for k, v in metrics:
            if isinstance(v, float):
                print(f"    {k:<30s} {v:.6f}")
            else:
                print(f"    {k:<30s} {v}")

    # Tail history
    n = tail or 15
    print(f"\n--- HISTORY (last {n} steps) ---")
    history = list(run.scan_history())
    if not history:
        print("  (no history)")
        return

    recent = history[-n:]
    # Collect all numeric keys
    all_keys = set()
    for row in recent:
        all_keys.update(k for k in row.keys() if not k.startswith("_") and _finite(row.get(k)))

    sorted_keys = sorted(all_keys)
    # Print as aligned rows
    for row in recent:
        step = row.get("_step", "?")
        print(f"\n  step={step}")
        for k in sorted_keys:
            v = row.get(k)
            if v is None:
                continue
            if isinstance(v, float):
                print(f"    {k:<30s} {v:.6f}")
            else:
                print(f"    {k:<30s} {v}")


def compare_runs(id1, id2, at_step=None):
    """Side-by-side comparison of two runs: config diff + metrics at same step."""
    api = get_api()
    run1 = api.run(run_path(id1))
    run2 = api.run(run_path(id2))

    print(f"{'='*70}")
    print(f"COMPARE: {run1.name} ({id1})  vs  {run2.name} ({id2})")
    print(f"{'='*70}")

    # Config diff
    print("\n--- CONFIG DIFF ---")
    all_config_keys = sorted(set(run1.config.keys()) | set(run2.config.keys()))
    has_diff = False
    for k in all_config_keys:
        v1 = run1.config.get(k, "(missing)")
        v2 = run2.config.get(k, "(missing)")
        if v1 != v2:
            print(f"  {k:<30s}  {str(v1):>15s}  ->  {str(v2):<15s}")
            has_diff = True
    if not has_diff:
        print("  (identical)")

    # Summary comparison
    print(f"\n--- SUMMARY COMPARISON ---")
    s1 = dict(run1.summary)
    s2 = dict(run2.summary)
    print(f"  {'':30s}  {'run1':>15s}  {'run2':>15s}  {'ratio':>10s}")
    print(f"  {'-'*30}  {'-'*15}  {'-'*15}  {'-'*10}")

    all_metric_keys = sorted(
        set(k for k in s1 if _finite(s1.get(k)) and not k.startswith("_") and not k.startswith("system"))
        | set(k for k in s2 if _finite(s2.get(k)) and not k.startswith("_") and not k.startswith("system"))
    )

    for k in all_metric_keys:
        v1 = s1.get(k)
        v2 = s2.get(k)
        v1_s = f"{v1:.6f}" if isinstance(v1, float) else str(v1) if v1 is not None else "-"
        v2_s = f"{v2:.6f}" if isinstance(v2, float) else str(v2) if v2 is not None else "-"
        ratio = ""
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1 != 0:
            r = v2 / v1
            ratio = f"{r:.2f}x"
        print(f"  {k:<30s}  {v1_s:>15s}  {v2_s:>15s}  {ratio:>10s}")

    # History at matched steps
    print(f"\n--- METRICS AT MATCHED STEPS ---")
    h1 = list(run1.scan_history())
    h2 = list(run2.scan_history())

    # Build step->row maps
    map1 = {row.get("_step"): row for row in h1}
    map2 = {row.get("_step"): row for row in h2}

    # Find common steps, sample evenly
    common = sorted(set(map1.keys()) & set(map2.keys()))
    if not common:
        print("  (no common steps)")
        return

    # Sample ~10 evenly spaced common steps
    if len(common) > 10:
        indices = [int(i * (len(common) - 1) / 9) for i in range(10)]
        sampled = [common[i] for i in indices]
    else:
        sampled = common

    # Key metrics to compare
    compare_keys = [
        "episode/reward", "episode/length",
        "wm/pred", "wm/sigreg", "wm/rollout", "wm/aux_recon", "wm/reward", "wm/grad_norm", "wm/total",
        "ac/critic_loss", "ac/critic_grad_norm", "ac/actor_loss", "ac/entropy",
        "ac/imagined_value", "ac/imagined_reward",
    ]

    for step in sampled:
        r1 = map1[step]
        r2 = map2[step]
        print(f"\n  step={step}")
        print(f"    {'metric':<30s}  {'run1':>15s}  {'run2':>15s}")
        print(f"    {'-'*30}  {'-'*15}  {'-'*15}")
        for k in compare_keys:
            v1 = r1.get(k)
            v2 = r2.get(k)
            if v1 is None and v2 is None:
                continue
            v1_s = f"{v1:.6f}" if isinstance(v1, float) else str(v1) if v1 is not None else "-"
            v2_s = f"{v2:.6f}" if isinstance(v2, float) else str(v2) if v2 is not None else "-"
            print(f"    {k:<30s}  {v1_s:>15s}  {v2_s:>15s}")


def dump_history_csv(run_id):
    """Dump full history as CSV to stdout."""
    api = get_api()
    run = api.run(run_path(run_id))
    history = list(run.scan_history())
    if not history:
        print("(no history)")
        return

    all_keys = set()
    for row in history:
        all_keys.update(k for k in row.keys() if _finite(row.get(k)))
    sorted_keys = sorted(all_keys)

    print(",".join(sorted_keys))
    for row in history:
        vals = []
        for k in sorted_keys:
            v = row.get(k)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.8f}")
            else:
                vals.append(str(v))
        print(",".join(vals))


def main():
    parser = argparse.ArgumentParser(description="Dump all wandb run data")
    parser.add_argument("run_id", nargs="?", help="Run ID (default: latest)")
    parser.add_argument("--compare", nargs=2, metavar=("ID1", "ID2"), help="Compare two runs")
    parser.add_argument("--history", metavar="ID", help="Dump full history as CSV")
    parser.add_argument("--fields", metavar="ID", help="List all available fields")
    parser.add_argument("--tail", type=int, default=None, help="Number of tail steps to show")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
    elif args.history:
        dump_history_csv(args.history)
    elif args.fields:
        list_fields(args.fields)
    elif args.run_id:
        dump_run(args.run_id, tail=args.tail)
    else:
        dump_run(get_latest_id(), tail=args.tail)


if __name__ == "__main__":
    main()
