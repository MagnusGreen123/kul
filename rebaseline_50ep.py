"""Re-process v6, v11, v14 wandb history with 50-ep rolling window.

Compares peak/final 50-ep window_avg metrics across runs to check whether
"v11 hit +11" claim holds when measured the same way as v14.
"""
import os
import sys
from collections import deque

from dotenv import load_dotenv
load_dotenv()
import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "magnus100304-universitetet-i-agder")
PROJECT = os.environ.get("WANDB_PROJECT", "dreamer")

RUNS = {
    "v6":  "74uc7yum",
    "v11": "ru1kx6zd",
    "v14": "jawetgeq",
}

WINDOW = 50


def get_api():
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        sys.exit("ERROR: WANDB_API_KEY not set")
    return wandb.Api(api_key=key)


def episodes_from_history(run):
    """Extract ordered episode rewards from wandb history.

    Returns list of (step, episode_number, reward) tuples in episode order.
    """
    rows = list(run.scan_history(keys=["_step", "episode/number", "episode/reward"]))
    eps = []
    seen_ep_numbers = set()
    for r in rows:
        ep_num = r.get("episode/number")
        rew = r.get("episode/reward")
        step = r.get("_step")
        if ep_num is None or rew is None:
            continue
        if ep_num in seen_ep_numbers:
            continue
        seen_ep_numbers.add(ep_num)
        eps.append((step, ep_num, rew))
    eps.sort(key=lambda t: t[1])
    return eps


def rolling_stats(eps, window=WINDOW):
    """Compute rolling 50-ep window: at each new episode, what was the
    50-ep avg, and track the running peak.
    """
    window_avgs = []  # (step, ep_num, window_avg)
    buf = deque(maxlen=window)
    peak_avg = float("-inf")
    peak_at = None  # (step, ep_num)
    for step, ep_num, rew in eps:
        buf.append(rew)
        if len(buf) >= window:
            avg = sum(buf) / len(buf)
            window_avgs.append((step, ep_num, avg))
            if avg > peak_avg:
                peak_avg = avg
                peak_at = (step, ep_num)
    return window_avgs, peak_avg, peak_at


def report(label, eps, window_avgs, peak_avg, peak_at):
    n_eps = len(eps)
    raw_max = max(r for _, _, r in eps) if eps else None
    raw_min = min(r for _, _, r in eps) if eps else None
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"  total episodes: {n_eps}")
    print(f"  raw episode reward range: [{raw_min}, {raw_max}]")
    print(f"  raw episode peak: {raw_max}")
    if not window_avgs:
        print(f"  (fewer than {WINDOW} episodes — no 50-ep window)")
        return
    final_avg = window_avgs[-1][2]
    print(f"  50-ep window peak avg: {peak_avg:.2f} at step={peak_at[0]}, ep#{peak_at[1]}")
    print(f"  50-ep window final avg: {final_avg:.2f} at step={window_avgs[-1][0]}, ep#{window_avgs[-1][1]}")
    # Sample trajectory
    n_samples = 8
    if len(window_avgs) > n_samples:
        idx = [int(i * (len(window_avgs) - 1) / (n_samples - 1)) for i in range(n_samples)]
        sample = [window_avgs[i] for i in idx]
    else:
        sample = window_avgs
    print(f"  trajectory (step, ep#, 50-ep avg):")
    for step, ep_num, avg in sample:
        print(f"    step={step:>7} ep#{ep_num:>4}  window_avg={avg:+6.2f}")


def main():
    api = get_api()
    print("Fetching wandb history (this takes ~30s per run)...")

    results = {}
    for label, run_id in RUNS.items():
        print(f"  fetching {label} ({run_id}) ...", flush=True)
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        eps = episodes_from_history(run)
        wa, peak, peak_at = rolling_stats(eps, WINDOW)
        results[label] = (eps, wa, peak, peak_at)

    for label, (eps, wa, peak, peak_at) in results.items():
        report(f"{label} ({RUNS[label]})", eps, wa, peak, peak_at)

    # Side-by-side summary
    print(f"\n{'='*60}")
    print("SIDE-BY-SIDE 50-ep WINDOW COMPARISON")
    print(f"{'='*60}")
    print(f"  {'run':<6} {'raw_peak':>10} {'win_peak':>10} {'win_final':>10} {'eps':>6}")
    for label in RUNS:
        eps, wa, peak, peak_at = results[label]
        raw_peak = max(r for _, _, r in eps) if eps else None
        final = wa[-1][2] if wa else None
        n = len(eps)
        peak_s = f"{peak:+.2f}" if wa else "-"
        final_s = f"{final:+.2f}" if wa else "-"
        print(f"  {label:<6} {raw_peak:>+10.0f} {peak_s:>10} {final_s:>10} {n:>6}")


if __name__ == "__main__":
    main()
