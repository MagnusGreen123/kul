"""Compute true 50-ep rolling window from wandb output.log batch data.

Each "Step X | avg_reward=Y | n_eps=Z" line gives Z episodes with mean reward Y.
Reconstruct per-episode stream by repeating Y, Z times — an approximation but
much closer to truth than the wandb history (which only logs ~30% of episodes
due to step-keyed deduplication).
"""
import os, re, sys, tempfile
from collections import deque
from dotenv import load_dotenv
load_dotenv()
import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "magnus100304-universitetet-i-agder")
PROJECT = os.environ.get("WANDB_PROJECT", "dreamer")

PATTERN = re.compile(
    r"Step\s+(\d+)\s+\|.*?avg_reward=(-?[\d.]+).*?episodes=([\d.]+).*?n_eps=([\d.]+)"
)


def parse_log(content):
    """Parse stdout log → list of (step, avg_reward, total_episodes, n_eps)."""
    out = []
    for ln in content.splitlines():
        m = PATTERN.search(ln)
        if not m:
            continue
        step = int(m.group(1))
        avg = float(m.group(2))
        total = int(float(m.group(3)))
        n_eps = int(float(m.group(4)))
        out.append((step, avg, total, n_eps))
    return out


def expand(batches):
    """Expand batches → approximate per-episode (step, reward) stream."""
    eps = []
    for step, avg, total, n in batches:
        for _ in range(n):
            eps.append((step, avg))
    return eps


def rolling_50(eps, window=50):
    buf = deque(maxlen=window)
    peak = float("-inf")
    peak_at = None
    history = []
    for step, r in eps:
        buf.append(r)
        if len(buf) >= window:
            avg = sum(buf) / window
            history.append((step, avg))
            if avg > peak:
                peak = avg
                peak_at = step
    return history, peak, peak_at


def fetch_log(run_id):
    api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    files = list(run.files())
    for f in files:
        if f.name == "output.log":
            tmp = tempfile.mkdtemp()
            local = f.download(root=tmp, replace=True, exist_ok=True)
            with open(local.name, "r", errors="replace") as fp:
                return run.name, fp.read()
    return run.name, ""


def report(label, run_id):
    name, content = fetch_log(run_id)
    batches = parse_log(content)
    if not batches:
        print(f"{label}: no parseable batches")
        return
    eps = expand(batches)
    history, peak, peak_at = rolling_50(eps, 50)
    first_step = batches[0][0]
    last_step, last_avg, last_total, last_n = batches[-1]
    print(f"\n=== {label}: {name} ({run_id}) ===")
    print(f"  log spans steps {first_step} - {last_step}")
    print(f"  total episodes: {last_total}  ({len(eps)} reconstructed from batches)")
    print(f"  raw episode reward range across batches: "
          f"[{min(b[1] for b in batches):.1f}, {max(b[1] for b in batches):.1f}]")
    print(f"  TRUE 50-ep window peak: {peak:+.2f} at step={peak_at}")
    if history:
        print(f"  TRUE 50-ep window final: {history[-1][1]:+.2f} at step={history[-1][0]}")
    if history:
        n_show = 8
        idx = [int(i * (len(history) - 1) / (n_show - 1)) for i in range(n_show)]
        print(f"  trajectory:")
        for i in idx:
            s, a = history[i]
            print(f"    step={s:>7}  win50_avg={a:+6.2f}")


if __name__ == "__main__":
    for label, run_id in [("v6 (RESUMED)", "74uc7yum"),
                          ("v11", "ru1kx6zd"),
                          ("v14", "jawetgeq")]:
        report(label, run_id)
