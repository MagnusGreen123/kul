"""Fetch wandb output.log for a run and print it (or a head of it)."""
import os, sys
from dotenv import load_dotenv
load_dotenv()
import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "magnus100304-universitetet-i-agder")
PROJECT = os.environ.get("WANDB_PROJECT", "dreamer")

def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    n_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    if not run_id:
        sys.exit("usage: fetch_wandb_log.py <run_id> [n_lines]")

    api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    files = list(run.files())
    print(f"=== Files in run {run.name} ({run_id}) ===")
    for f in files:
        print(f"  {f.name} ({f.size} bytes)")
    print()

    # Try output.log
    target_names = ["output.log", "wandb-summary.json"]
    for f in files:
        if f.name in target_names:
            print(f"=== {f.name} ===")
            local = f.download(root="/tmp/wandb_dl", replace=True, exist_ok=True)
            with open(local.name, "r", errors="replace") as fp:
                content = fp.read()
            lines = content.splitlines()
            print(f"  total lines: {len(lines)}, total chars: {len(content)}")
            print(f"  --- first {n_lines} lines ---")
            for ln in lines[:n_lines]:
                print(ln)
            print(f"  --- last {n_lines} lines ---")
            for ln in lines[-n_lines:]:
                print(ln)


if __name__ == "__main__":
    main()
