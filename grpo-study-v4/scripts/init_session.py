"""Record a real GPU session's price and billable start before any CUDA jobs."""
import argparse
import json
import math
import time

from runtime import PROJECT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=["screen", "confirm"], required=True)
parser.add_argument("--hourly-yuan", type=float, required=True)
parser.add_argument("--started-at-unix", type=float, required=True)
parser.add_argument("--prior-spend-yuan", type=float, default=0)
args = parser.parse_args()
cap = 40 if args.phase == "screen" else 90
if (not all(math.isfinite(x) for x in (args.hourly_yuan, args.started_at_unix, args.prior_spend_yuan))
        or not 0 < args.hourly_yuan or not 0 < args.started_at_unix <= time.time()
        or not 0 <= args.prior_spend_yuan < cap - 5):
    parser.error("Use the verified rate, billable start and accumulated expense")
report = PROJECT / "reports"
if args.phase == "confirm":
    decision = json.loads((report / "selection.json").read_text())
    if decision["method"] is None:
        parser.error("No candidate qualified")
session = dict(phase=args.phase, budget_yuan=cap, reserve_yuan=5, hourly_yuan=args.hourly_yuan,
               started_at_unix=args.started_at_unix, prior_spend_yuan=args.prior_spend_yuan)
with (report / "session.json").open("x") as stream:
    json.dump(session, stream, indent=2)
print(f"Recorded {args.phase} budget: {cap} yuan total, 5 yuan reserved.")
