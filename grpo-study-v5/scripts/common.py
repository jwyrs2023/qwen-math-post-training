"""Frozen V5 inputs, artifact names and a cumulative paid-session guard."""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
V4 = ROOT / "grpo-study-v4"
sys.path.append(str(V4 / "scripts"))
from runtime import PROMPT, SFT, load_policy, policy_lora, protocol as v4_protocol, sha, training_rows
from math_posttrain.checkpoints import seal, verify

REPORTS = PROJECT / "reports"
RUNS = PROJECT / "runs"
PROTOCOL = REPORTS / "protocol.json"
SESSION = REPORTS / "session.json"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def locked_protocol():
    p = read(PROTOCOL)
    if sha(V4 / "reports/evaluation-protocol.json") != p["v4_protocol_sha256"]:
        raise ValueError("V4 protocol hash changed")
    v4 = v4_protocol()
    for relative, expected in p["input_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    return p, v4


def evaluation_path(p, split):
    return ROOT / p["splits"][split]["path"]


def checkpoint(seed, step):
    return RUNS / f"seed-{seed}" / f"checkpoint-{step}"


def completed_checkpoints(directory):
    return sorted(
        (int(path.parent.name.removeprefix("checkpoint-")), path.parent)
        for path in directory.glob("checkpoint-*/checkpoint-manifest.json")
    )


@contextmanager
def gpu_lock():
    REPORTS.mkdir(exist_ok=True)
    with (REPORTS / "gpu.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another V5 GPU process owns the lock") from exc
        yield


class Budget:
    def __init__(self):
        s = read(SESSION)
        self.started = time.time()
        self.deadline = s["started_at_unix"] + 3600 * (
            s["cap_yuan"] - s["reserve_yuan"] - s["prior_spend_yuan"]
        ) / s["hourly_yuan"]
        self.check()

    def check(self, reserve_seconds=120):
        if time.time() + reserve_seconds >= self.deadline:
            raise RuntimeError("GPU budget exhausted except for saving and shutdown reserve")


def init_session(args):
    values = (args.hourly_yuan, args.cap_yuan, args.started_at_unix, args.prior_spend_yuan)
    if (not all(map(math.isfinite, values)) or args.hourly_yuan <= 0
            or not 0 < args.started_at_unix <= time.time()
            or not 0 <= args.prior_spend_yuan < args.cap_yuan - 5):
        raise ValueError("Invalid paid session values")
    available = args.cap_yuan - args.prior_spend_yuan - 5
    spent = (time.time() - args.started_at_unix) * args.hourly_yuan / 3600
    if spent >= available:
        raise ValueError("Session would spend the shutdown reserve")
    write(SESSION, dict(hourly_yuan=args.hourly_yuan, cap_yuan=args.cap_yuan,
                        reserve_yuan=5, started_at_unix=args.started_at_unix,
                        prior_spend_yuan=args.prior_spend_yuan))



def raise_cap(cap_yuan):
    s = read(SESSION)
    if not math.isfinite(cap_yuan) or cap_yuan <= s["cap_yuan"]:
        raise ValueError("The new cumulative cap must exceed the current cap")
    spent = s["prior_spend_yuan"] + (time.time() - s["started_at_unix"]) * s["hourly_yuan"] / 3600
    if spent >= cap_yuan - s["reserve_yuan"]:
        raise ValueError("The new cap would spend the shutdown reserve")
    previous = REPORTS / f"session-before-raise-{time.time_ns()}.json"
    write(previous, s)
    atomic_json(SESSION, s | {"cap_yuan": cap_yuan})

def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Existing artifact has a different identity: {path}")
    else:
        write(path, value)


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        name = stream.name
    os.replace(name, path)
