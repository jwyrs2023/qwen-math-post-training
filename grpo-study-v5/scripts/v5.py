"""Local preflight and single-GPU phase entry point for the frozen V5 plan."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

from common import PROTOCOL, REPORTS, RUNS, evaluation_path, gpu_lock, init_session
from common import locked_protocol, raise_cap, read, sha, training_rows


def preflight(p, v4, gpu=False):
    from math_posttrain.parsing import extract_gold

    versions = {name: version(name) for name in p["versions"]}
    if any(v.split("+")[0] != p["versions"][name] for name, v in versions.items()):
        raise ValueError(f"Environment versions differ from frozen versions: {versions}")
    pool = training_rows()
    ids = {row["sample_id"] for row in pool}
    if len(ids) != len(pool) or len(ids) != p["training_rows"]:
        raise ValueError("Unexpected training pool identity")
    for split, spec in p["splits"].items():
        rows = read(evaluation_path(p, split))
        eval_ids = {row["sample_id"] for row in rows}
        if len(rows) != len(eval_ids) or len(rows) != spec["rows"] or ids & eval_ids:
            raise ValueError(f"Invalid or overlapping evaluation IDs for {split}")
        for row in rows:
            extract_gold(row["answer"])
    if gpu:
        import torch
        from huggingface_hub import snapshot_download
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("GPU preflight requires CUDA with BF16")
        cached = Path(snapshot_download(v4["model"], revision=v4["revision"], local_files_only=True))
        index = cached / "model.safetensors.index.json"
        weights = set(read(index)["weight_map"].values()) if index.exists() else {"model.safetensors"}
        if not all((cached / filename).is_file() for filename in weights):
            raise ValueError("Pinned Qwen model weights are incomplete")
        if shutil.disk_usage(RUNS.parent).free < 8 * 1024**3:
            raise ValueError("Need at least 8 GiB free after caching the model")
    print(json.dumps(dict(python=sys.version.split()[0], versions=versions, training_rows=len(pool),
                          evaluations={k: s["rows"] for k, s in p["splits"].items()}, gpu=gpu), indent=2))


def require_gpu_pilot():
    import torch
    pilot = read(REPORTS / "gpu-resume-pilot.json")
    expected = dict(device="cuda", protocol_sha256=sha(PROTOCOL), rollout_matches=True,
                    prompt_plan_matches=True, gpu_name=torch.cuda.get_device_name(),
                    torch_version=torch.__version__, cuda=torch.version.cuda)
    if any(pilot.get(key) != value for key, value in expected.items()):
        raise ValueError("GPU resume pilot does not match the current protocol or environment")


def run_stage(p, v4, target):
    import analyze
    import curve_eval
    import train

    targets = p["phase_targets"]
    if target not in targets:
        raise ValueError(f"Stage target must be one of {targets}")
    index = targets.index(target)
    if index:
        prior = read(RUNS / "analysis" / str(targets[index-1]) / "results.json")
        if (prior.get("protocol_sha256") != sha(PROTOCOL)
                or prior["target"] != targets[index-1] or prior["decision"]["action"] != "continue"):
            raise ValueError("Frozen rule does not authorize this extension")
    require_gpu_pilot()
    for seed in p["seeds"]:
        train.run(p, v4, seed, target)
    for split in p["splits"]:
        curve_eval.run(p, v4, split, target)
    analyze.run(p, v4, target)


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("preflight").add_argument("--gpu", action="store_true")
    modes.add_parser("resume-check").add_argument("--gpu", action="store_true")
    session = modes.add_parser("init-session")
    for name in ("hourly-yuan", "cap-yuan", "started-at-unix"):
        session.add_argument(f"--{name}", type=float, required=True)
    session.add_argument("--prior-spend-yuan", type=float, default=0)
    modes.add_parser("raise-cap").add_argument("--cap-yuan", type=float, required=True)
    for mode in ("stage", "report"):
        modes.add_parser(mode).add_argument("--target", type=int, required=True)
    args = parser.parse_args()
    p, v4 = locked_protocol()
    if args.mode == "preflight":
        preflight(p, v4, args.gpu)
    elif args.mode == "init-session":
        with gpu_lock():
            init_session(args)
    elif args.mode == "raise-cap":
        with gpu_lock():
            raise_cap(args.cap_yuan)
    elif args.mode == "resume-check":
        from resume_check import run
        if args.gpu:
            with gpu_lock():
                preflight(p, v4, gpu=True)
                run(p, v4, gpu=True)
        else:
            run(p, v4, gpu=False)
    elif args.mode == "stage":
        with gpu_lock():
            run_stage(p, v4, args.target)
    else:
        if args.target not in p["phase_targets"]:
            raise ValueError("Report requires a frozen phase target")
        from analyze import run
        run(p, v4, args.target)


if __name__ == "__main__":
    main()
