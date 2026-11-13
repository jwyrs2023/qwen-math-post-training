"""Generate the immutable V5 protocol from audited, already-local V4 inputs."""

from common import PROJECT, PROTOCOL, ROOT, V4, immutable, sha, training_rows, v4_protocol


def main():
    v4 = v4_protocol()
    validation = V4 / "data/validation.json"
    svamp = V4 / "data/svamp-heldout.json"
    for path, key in ((validation, "validation"), (svamp, "holdout")):
        if sha(path) != v4[key]["sha256"]:
            raise ValueError(f"V4 data hash mismatch: {path}")
    paths = [
        validation, svamp,
        ROOT / "data/raw/train.json", ROOT / "data/raw/test.json",
        ROOT / "data/manifests/splits_clean.json", ROOT / "data/manifests/splits_clean.audit.json",
        ROOT / "prompts/math_v1.txt",
        V4 / "scripts/runtime.py", V4 / "scripts/prepare_dpo.py",
    ]
    paths += sorted((ROOT / "src/math_posttrain").glob("*.py"))
    paths += sorted((PROJECT / "scripts").glob("*.py"))
    pool = training_rows()
    if len(pool) != 1999:
        raise ValueError("Frozen V4 training pool does not have 1,999 rows")
    immutable(PROTOCOL, dict(
        study="v5-standard-grpo-optimizer-step-curve",
        v4_protocol_sha256=sha(V4 / "reports/evaluation-protocol.json"),
        input_sha256={str(path.relative_to(ROOT)): sha(path) for path in paths},
        versions=dict(torch="2.9.1", transformers="4.57.3", trl="0.26.0", peft="0.18.0",
                      datasets="4.4.1", accelerate="1.12.0"),
        seeds=[42, 43, 44], training_rows=len(pool), max_steps=2560,
        phase_targets=[1280, 1600, 1920, 2240, 2560],
        splits=dict(
            gsm8k=dict(path=str(validation.relative_to(ROOT)), sha256=sha(validation), rows=500,
                       steps=[80, 160, 240, 320, 480, 640, 960, 1280, 1600, 1920, 2240, 2560]),
            svamp=dict(path=str(svamp.relative_to(ROOT)), sha256=sha(svamp), rows=854,
                       steps=[160, 320, 640, 960, 1280, 1600, 1920, 2240, 2560]),
        ),
        training=dict(
            bf16=True, learning_rate=5e-6, lr_scheduler_type="constant",
            per_device_train_batch_size=4, gradient_accumulation_steps=4,
            steps_per_generation=4, num_generations=4, shuffle_dataset=False,
            max_completion_length=512, temperature=0.8, top_p=1.0, top_k=0,
            use_vllm=False, beta=0.02, num_iterations=1, loss_type="grpo",
            scale_rewards="group", gradient_checkpointing=True,
            logging_steps=1, logging_nan_inf_filter=False, report_to="none",
            save_strategy="steps", save_steps=2560, save_only_model=False, remove_unused_columns=False,
        ),
        bootstrap_draws=100000, bootstrap_seed=43,
        decision=dict(onset_pp=1.0, plateau_pp=1.0, first_window_start=640, window_steps=320,
                      consecutive_windows=2, simultaneous_method="Bonferroni paired percentile bootstrap",
                      joint_stop="Same two consecutive windows ending at the current target on both sets."),
        independent_confirmation=None,
    ))
    print(f"{PROTOCOL.name}: {sha(PROTOCOL)}")


if __name__ == "__main__":
    main()
