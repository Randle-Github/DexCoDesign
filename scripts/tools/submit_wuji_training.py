# Copyright (c) 2026, The DexCoDesign Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Submit the existing WUJI batch launcher from YAML, without importing Isaac.

Requires PyYAML in the login-node Python. No shell expressions are evaluated.
Experiment overrides come from YAML or batch defaults, not previous exports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from string import Template

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_SCRIPT = REPO_ROOT / "train_hand_hybrid_sac_morphology.sbatch"
SLURM_OPTIONS = {
    "partition", "account", "qos", "gpus", "cpus-per-task", "mem", "time",
    "nodelist", "exclude", "constraint", "job-name",
}


def mapping(value, name: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value


def scalar(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if not isinstance(value, (str, int, float)):
        raise ValueError(f"Expected a scalar value, got {value!r}")
    return str(value)


def prepare_submission(config: dict) -> tuple[list[str], dict[str, str], dict]:
    config = mapping(config, "config")
    unknown = set(config) - {"run_name_prefix", "slurm", "env", "args"}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    prefix = config.get("run_name_prefix", "wuji_skynet")
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError("run_name_prefix must contain only letters, digits, '.', '_', or '-'")
    run_name = f"{prefix}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}"
    substitutions = {"REPO_ROOT": str(REPO_ROOT), "RUN_NAME": run_name}

    # Read the existing launcher's ${NAME:-default}, ${NAME-default}, etc.
    # This includes aliases such as POPULATION and NUM_MORPHOLOGIES, so neither
    # can silently leak into a YAML submission from an earlier terminal export.
    overrides = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)(?::[-+]|[-+])", BATCH_SCRIPT.read_text()))
    overrides -= {"SLURM_JOB_ID", "SLURM_SUBMIT_DIR", "TMPDIR", "LD_LIBRARY_PATH"}
    configured = mapping(config.get("env", {}), "env")
    unknown = set(configured) - (overrides - {"REPO_ROOT_OVERRIDE"})
    if unknown:
        raise ValueError(f"Unknown or reserved batch environment keys: {sorted(unknown)}")
    env = {
        key: value for key, value in os.environ.items()
        if key not in overrides and not key.startswith(("SBATCH_", "DEXCODESIGN_"))
    }
    resolved = {"REPO_ROOT_OVERRIDE": str(REPO_ROOT)}
    for key, value in configured.items():
        if value is not None:
            resolved[key] = Template(scalar(value)).substitute(substitutions)
    resolved.setdefault("OUTPUT_ROOT", str(REPO_ROOT / "artifacts/wuji_sac" / run_name))
    resolved.setdefault("WANDB_RUN_NAME", run_name)
    if not resolved["OUTPUT_ROOT"]:
        raise ValueError("OUTPUT_ROOT must not be empty")
    output = Path(resolved["OUTPUT_ROOT"]).expanduser()
    if not output.is_absolute():
        output = REPO_ROOT / output
    resolved["OUTPUT_ROOT"] = str(output.resolve())
    env.update(resolved)

    slurm = mapping(config.get("slurm", {}), "slurm")
    unknown = set(slurm) - SLURM_OPTIONS
    if unknown:
        raise ValueError(f"Unsupported Slurm options: {sorted(unknown)}")
    command = ["sbatch", "--parsable", "--export=ALL", f"--chdir={REPO_ROOT}"]
    for key, value in slurm.items():
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"slurm.{key} must be a string or integer")
        command.append(f"--{key}={value}")
    args = config.get("args", [])
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        raise ValueError("args must be a list of strings (quote numeric arguments)")
    command.extend([str(BATCH_SCRIPT), *args])
    record = {"run_name": run_name, "env": resolved, "command": command}
    return command, env, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Experiment YAML")
    parser.add_argument("--dry-run", action="store_true", help="Show resolved settings without submitting or writing files")
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text())
        command, env, record = prepare_submission(config)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    print(yaml.safe_dump(record, sort_keys=False), flush=True)
    if args.dry_run:
        print("Dry run: no job submitted.")
        return

    output = Path(record["env"]["OUTPUT_ROOT"])
    output.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "slurm_logs").mkdir(exist_ok=True)
    # Keep the exact input and resolved arguments with the run for reproducibility.
    # Refuse to overwrite another submission's record if OUTPUT_ROOT was reused.
    with (output / "submission.yaml").open("x") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    record_path = output / "submission.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"sbatch failed: {exc.stderr.strip()}") from exc
    if result.stderr:
        print(result.stderr.strip())
    job_id = result.stdout.strip().split(";", 1)[0]
    record["job_id"] = job_id
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Submitted job {job_id}; results: {output}")
    print(f"squeue -j {shlex.quote(job_id)}")
    print(f"tail -f {shlex.quote(str(REPO_ROOT / 'slurm_logs' / f'slurm-wuji-hybrid-sac-{job_id}.out'))}")


if __name__ == "__main__":
    main()
