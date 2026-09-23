"""Exercise YAML submission at the process boundary without allocating a GPU."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def submission(tmp_path):
    repo = tmp_path / "checkout with spaces"
    helper = repo / "scripts/tools/submit_wuji_training.py"
    helper.parent.mkdir(parents=True)
    shutil.copy(ROOT / "scripts/tools/submit_wuji_training.py", helper)
    shutil.copy(ROOT / "train_hand_hybrid_sac_morphology.sbatch", repo)
    config = yaml.safe_load((ROOT / "configs/wuji/fixed8_palm0_ppo_skynet.yaml").read_text())
    config["env"]["OUTPUT_ROOT"] = "${REPO_ROOT}/test run"
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "sbatch"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "keys = ['NUM_MORPHOLOGIES', 'TRAIN_ENVS_PER_MORPHOLOGY', 'PPO_CHECKPOINT', "
        "'FIXED_REFERENCE', 'POPULATION', 'SBATCH_PARTITION', 'WANDB_API_KEY', "
        "'DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST', 'WANDB_RUN_NAME']\n"
        "Path('received.json').write_text(json.dumps({'args': sys.argv[1:], "
        "'cwd': os.getcwd(), 'env': {k: os.environ.get(k) for k in keys}}))\n"
        "print('123456;test-cluster')\n"
    )
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}",
               NUM_MORPHOLOGIES="99", TRAIN_ENVS_PER_MORPHOLOGY="7",
               PPO_CHECKPOINT="old.pt", FIXED_REFERENCE="old.npz", POPULATION="99",
               SBATCH_PARTITION="old-partition", WANDB_API_KEY="test-only-not-a-real-key",
               DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST="old-manifest.json")

    def run(*args):
        return subprocess.run([sys.executable, str(helper), str(config_path), *args],
                              cwd=tmp_path, env=env, text=True, capture_output=True)

    return repo, config_path, config, run


def test_yaml_submission_preserves_experiment_and_ignores_stale_overrides(submission):
    repo, _, config, run = submission
    result = run()
    assert result.returncode == 0, result.stderr
    received = json.loads((repo / "received.json").read_text())
    assert received["cwd"] == str(repo)
    assert received["env"]["NUM_MORPHOLOGIES"] == "8"
    assert received["env"]["TRAIN_ENVS_PER_MORPHOLOGY"] == "512"
    for key in ("PPO_CHECKPOINT", "FIXED_REFERENCE", "POPULATION", "SBATCH_PARTITION",
                "DEXCODESIGN_MORPHOLOGY_BATCH_MANIFEST"):
        assert received["env"][key] is None
    assert received["env"]["WANDB_API_KEY"] == "test-only-not-a-real-key"
    assert f"--partition={config['slurm']['partition']}" in received["args"]
    assert "--export=ALL" in received["args"]
    assert received["args"][-5:] == [str(repo / "train_hand_hybrid_sac_morphology.sbatch"),
                                    "--seed", "42", "--ppo-episode-log-window-steps", "1600"]
    saved = json.loads((repo / "test run/submission.json").read_text())
    assert saved["job_id"] == "123456"
    assert saved["env"]["SAC_UPDATES"] == "0"
    assert saved["env"]["PPO_CYCLES_PER_GENERATION"] == "64"
    assert saved["env"]["WANDB_RUN_NAME"] == saved["run_name"]
    assert "WANDB_API_KEY" not in saved["env"]
    assert (repo / "test run/submission.yaml").is_file()
    assert (repo / "slurm_logs").is_dir()


def test_dry_run_has_no_submission_or_filesystem_side_effects(submission):
    repo, _, _, run = submission
    result = run("--dry-run")
    assert result.returncode == 0, result.stderr
    assert "no job submitted" in result.stdout
    assert not (repo / "received.json").exists()
    assert not (repo / "test run").exists()
    assert not (repo / "slurm_logs").exists()


@pytest.mark.parametrize("section,key,value", [
    ("env", "PPO_CYLCES_PER_GENERATION", 64),
    ("env", "BANK_ROOT", "${UNKNOWN}/bank"),
    ("slurm", "partiton", "overcap"),
])
def test_invalid_config_fails_before_submission(submission, section, key, value):
    repo, config_path, config, run = submission
    config[section][key] = value
    config_path.write_text(yaml.safe_dump(config))
    result = run()
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert not (repo / "received.json").exists()
