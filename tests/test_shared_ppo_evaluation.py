"""Exercise the production evaluator without starting Isaac Sim."""
import ast
import inspect
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


path = Path(__file__).resolve().parents[1] / "temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.py"
node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_shared_policy")
namespace = dict(torch=torch, np=np, inspect=inspect, time=time, Runner=object)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
evaluate = namespace["evaluate_shared_policy"]


class Env:
    num_envs = 6
    device = "cpu"
    _reference_length = 4

    def reset(self):
        self.steps = 0
        return torch.zeros(6, 1), {}

    def step(self, actions):
        assert torch.equal(actions, torch.zeros(6, 1))  # mean, not sampled actions
        self.steps += 1
        self._last_pose_tracking_reward = torch.tensor([1., 10., 2., 20., 1000., 2000.])
        self._last_contact_reward = torch.zeros(6)
        self._last_pinch_contact = torch.zeros(6, dtype=torch.bool)
        self._last_evaluated_phase = torch.full((6,), self.steps - 1)
        self._object_position_error = torch.zeros(6)
        self._object_rotation_error = torch.zeros(6)
        # Selected replicas finish at different steps; unselected replicas live longer.
        done = torch.tensor([self.steps >= n for n in [1, 1, 2, 2, 4, 4]])
        return torch.zeros(6, 1), None, done, torch.zeros(6, dtype=torch.bool), {}


class Agent:
    def set_running_mode(self, mode):
        assert mode == "eval"

    def act(self, observations, timestep, timesteps):
        return torch.ones(6, 1), {"mean_actions": torch.zeros(6, 1)}


def run(count):
    env = Env()
    # Interleaved mapping also checks that selection is per morphology, not contiguous slices.
    manifest = {"morphology_indices": [3, 7, 3, 7, 3, 7], "vectors": [[0.]] * 6}
    rows, _ = evaluate(env, env, SimpleNamespace(agent=Agent()), manifest, count)
    return env, rows


def test_subset_means_only_first_episode_and_stops_without_waiting_for_others():
    env, rows = run(2)
    assert env.steps == 2
    assert [r["total_reward"] for r in rows] == [2.5, 25.]
    assert [r["replicas"] for r in rows] == [2, 2]
    assert [r["environment_steps"] for r in rows] == [3, 3]


def test_default_scores_all_replicas():
    env, rows = run(None)
    assert env.steps == 4
    assert [r["replicas"] for r in rows] == [3, 3]
    assert rows[0]["total_reward"] == pytest.approx((1 + 4 + 4000) / 3)


@pytest.mark.parametrize("count", [0, -1, 4])
def test_invalid_counts(count):
    with pytest.raises(ValueError, match="evaluation count"):
        run(count)
