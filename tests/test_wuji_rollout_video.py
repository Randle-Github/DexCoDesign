"""Headless rollout videos are optional diagnostics, never additional episodes."""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "wuji_rollout_video", ROOT / "temp/hocap_mano_replay/scripts/wuji_rollout_video.py"
)
video_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(video_module)


def args(**changes):
    values = dict(
        video=True,
        video_interval=10,
        video_candidate_index=1,
        video_length=10,
        video_stride=2,
        video_width=16,
        video_height=12,
    )
    return SimpleNamespace(**(values | changes))


def test_schedule_and_candidate_identity(tmp_path):
    assert video_module.generation_video_spec(tmp_path, 1, args()) is None
    assert video_module.generation_video_spec(tmp_path, 0, args(video=False)) is None
    spec = video_module.generation_video_spec(tmp_path, 10, args())
    assert spec["candidate_index"] == 1 and spec["replica_index"] == 0
    assert Path(spec["path"]) == tmp_path / "generation_010/videos/candidate_000001_rollout_000.mp4"


def test_evaluator_visibility_is_gui_only_and_does_not_step_physics():
    path = ROOT / "temp/hocap_mano_replay/isaaclab/evaluate_wuji_morphology_physx_batch.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_gui_hand_visibility")
    calls = []
    env = SimpleNamespace(unwrapped=SimpleNamespace(
        cfg=SimpleNamespace(hand_cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Hand")),
        sim=SimpleNamespace(render=lambda: calls.append("render")),
    ))
    namespace = {
        "sim_utils": SimpleNamespace(get_current_stage=lambda: "scene"),
        "make_hand_visible": lambda stage, path: calls.append((stage, path)) or 21,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    show = namespace["prepare_gui_hand_visibility"]
    assert show(env, 4, True) == [] and calls == []
    assert show(env, 4, False) == [21] * 4
    assert calls == [("scene", f"/World/envs/env_{i}/Hand") for i in range(4)] + ["render"]


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_video_finalizes_and_releases_render_resources(fail):
    path = ROOT / "temp/hocap_mano_replay/isaaclab/evaluate_wuji_morphology_physx_batch.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluation_video")
    calls = []

    class Recorder:
        def __init__(self, spec, dt, index):
            assert dt == 1 / 30 and index == 0
        def warm_up(self, env):
            calls.append("warm_up")
        def finish(self):
            calls.append("finish")
            return {"frames": 3}

    namespace = {
        "contextmanager": contextmanager, "StreamedRolloutVideo": Recorder, "json": json,
        "close_rgb_render_product": lambda env: calls.append("cleanup"),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    record = namespace["evaluation_video"]
    env = SimpleNamespace(unwrapped=SimpleNamespace(step_dt=1 / 30))
    records = []
    with record(env, None, records) as video:
        assert video is None and calls == []
    try:
        with record(env, {"candidate_index": 0}, records) as video:
            assert isinstance(video, Recorder)
            if fail:
                raise RuntimeError("rollout failed")
    except RuntimeError:
        assert fail
    assert calls == ["warm_up", "finish", "cleanup"]
    assert records == [{"frames": 3}]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires ffmpeg/ffprobe")
def test_streamed_mp4_has_correct_stride_duration_and_frame_count(tmp_path):
    spec = video_module.generation_video_spec(tmp_path, 0, args())
    recorder = video_module.StreamedRolloutVideo(spec, step_dt=0.02, env_index=4)
    calls = []

    def render():
        calls.append(1)
        return np.full((12, 16, 3), 100, dtype=np.uint8)

    for step in range(15):
        recorder.capture(render, step)
    result = recorder.finish()
    assert result["error"] is None and result["frames"] == 5
    assert len(calls) == 5 and result["fps"] == 25 and result["duration_seconds"] == 0.2
    assert recorder.finish() == result
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            result["path"],
        ],
        check=True,
        capture_output=True,
        text=True,
        env=video_module.encoder_environment(shutil.which("ffprobe")),
    )
    assert json.loads(probe.stdout)["streams"][0] == {"width": 16, "height": 12, "nb_read_frames": "5"}
    assert not recorder.partial_path.exists()


def test_bad_render_is_nonfatal_diagnostic(tmp_path):
    recorder = video_module.StreamedRolloutVideo(
        video_module.generation_video_spec(tmp_path, 0, args()), step_dt=0.02, env_index=0
    )
    recorder.capture(lambda: None, 0)
    result = recorder.finish()
    assert result["path"] is None and result["error"] and result["frames"] == 0


def test_rgb_resources_are_detached_destroyed_and_idempotent():
    calls = []
    raw = SimpleNamespace(
        _rgb_annotator=SimpleNamespace(detach=lambda: calls.append("detach")),
        _render_product=SimpleNamespace(destroy=lambda: calls.append("destroy")),
    )
    env = SimpleNamespace(unwrapped=raw)
    assert video_module.close_rgb_render_product(env) == []
    assert video_module.close_rgb_render_product(env) == []
    assert calls == ["detach", "destroy"]


def test_wandb_media_is_explicitly_not_best_candidate(tmp_path):
    path = tmp_path / "rollout.mp4"
    path.write_bytes(b"video stub")
    record = {"generation": 10, "candidate_index": 1, "replica_index": 0, "path": str(path)}
    wandb = SimpleNamespace(Video=lambda path, **kwargs: (path, kwargs))
    metrics = video_module.video_wandb_metrics([record], wandb)
    _, kwargs = metrics["Video / Training rollout"]
    assert kwargs["format"] == "mp4" and "not the best" in kwargs["caption"]
    assert metrics["Video / Candidate index"] == 1
    assert video_module.video_wandb_metrics([dict(record, path=None)], wandb) == {}


def evaluate_function():
    # Load only the existing rollout function, avoiding simulator startup while
    # testing that its diagnostics do not alter step/reset/reward semantics.
    source = ROOT / "temp/hocap_mano_replay/isaaclab/train_wuji_hybrid_sac_morphology.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_batch")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = dict(torch=torch, time=time, StreamedRolloutVideo=video_module.StreamedRolloutVideo)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["evaluate_batch"]


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.device = "cpu"
        self.action_dim = 1
        self._reference_length = 8
        self.steps = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def step(self, actions):
        assert torch.count_nonzero(actions) == 0
        self.steps += 1
        self._last_pose_tracking_reward = torch.tensor([1.0, 2.0])
        self._last_contact_reward = torch.tensor([2.0, 3.0])
        self._last_pinch_contact = torch.tensor([True, False])
        self._last_thumb_contact_force = torch.tensor([3.0, 4.0])
        self._last_other_finger_contact_force = torch.tensor([5.0, 6.0])
        self._last_evaluated_phase = torch.tensor([self.steps, self.steps])
        self._object_position_error = torch.tensor([0.01, 0.02])
        self._object_rotation_error = torch.tensor([0.1, 0.2])
        return None, None, torch.tensor([self.steps >= 2, self.steps >= 4]), torch.tensor([False, False]), {}

    def render(self):
        return self.steps


def test_recording_does_not_add_steps_resets_or_change_reward():
    evaluate = evaluate_function()
    manifest = dict(vectors=[[0], [1]], candidate_ids=["a", "b"])
    baseline_env = FakeEnv()
    baseline_rows, _ = evaluate(baseline_env, manifest, 0)
    video_env = FakeEnv()
    frames = []
    recorder = SimpleNamespace(
        env_index=0, warm_up=lambda env: None, capture=lambda render, step: frames.append((step, render()))
    )
    video_rows, _ = evaluate(video_env, manifest, 0, recorder)
    assert video_rows == baseline_rows
    assert video_env.resets == baseline_env.resets == 1
    assert video_env.steps == baseline_env.steps == 4
    assert frames == [(0, 0), (1, 1)]  # no auto-reset frames after replica 0 ends


def test_collision_only_visibility_does_not_change_physics_or_source_asset(tmp_path):
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    template_path = tmp_path / "hand.usda"
    template = Usd.Stage.CreateNew(str(template_path))
    root = UsdGeom.Xform.Define(template, "/Hand")
    template.SetDefaultPrim(root.GetPrim())
    scope = UsdGeom.Xform.Define(template, "/Hand/collisions")
    scope.GetVisibilityAttr().Set("invisible")
    scope.GetPurposeAttr().Set("guide")
    mesh = UsdGeom.Mesh.Define(template, "/Hand/collisions/mesh")
    points = Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.GetPointsAttr().Set(points)
    collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr(True)
    template.GetRootLayer().Save()
    original_bytes = template_path.read_bytes()

    scene = Usd.Stage.CreateInMemory()
    hand = scene.DefinePrim("/World/envs/env_4/Hand", "Xform")
    hand.GetReferences().AddReference(str(template_path))
    assert video_module.make_hand_visible(scene, str(hand.GetPath())) == 1
    displayed = UsdGeom.Mesh.Get(scene, str(hand.GetPath()) + "/collisions/mesh")
    assert displayed.ComputeVisibility() == "inherited" and displayed.ComputePurpose() == "default"
    np.testing.assert_array_equal(displayed.GetPointsAttr().Get(), points)
    assert UsdPhysics.CollisionAPI(displayed.GetPrim()).GetCollisionEnabledAttr().Get() is True
    assert template_path.read_bytes() == original_bytes


def test_system_encoder_does_not_inherit_isaac_library_path(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/some/isaac/codec/libraries")
    assert "LD_LIBRARY_PATH" not in video_module.encoder_environment("/usr/bin/ffmpeg")
