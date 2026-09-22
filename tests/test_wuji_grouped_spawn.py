"""Check morphology scene setup without launching Isaac Sim."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / (
    "source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/mano_residual_env.py"
)


def method(name, namespace):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ManoResidualEnv")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def test_grouped_hands_receive_physics_overrides_before_objects_spawn():
    events = []
    props = dict(
        rigid_props=SimpleNamespace(disable_gravity=True, max_depenetration_velocity=1.0),
        articulation_props=SimpleNamespace(solver_position_iteration_count=8, solver_velocity_iteration_count=2),
        collision_props=None, mass_props=None, fixed_tendons_props=None,
        spatial_tendons_props=None, joint_drive_props=None, activate_contact_sensors=False,
    )

    def make_usd(**kwargs):
        return SimpleNamespace(**kwargs, func=lambda path, cfg: events.append(("hand", path, cfg)))

    obj = SimpleNamespace(
        rigid_props=SimpleNamespace(disable_gravity=False),
        func=lambda path, cfg, **kwargs: events.append(("object", path, cfg)),
    )
    env = SimpleNamespace(num_envs=8, cfg=SimpleNamespace(
        hand_cfg=SimpleNamespace(spawn=SimpleNamespace(**props)),
        object_cfg=SimpleNamespace(spawn=obj),
    ))
    manifest = dict(unique_morphology_count=2, hand_super_environment_usd="/tmp/hands.usd",
                    hand_super_environment_origins=[[0, 0, 0], [1, 0, 0]])
    spawn = method("_spawn_grouped_morphology_sources", dict(
        MORPHOLOGY_BATCH_MANIFEST=manifest, np=np, Path=Path, copy=copy,
        sim_utils=SimpleNamespace(UsdFileCfg=make_usd),
    ))
    spawn(env)
    assert [e[0] for e in events] == ["hand", "object"]
    for key, value in props.items():
        assert getattr(events[0][2], key) is value
    assert events[1][2].rigid_props.disable_gravity is False
    np.testing.assert_array_equal(env._morphology_local_origins, [[0, 0, 0], [1, 0, 0]])


@pytest.mark.parametrize("grouped,num_envs,mismatch", [(False, 8, True), (False, 64, False), (True, 8, False)])
def test_manifest_count_checked_before_reference_loading(grouped, num_envs, mismatch):
    initialize = method("__init__", dict(
        MORPHOLOGY_BATCH_MANIFEST={"grouped_physics_replication": grouped},
        MORPHOLOGY_BATCH_MANIFEST_PATH="batch.json", _batch_reference_paths=["ref.npz"] * 64,
    ))
    # An invalid mode is a sentinel for reaching the next initialization check.
    cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=num_envs), observation_mode="sentinel")
    message = "contains 64 environment rows" if mismatch else "Unknown observation_mode: sentinel"
    with pytest.raises(ValueError, match=message):
        initialize(SimpleNamespace(), cfg)
