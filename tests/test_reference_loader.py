"""Replicated reference loading closes archives and preserves candidate order."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reference_loader",
    ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/reference_loader.py",
)
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


def test_4096_replicas_load_32_archives_once_and_close(tmp_path, monkeypatch):
    paths = []
    for index in range(32):
        path = tmp_path / f"reference_{index}.npz"
        np.savez_compressed(path, hand_q=np.full((3, 26), index), joint_names=["joint"])
        paths.append(path)
    original_load = np.load
    archives = []

    def tracked_load(*args, **kwargs):
        # At most one archive should be open, regardless of replica count.
        assert all(archive.zip is None for archive in archives)
        archive = original_load(*args, **kwargs)
        archives.append(archive)
        return archive

    monkeypatch.setattr(loader.np, "load", tracked_load)
    references = loader.load_references([path for path in paths for _ in range(128)])
    assert len(references) == 4096
    assert len(archives) == 32
    assert all(archive.zip is None for archive in archives)
    for index, reference in enumerate(references):
        np.testing.assert_array_equal(reference["hand_q"], np.full((3, 26), index // 128))
        assert reference is references[(index // 128) * 128]
    # Materialized data remains usable after the source file is removed.
    paths[0].unlink()
    assert references[0]["joint_names"].tolist() == ["joint"]


def test_archive_closed_on_array_read_error(tmp_path, monkeypatch):
    path = tmp_path / "invalid.npz"
    np.savez(path, unsafe=np.array([object()], dtype=object))
    original_load = np.load
    archives = []

    def tracked_load(*args, **kwargs):
        archive = original_load(*args, **kwargs)
        archives.append(archive)
        return archive

    monkeypatch.setattr(loader.np, "load", tracked_load)
    with pytest.raises(ValueError, match="Object arrays"):
        loader.load_references([path])
    assert archives[0].zip is None


def test_replica_bank_preserves_order_and_rejects_geometry_mismatch():
    manifest = {
        'grouped_physics_replication': True,
        'morphology_indices': [7, 3, 7, 3],
        'reference_paths': ['a', 'b', 'a', 'b'],
        'parametric_joint_local_positions': [[[1]], [[2]], [[1]], [[2]]],
    }
    assert loader.reference_layout(manifest, 4) == ([0, 1], [0, 1, 0, 1])
    manifest['parametric_joint_local_positions'][2] = [[9]]
    with pytest.raises(ValueError, match='disagree'):
        loader.reference_layout(manifest, 4)
    assert loader.reference_layout(None, 1) == ([0], [0])


def test_compact_reference_gather_matches_expanded_for_resets_and_phases():
    import ast
    from collections.abc import Sequence
    from types import SimpleNamespace
    import torch
    env_path = ROOT / 'source/isaaclab_tasks/isaaclab_tasks/direct/mano_residual/mano_residual_env.py'
    tree = ast.parse(env_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ManoResidualEnv')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_reference_at')
    namespace = {'torch': torch, 'Sequence': Sequence}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(env_path), 'exec'), namespace)
    gather = namespace['_reference_at']
    mapping = torch.tensor([0, 1, 0, 1])
    bank = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    expanded = bank[mapping]
    env = SimpleNamespace(_morphology_batch=True, num_envs=4, device='cpu', _reference_indices=mapping)
    phases = torch.tensor([4, 0, 1, 3])
    assert torch.equal(gather(env, bank, phases), expanded[torch.arange(4), phases])
    for ids in ([3, 0], torch.tensor([3, 0])):
        assert torch.equal(gather(env, bank, phases[[3, 0]], ids), expanded[[3, 0], phases[[3, 0]]])
    env._morphology_batch = False
    assert torch.equal(gather(env, bank[0], phases), bank[0][phases])
