"""USD overlay utilities shared by WUJI evaluation and persistent search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom


def attach_parametric_collisions(
    candidate_usd: Path,
    template_usd: Path,
    link_names: list[str],
    transforms: list[list[list[float]]],
    link_translations: list[list[float]],
    joint_names: list[str],
    joint_local_positions: list[list[float]],
) -> None:
    """Author one candidate's affine geometry and joint-frame overlay."""
    template_stage = Usd.Stage.Open(str(template_usd))
    template_root = template_stage.GetDefaultPrim().GetPath()
    candidate_stage = Usd.Stage.Open(str(candidate_usd))
    candidate_root = candidate_stage.GetDefaultPrim().GetPath()
    if not (len(link_names) == len(transforms) == len(link_translations)):
        raise ValueError("parametric link/transform count mismatch")
    for link_name, transform, translation in zip(
        link_names, transforms, link_translations, strict=True
    ):
        template_link = template_root.AppendChild(link_name)
        if not template_stage.GetPrimAtPath(template_link):
            matches = [
                child.GetPath()
                for child in template_stage.GetPrimAtPath(template_root).GetChildren()
                if child.GetName().endswith(f"__{link_name}")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"cannot uniquely map template link {link_name}: {matches}"
                )
            template_link = matches[0]
        candidate_path = candidate_root.AppendChild(
            template_link.name
        ).AppendChild("collisions")
        candidate_link = candidate_stage.OverridePrim(candidate_path.GetParentPath())
        candidate_link.GetAttribute("xformOp:translate").Set(
            Gf.Vec3d(*translation)
        )
        collision = candidate_stage.OverridePrim(candidate_path)
        matrix = np.asarray(transform, dtype=np.float64)
        if not np.allclose(matrix, np.eye(4), atol=1.0e-12):
            xform = UsdGeom.Xformable(collision)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(Gf.Matrix4d(*matrix.reshape(-1).tolist()))
    if len(joint_names) != len(joint_local_positions):
        raise ValueError("parametric joint/position count mismatch")
    joint_scope = candidate_root.AppendChild("joints")
    template_joint_scope = template_root.AppendChild("joints")
    for joint_name, position in zip(
        joint_names, joint_local_positions, strict=True
    ):
        resolved_name = joint_name
        template_joint = template_stage.GetPrimAtPath(
            template_joint_scope.AppendChild(resolved_name)
        )
        if not template_joint:
            matches = [
                child.GetName()
                for child in template_stage.GetPrimAtPath(
                    template_joint_scope
                ).GetChildren()
                if child.GetName().endswith(f"__{joint_name}")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"cannot uniquely map template joint {joint_name}: {matches}"
                )
            resolved_name = matches[0]
        joint = candidate_stage.OverridePrim(joint_scope.AppendChild(resolved_name))
        joint.GetAttribute("physics:localPos0").Set(Gf.Vec3f(*position))
    candidate_stage.GetRootLayer().Save()


def attach_manifest(manifest: dict) -> float:
    """Attach all overlays and return wall-clock seconds."""
    import time

    template_value = manifest.get("parametric_template_usd")
    template_values = manifest.get("parametric_template_usd_paths")
    if not (template_value or template_values):
        return 0.0
    start = time.perf_counter()
    for index, usd_value in enumerate(manifest["hand_usd_paths"]):
        template = Path(
            template_value if template_values is None else template_values[index]
        ).resolve()
        attach_parametric_collisions(
            Path(usd_value).resolve(),
            template,
            manifest["parametric_link_names"][index],
            manifest["parametric_relative_transforms"][index],
            manifest["parametric_link_translations"][index],
            manifest["parametric_joint_names"][index],
            manifest["parametric_joint_local_positions"][index],
        )
    return time.perf_counter() - start


def build_hand_super_environment(
    manifest: dict,
    output: Path,
    spacing: float,
) -> tuple[Path, np.ndarray]:
    """Compose exact candidate hand USDs into one static replication source."""

    count = len(manifest["hand_usd_paths"])
    per_row = int(np.ceil(np.sqrt(count)))
    local_origins = np.zeros((count, 3), dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    root = UsdGeom.Xform.Define(stage, "/SuperEnvironment").GetPrim()
    stage.SetDefaultPrim(root)
    for index, usd_value in enumerate(manifest["hand_usd_paths"]):
        row, column = divmod(index, per_row)
        local_origins[index, :2] = (column * spacing, row * spacing)
        morph = UsdGeom.Xform.Define(
            stage, f"/SuperEnvironment/morph_{index:06d}"
        )
        morph.AddTranslateOp().Set(Gf.Vec3d(*local_origins[index]))
        hand = stage.DefinePrim(
            f"/SuperEnvironment/morph_{index:06d}/Hand", "Xform"
        )
        hand.GetReferences().AddReference(str(Path(usd_value).resolve()))
    stage.GetRootLayer().Save()
    return output.resolve(), local_origins
