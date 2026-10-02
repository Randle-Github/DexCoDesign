"""CPU-only batch preview of canonical MANO joints and original object meshes.

This intentionally bypasses IK and Isaac.  It uses a fixed world-space camera
per clip and records the canonical frame alongside the original source frame.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is not None:
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plot_hand_object_geometry import ROOT, draw_skeleton, load_mesh, world_mesh
from render_hand_object_geometry_video import draw_ground, light_mesh, scene_limits, view_points


def load_record(record: dict, max_faces: int):
    data = np.load(ROOT / record["trajectory"], allow_pickle=False)
    source_meshes = [load_mesh(spec["path"], float(spec["scale"])) for spec in record["objects"]]
    meshes = [light_mesh(mesh, max_faces=max_faces) for mesh in source_meshes]
    return data, source_meshes, meshes


def ground_and_offset(record: dict, data, source_meshes) -> tuple[float | None, str, float]:
    start = int(record["clip_start"])
    roots = data["object_root_pose_wxyz"]
    dataset = record["dataset"]
    if dataset == "dexterhand":
        offset = float(world_mesh(source_meshes[0], roots[start, 0])[:, 2].max())
        return 0.0, "first-clip-frame mesh bottom z=0", offset
    if dataset == "arctic":
        # These are the separately tabletop-aligned ARCTIC trajectories.
        return 0.0, "aligned tabletop z=0", 0.0
    if dataset == "taco":
        support = min(
            float(world_mesh(mesh, roots[start, spec["object_index"]])[:, 2].min())
            for mesh, spec in zip(source_meshes, record["objects"])
        )
        return support, f"first-frame support proxy z={support:.2f} m; not calibrated", 0.0
    if dataset == "hocap":
        return 0.0, "world z=0 support proxy; not calibrated", 0.0
    # GigaHands has no verified gravity/table transform in this preview.
    # Its 10 sampled objects have first-frame minimum Z spanning 0.007-0.326 m.
    return None, "source XYZ; gravity/table not calibrated", 0.0


def posed_vertices(mesh, spec, data, frame: int, dataset: str, dexter_offset: float):
    object_index = int(spec["object_index"])
    angle = 0.0
    if dataset == "arctic" and spec["part"] == "top":
        # world_mesh applies ARCTIC's official local -Z axis exactly once.
        # Ignore stale selection manifests that contain articulation_sign=-1.
        angle = float(data["object_joint_positions_rad"][frame, object_index, 0])
    vertices = world_mesh(mesh, data["object_root_pose_wxyz"][frame, object_index], angle)
    return view_points(vertices, dataset, dexter_offset)


def render(record: dict, output_root: Path, max_faces: int, max_frames: int) -> dict:
    dataset = record["dataset"]
    sample_id = record["sample_id"]
    destination = output_root / dataset / f"{sample_id}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    data, source_meshes, meshes = load_record(record, max_faces)
    ground_z, ground_label, dexter_offset = ground_and_offset(record, data, source_meshes)
    start, stop = int(record["clip_start"]), int(record["clip_stop"])
    stride = max(1, int(np.ceil((stop - start) / max_frames)))
    frames = range(start, stop, stride)
    fps = float(data["fps"])
    xlim, ylim, zlim = scene_limits(data, meshes, dataset, frames, ground_z, dexter_offset)

    fig = plt.figure(figsize=(7.2, 6.3), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    writer = FFMpegWriter(fps=fps / stride, codec="libx264", bitrate=1200, extra_args=["-pix_fmt", "yuv420p"])
    hands, valid = data["hand_joints_m"], data["hand_valid"]
    with writer.saving(fig, str(destination), dpi=100):
        for frame in frames:
            ax.cla()
            if ground_z is not None:
                draw_ground(ax, xlim, ylim, ground_z, ground_label)
            for side, color, label in ((0, "#265bb4", "left"), (1, "#c43b3b", "right")):
                if valid[frame, side] and np.isfinite(hands[frame, side]).all():
                    draw_skeleton(ax, view_points(hands[frame, side], dataset, dexter_offset), color, label)
            for index, (mesh, spec) in enumerate(zip(meshes, record["objects"])):
                vertices = posed_vertices(mesh, spec, data, frame, dataset, dexter_offset)
                color = ("#d9a623", "#4e9e77", "#9a73b5")[index % 3]
                ax.add_collection3d(Poly3DCollection(vertices[mesh.faces], facecolor=color, edgecolor="none", alpha=0.90))
            ax.set(xlim=xlim, ylim=ylim, zlim=zlim, xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
            ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
            ax.view_init(elev=23, azim=-60)
            ax.grid(False)
            ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
            source_frame = int(data["frame_indices"][frame])
            suffix = " | rest cube only" if record.get("articulation_caveat") else ""
            coordinate_note = " | source XYZ; up unverified" if dataset == "gigahands" else ""
            ax.set_title(f"{sample_id[:33]} | frame {frame} / source {source_frame} | {frame / fps:.2f} s{suffix}{coordinate_note}", fontsize=9)
            writer.grab_frame()
    plt.close(fig)
    return {"dataset": dataset, "sample_id": sample_id, "path": str(destination), "frames": len(frames),
            "source_fps": fps, "render_fps": fps / stride, "stride": stride,
            "clip_start": start, "clip_stop": stop, "mesh_face_limit_per_part": max_faces,
            "ground_z": ground_z, "ground_status": ground_label,
            "dexter_source_to_up_z_offset": dexter_offset}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("all", "taco", "arctic", "hocap", "dexterhand", "gigahands"), default="all")
    parser.add_argument("--max-faces", type=int, default=5000)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="Render only the first N selected items per dataset for smoke tests")
    parser.add_argument("--sample-id", type=str, default="", help="Render only one selected sample")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())["groups"]
    groups = selection if args.dataset == "all" else {args.dataset: selection[args.dataset]}
    records = []
    failures = []
    for dataset, group in groups.items():
        if args.sample_id:
            group = [item for item in group if item["sample_id"] == args.sample_id]
        if args.limit > 0:
            group = group[: args.limit]
        for index, item in enumerate(group, 1):
            destination = args.output_root / dataset / f"{item['sample_id']}.mp4"
            if args.skip_existing and destination.is_file() and destination.stat().st_size > 1000:
                print(f"SKIP_EXISTING {destination}", flush=True)
                continue
            try:
                result = render(item, args.output_root, args.max_faces, args.max_frames)
                records.append(result)
                print(f"READY {dataset} {index}/{len(group)} {destination}", flush=True)
            except Exception as error:
                failures.append({"dataset": dataset, "sample_id": item["sample_id"], "error": repr(error)})
                print(f"FAILED {dataset} {item['sample_id']}: {error!r}", flush=True)
    audit = args.output_root / f"render_audit_{args.dataset}.json"
    audit.write_text(json.dumps({"rendered": records, "failures": failures}, indent=2) + "\n")
    print(f"SUMMARY rendered={len(records)} failed={len(failures)} audit={audit}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
