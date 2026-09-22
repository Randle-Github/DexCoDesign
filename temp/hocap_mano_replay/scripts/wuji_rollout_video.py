"""Optional, streamed video diagnostics for existing headless PhysX rollouts.

Frames are encoded as they arrive, rather than accumulated in RAM. Recording
does not reset/step the environment and never contributes an optimizer reward.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


def encoder_environment(executable: str) -> dict[str, str]:
    """Do not load Isaac/other Conda environments' codecs into the encoder."""
    environment = os.environ.copy()
    prefix = Path(executable).resolve().parent.parent
    if (prefix / "conda-meta").is_dir():
        environment["LD_LIBRARY_PATH"] = str(prefix / "lib")
    else:
        environment.pop("LD_LIBRARY_PATH", None)
    return environment


def make_hand_visible(stage, hand_path: str) -> int:
    """Display collision surfaces on links without their own visual meshes."""
    from pxr import Usd, UsdGeom, UsdShade, Vt

    hand = stage.GetPrimAtPath(hand_path)
    if not hand:
        raise ValueError(f"Cannot locate recorded hand at {hand_path}")
    proxies = list(Usd.PrimRange(hand, Usd.TraverseInstanceProxies()))

    def geometry_owner(prim, scope_name):
        current = prim.GetParent()
        while current and current.GetPath().HasPrefix(hand.GetPath()):
            if current.GetName() == scope_name:
                return current.GetParent().GetPath()
            current = current.GetParent()
        return None

    # Some generated hands have palm/base visuals but collision-only fingers.
    # A visual on one link must not suppress the fallback for the entire hand.
    visual_links = {
        owner
        for prim in proxies
        if prim.IsA(UsdGeom.Mesh)
        and (owner := geometry_owner(prim, "visuals")) is not None
    }

    def needs_collision_display(prim):
        if not prim.IsA(UsdGeom.Mesh):
            return False
        owner = geometry_owner(prim, "collisions")
        return owner is not None and owner not in visual_links

    if visual_links and not any(needs_collision_display(p) for p in proxies):
        return 0
    # Instance-proxy attributes cannot be overridden. Deinstance only this
    # recorded hand's geometry in the current scene layer, never its asset file.
    pending = [hand]
    while pending:
        prim = pending.pop()
        if prim.IsInstance() or prim.IsInstanceable():
            prim.SetInstanceable(False)
        pending.extend(prim.GetChildren())
    count = 0
    for prim in Usd.PrimRange(hand):
        if not needs_collision_display(prim):
            continue
        current = prim
        while current and current.GetPath().HasPrefix(hand.GetPath()):
            imageable = UsdGeom.Imageable(current)
            if imageable:
                imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                imageable.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
            current = current.GetParent()
        UsdGeom.Mesh(prim).GetDisplayColorAttr().Set(Vt.Vec3fArray([(0.22, 0.66, 0.90)]))
        UsdShade.MaterialBindingAPI.Apply(prim).UnbindAllBindings()
        count += 1
    if not count:
        raise ValueError(f"No renderable hand surfaces at {hand_path}")
    return count


def generation_video_spec(output_root: Path, generation: int, args) -> dict | None:
    if not args.video or generation % args.video_interval:
        return None
    return {
        "generation": generation,
        "candidate_index": args.video_candidate_index,
        "replica_index": 0,
        "max_steps": args.video_length,
        "stride": args.video_stride,
        "width": args.video_width,
        "height": args.video_height,
        "path": str(
            (
                output_root
                / f"generation_{generation:03d}"
                / "videos"
                / f"candidate_{args.video_candidate_index:06d}_rollout_000.mp4"
            ).resolve()
        ),
        "semantics": "actual training rollout, recorded before steps; not selected by best reward",
    }


class StreamedRolloutVideo:
    def __init__(self, spec: dict, step_dt: float, env_index: int):
        self.spec = dict(spec)
        self.env_index = env_index
        self.path = Path(spec["path"])
        self.partial_path = self.path.with_suffix(".partial.mp4")
        self.fps = 1.0 / (step_dt * spec["stride"])
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("video requires a positive control-step duration")
        self.frames = 0
        self.error = None
        self.process = None
        self.stderr = None
        self.result = None

    def warm_up(self, env) -> None:
        """Initialize the RGB render product without advancing physics."""
        try:
            import isaaclab.sim as sim_utils

            raw = env.unwrapped
            hand_path = raw.cfg.hand_cfg.prim_path.replace("env_.*", f"env_{self.env_index}")
            self.spec["collision_surfaces_shown"] = make_hand_visible(sim_utils.get_current_stage(), hand_path)
            env.unwrapped.render()
            for _ in range(3):
                env.unwrapped.sim.render()
        except Exception as exc:
            self.error = repr(exc)

    def capture(self, render_frame, step: int) -> None:
        if self.error or self.result is not None or step >= self.spec["max_steps"] or step % self.spec["stride"]:
            return
        try:
            frame = np.asarray(render_frame())
            expected = (self.spec["height"], self.spec["width"], 3)
            if frame.shape != expected or frame.dtype != np.uint8:
                raise ValueError(f"Expected uint8 RGB frame {expected}, got {frame.shape}/{frame.dtype}")
            if self.process is None:
                ffmpeg = shutil.which("ffmpeg")
                if ffmpeg is None:
                    raise RuntimeError("ffmpeg is required for --video")
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.stderr = self.path.with_suffix(".ffmpeg.log").open("wb")
                self.process = subprocess.Popen(
                    [
                        ffmpeg,
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-video_size",
                        f"{self.spec['width']}x{self.spec['height']}",
                        "-framerate",
                        str(self.fps),
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-threads",
                        "1",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "23",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(self.partial_path),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=self.stderr,
                    env=encoder_environment(ffmpeg),
                )
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
            self.frames += 1
        except Exception as exc:
            self.error = repr(exc)

    def finish(self) -> dict:
        """Always reap our encoder; failures remain diagnostics, not training failures."""
        if self.result is not None:
            return self.result
        try:
            if self.process is not None:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    code = self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                    raise RuntimeError("video encoder timed out")
                if code:
                    raise RuntimeError(f"ffmpeg exited {code}; see {self.path.with_suffix('.ffmpeg.log')}")
            if not self.error and self.frames:
                self.partial_path.replace(self.path)
        except Exception as exc:
            self.error = repr(exc)
        finally:
            if self.stderr is not None:
                self.stderr.close()
        self.result = {
            **self.spec,
            "fps": self.fps,
            "frames": self.frames,
            "duration_seconds": self.frames / self.fps,
            "error": self.error,
        }
        if self.error or not self.frames:
            self.result["path"] = None
        sidecar = self.path.with_suffix(".json")
        try:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps(self.result, indent=2) + "\n")
        except OSError as exc:
            self.result["metadata_write_error"] = repr(exc)
        return self.result


def close_rgb_render_product(env) -> list[str]:
    """Detach the extra RGB resources before environment/stage cleanup."""
    errors = []
    raw = env.unwrapped
    annotator = getattr(raw, "_rgb_annotator", None)
    product = getattr(raw, "_render_product", None)
    if annotator is not None:
        try:
            annotator.detach()
        except Exception as exc:
            errors.append(repr(exc))
        del raw._rgb_annotator
    if product is not None:
        try:
            product.destroy()
        except Exception as exc:
            errors.append(repr(exc))
        del raw._render_product
    return errors


def video_wandb_metrics(records: list[dict], wandb) -> dict:
    """Merge media into the existing generation log, without a second step/commit."""
    for record in records:
        if record.get("path") and Path(record["path"]).is_file():
            caption = (
                f"Generation {record['generation']}, candidate {record['candidate_index']}, "
                f"rollout {record['replica_index']}; actual training rollout, not the best-candidate selection"
            )
            return {
                "Video / Training rollout": wandb.Video(record["path"], format="mp4", caption=caption),
                "Video / Candidate index": record["candidate_index"],
                "Video / Replica index": record["replica_index"],
            }
    return {}
