#!/usr/bin/env python3
"""Download only one HO-Cap sequence's 3D hand-joint labels.

The official labels archive is about 1.5 GB.  This script exposes it to
``zipfile.ZipFile`` through cached HTTP byte ranges, so the central directory
and the requested sequence are transferred without downloading the full ZIP.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import requests
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "subject_7/20231022_192832"
CAMERA = "043422252387"
HAND_SLOT = 1  # HO-Cap order is [right, left].
LABELS_URL = (
    "https://utdallas.box.com/shared/static/"
    "ayd4st2wo588z2yqbuxalptxnz2qxlj5.zip"
)
OUTPUT = (
    EXPERIMENT_ROOT
    / "data"
    / "subset"
    / SEQUENCE
    / "hand_joints_3d_left.npy"
)
EXTRINSICS = (
    EXPERIMENT_ROOT
    / "data"
    / "subset"
    / "calibration"
    / "extrinsics"
    / "extrinsics_20231014.yaml"
)


class HTTPRangeReader(io.RawIOBase):
    """Seekable, chunk-cached reader over an HTTP resource."""

    def __init__(self, url: str, chunk_size: int = 16 * 1024 * 1024):
        super().__init__()
        self._session = requests.Session()
        self._chunk_size = int(chunk_size)
        probe = self._session.get(
            url,
            headers={"Range": "bytes=0-0"},
            allow_redirects=True,
            timeout=60,
        )
        probe.raise_for_status()
        content_range = probe.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes 0-0/(\d+)", content_range)
        if probe.status_code != 206 or match is None:
            raise RuntimeError(
                "HO-Cap labels server did not honor byte ranges: "
                f"status={probe.status_code}, Content-Range={content_range!r}"
            )
        self._url = probe.url
        self._size = int(match.group(1))
        self._position = 0
        self._cache: dict[int, bytes] = {0: probe.content}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self._position = min(position, self._size)
        return self._position

    def _read_chunk(self, chunk_start: int) -> bytes:
        cached = self._cache.get(chunk_start)
        if cached is not None and not (chunk_start == 0 and len(cached) == 1):
            return cached
        chunk_end = min(chunk_start + self._chunk_size, self._size) - 1
        response = self._session.get(
            self._url,
            headers={"Range": f"bytes={chunk_start}-{chunk_end}"},
            timeout=180,
        )
        response.raise_for_status()
        expected = chunk_end - chunk_start + 1
        if response.status_code != 206 or len(response.content) != expected:
            raise RuntimeError(
                "Unexpected HO-Cap byte-range response: "
                f"status={response.status_code}, expected={expected}, "
                f"received={len(response.content)}"
            )
        self._cache[chunk_start] = response.content
        return response.content

    def read(self, size: int = -1) -> bytes:
        if self._position >= self._size:
            return b""
        if size is None or size < 0:
            size = self._size - self._position
        remaining = min(size, self._size - self._position)
        pieces = []
        while remaining:
            chunk_start = (self._position // self._chunk_size) * self._chunk_size
            chunk = self._read_chunk(chunk_start)
            offset = self._position - chunk_start
            take = min(remaining, len(chunk) - offset)
            pieces.append(chunk[offset : offset + take])
            self._position += take
            remaining -= take
        return b"".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=SEQUENCE)
    parser.add_argument("--camera", default=CAMERA)
    parser.add_argument("--hand-slot", type=int, default=HAND_SLOT)
    parser.add_argument("--extrinsics", type=Path, default=EXTRINSICS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    prefix = f"{args.sequence}/{args.camera}/label_"
    reader = HTTPRangeReader(LABELS_URL)
    with zipfile.ZipFile(reader) as archive:
        members = sorted(
            (
                name
                for name in archive.namelist()
                if name.startswith(prefix) and name.endswith(".npz")
            ),
            key=lambda name: int(Path(name).stem.split("_")[-1]),
        )
        if not members:
            raise RuntimeError(
                f"No labels found for {args.sequence}, camera {args.camera}"
            )

        frames = []
        frame_ids = []
        for member in members:
            frame_ids.append(int(Path(member).stem.split("_")[-1]))
            with np.load(io.BytesIO(archive.read(member))) as label:
                joints = np.asarray(label["hand_joints_3d"], dtype=np.float32)
            if joints.shape != (2, 21, 3):
                raise RuntimeError(
                    f"Unexpected hand_joints_3d shape in {member}: {joints.shape}"
                )
            frames.append(joints[args.hand_slot])

    expected_ids = list(range(len(frame_ids)))
    if frame_ids != expected_ids:
        raise RuntimeError(
            f"Non-contiguous label frames: first={frame_ids[:5]}, "
            f"last={frame_ids[-5:]}"
        )

    trajectory_camera = np.stack(frames)
    calibration = yaml.safe_load(
        args.extrinsics.read_text(encoding="utf-8")
    )["extrinsics"]

    def matrix(values: list[float]) -> np.ndarray:
        return np.asarray(
            [values[0:4], values[4:8], values[8:12], [0, 0, 0, 1]],
            dtype=np.float64,
        )

    camera_to_world = (
        np.linalg.inv(matrix(calibration["tag_1"]))
        @ matrix(calibration[args.camera])
    )
    trajectory = (
        trajectory_camera @ camera_to_world[:3, :3].T
        + camera_to_world[:3, 3]
    ).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, trajectory)
    metadata = {
        "source": "HO-Cap official labels.zip",
        "archive_url": LABELS_URL,
        "sequence": args.sequence,
        "camera": args.camera,
        "hand_slot": args.hand_slot,
        "source_coordinate_frame": f"camera/{args.camera}",
        "output_coordinate_frame": "HO-Cap world",
        "camera_to_world": camera_to_world.tolist(),
        "joint_order": [
            "wrist",
            "thumb_mcp",
            "thumb_pip",
            "thumb_dip",
            "thumb_tip",
            "index_mcp",
            "index_pip",
            "index_dip",
            "index_tip",
            "middle_mcp",
            "middle_pip",
            "middle_dip",
            "middle_tip",
            "ring_mcp",
            "ring_pip",
            "ring_dip",
            "ring_tip",
            "pinky_mcp",
            "pinky_pip",
            "pinky_dip",
            "pinky_tip",
        ],
        "shape": list(trajectory.shape),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
