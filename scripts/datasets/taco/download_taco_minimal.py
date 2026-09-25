#!/usr/bin/env python3
"""Download only the TACO modalities used by DexCoDesign (never videos)."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests


BASE = "https://huggingface.co/datasets/mzhobro/taco_dataset/resolve/main"
FILES = {
    "taco_info.csv": (
        1_663_778,
        "3d2f9ac34440aec5af32de44fa7d32b2a240eadf6f140061964beb38abbc165a",
    ),
    "Object_Poses.zip": (
        34_340_303,
        "39c0478395529b550fd218fdbcb9d0c866282c4dfc0d5bac1116461b85e91b64",
    ),
    "Hand_Poses_3D.zip": (
        167_832_020,
        "287593ce9b75ddda15d1544f878920e0ff97aae519cd0b6bd8ea7603def7ba35",
    ),
    "Object_Models.zip": (
        424_209_827,
        "7b5209c7e856246cf20b379a599d756d18e8bbbbf7820bf9fe48a334f9a06cfe",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, destination: Path, expected_bytes: int, expected_sha256: str) -> None:
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        if sha256(destination) == expected_sha256:
            print(f"verified {destination}")
            return

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    url = f"{BASE}/{name}?download=true"
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            partial.unlink(missing_ok=True)
            offset = 0
        mode = "ab" if offset else "wb"
        with partial.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    stream.write(chunk)
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(f"Size mismatch for {name}: {partial.stat().st_size} != {expected_bytes}")
    if sha256(partial) != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {name}")
    partial.replace(destination)
    print(f"downloaded and verified {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/taco_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    downloads = root / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for name, (size, checksum) in FILES.items():
        destination = root / name if name.endswith(".csv") else downloads / name
        download(name, destination, size, checksum)


if __name__ == "__main__":
    main()
