#!/usr/bin/env python3
"""Download only the ARCTIC motion ground truth needed by DexCoDesign.

The official archive cannot be downloaded anonymously.  Set
ARCTIC_USERNAME and ARCTIC_PASSWORD after registering at the ARCTIC site.
No RGB, video, features, processed splits, SMPL-X models, or checkpoints are
requested by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import requests


BASE = (
    "https://download.is.tue.mpg.de/download.php?domain=arctic&resume=1&"
    "sfile=arctic_release/c7216c3b205186106a1f8326ed7b948f838e4907e69b21c8b3c87bb69d87206e/"
    "v1_0/data/"
)
ARCHIVES = {
    "raw_seqs.zip": {
        "url": BASE + "raw_seqs.zip",
        "sha256": "3c74f8cdb5fb4f521d99132faf0471432bab5db97c7653493b01920d2ad48535",
    },
    "meta.zip": {
        "url": BASE + "meta.zip",
        "sha256": "2ec627bcb8f17be33defc985a79d1dd744ee44b1f1b5732ed163ecef217c0c6e",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, username: str, password: str) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    with requests.post(
        url,
        data={"username": username, "password": password},
        stream=True,
        verify=True,
        allow_redirects=True,
        timeout=(30, 300),
    ) as response:
        if response.status_code in (401, 403):
            raise RuntimeError("ARCTIC authentication failed; check ARCTIC_USERNAME/PASSWORD")
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            raise RuntimeError("ARCTIC returned HTML instead of an archive; login or license acceptance is missing")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=8 << 20):
                if chunk:
                    stream.write(chunk)
    partial.replace(destination)


def normalized_member(name: str) -> tuple[str, Path] | None:
    parts = PurePosixPath(name).parts
    for marker in ("raw_seqs", "meta"):
        if marker not in parts:
            continue
        index = parts.index(marker)
        relative = Path(*parts[index + 1 :])
        return marker, relative
    return None


def wanted(marker: str, relative: Path) -> bool:
    if not relative.parts or relative.name.startswith("."):
        return False
    if marker == "raw_seqs":
        # MANO and articulated-object state only.  SMPL-X, cameras and RGB are
        # unnecessary for the robot retargeting benchmark.
        return relative.name.endswith((".mano.npy", ".object.npy"))
    if marker == "meta":
        if relative.name in {"misc.json", "object_meta.json"}:
            return True
        if "object_vtemplates" not in relative.parts:
            return False
        return relative.name in {
            "top.obj",
            "bottom.obj",
            "mesh.obj",
            "parts.json",
            "object_params.json",
            "top_keypoints_300.json",
            "bottom_keypoints_300.json",
        }
    return False


def extract_selected(archive: Path, output_root: Path) -> int:
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            normalized = normalized_member(member.filename)
            if normalized is None:
                continue
            marker, relative = normalized
            if member.is_dir() or not wanted(marker, relative):
                continue
            destination = output_root / marker / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 << 20)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/arctic_v1"))
    parser.add_argument(
        "--include-official-meta",
        action="store_true",
        help="Also fetch official object templates. The repository already carries a compatible 11-object asset subset.",
    )
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    username = os.environ.get("ARCTIC_USERNAME")
    password = os.environ.get("ARCTIC_PASSWORD")
    if not args.verify_only and (not username or not password):
        raise SystemExit(
            "Missing ARCTIC credentials. Export ARCTIC_USERNAME and ARCTIC_PASSWORD "
            "after accepting the license at https://arctic.is.tue.mpg.de/."
        )

    selected = ["raw_seqs.zip"]
    if args.include_official_meta:
        selected.append("meta.zip")
    archive_root = args.root / "downloads"
    raw_root = args.root / "raw"
    for filename in selected:
        spec = ARCHIVES[filename]
        archive = archive_root / filename
        if not archive.exists() and not args.verify_only:
            print(f"Downloading {filename} ...", flush=True)
            download(spec["url"], archive, username or "", password or "")
        if not archive.exists():
            raise SystemExit(f"Missing archive: {archive}")
        actual = sha256(archive)
        if actual != spec["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {filename}: {actual}")
        extracted = extract_selected(archive, raw_root)
        print(f"Verified and extracted {filename}: {extracted} selected files")
        if not args.keep_archives:
            archive.unlink()

    print(f"ARCTIC_MINIMAL_READY root={args.root.resolve()}")


if __name__ == "__main__":
    main()
