#!/usr/bin/env python3
"""Download only the small HO-Cap archives needed for trajectory replay."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = EXPERIMENT_ROOT / "data" / "raw"
ARCHIVES = {
    "models.zip": (
        "https://utdallas.box.com/shared/static/"
        "con44iqej33weg9f3rpxof61eh3x2x21.zip"
    ),
    "poses.zip": (
        "https://utdallas.box.com/shared/static/"
        "2lofbp2yd005d8o213ns77mdrtxg8eep.zip"
    ),
    "calibration.zip": (
        "https://utdallas.box.com/shared/static/"
        "nlp4c6vtd0n8o0entxlh1vxdpcdeh0h8.zip"
    ),
}


def main() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    for filename, url in ARCHIVES.items():
        destination = RAW_ROOT / filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"skip {filename}: already present")
            continue
        partial = destination.with_suffix(destination.suffix + ".part")
        print(f"download {filename}")
        with urllib.request.urlopen(url) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        partial.replace(destination)


if __name__ == "__main__":
    main()
