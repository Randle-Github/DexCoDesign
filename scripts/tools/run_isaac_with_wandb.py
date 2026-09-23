# Copyright (c) 2026, The DexCoDesign Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Run an unchanged Isaac script, flushing W&B before Kit's native fast exit.

Usage: ./isaaclab.sh -p scripts/tools/run_isaac_with_wandb.py SCRIPT [ARGS...]
Only process shutdown is wrapped; script arguments and training are unchanged.
Import W&B only through the script so Isaac's bundled dependencies are ready.
"""

from __future__ import annotations

import functools
import runpy
import sys
from pathlib import Path


def install_wandb_close_hook(simulation_app_class) -> None:
    original_close = simulation_app_class.close

    @functools.wraps(original_close)
    def close_with_wandb(self, *args, **kwargs):
        wandb = sys.modules.get("wandb")
        if wandb is not None and getattr(wandb, "run", None) is not None:
            print("[WUJI_RUNTIME] Flushing W&B before Isaac Sim shutdown", flush=True)
            wandb.finish(exit_code=int(sys.exc_info()[0] is not None))
        return original_close(self, *args, **kwargs)

    simulation_app_class.close = close_with_wandb


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_isaac_with_wandb.py SCRIPT [ARGS...]")
    script = Path(sys.argv[1]).expanduser().resolve(strict=True)
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path[0] = str(script.parent)

    # Use the same SimulationApp class/import path as every AppLauncher script.
    from isaaclab.app.app_launcher import SimulationApp

    install_wandb_close_hook(SimulationApp)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
