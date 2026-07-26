# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Configuration for different assets.
##

try:
    from .gelsight import *
except ModuleNotFoundError as exc:
    if not exc.name or not exc.name.startswith("isaaclab_contrib"):
        raise
from .velodyne import *
