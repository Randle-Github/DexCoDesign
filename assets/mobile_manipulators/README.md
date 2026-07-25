# Mobile-manipulator source assets

This directory contains vendor-source robot descriptions and generated
end-effector-free URDFs for the first DexCoDesign mobile-manipulator library.
Source files are preserved under each robot's `source/` directory. Generated
files under `eef_free/` are reproducible with:

```bash
.venv-morphology/bin/python scripts/assets/prepare_mobile_manipulators.py
```

The generated models keep each vendor's mobile base, torso, head, arms,
kinematics, joint limits, and inertials. Gripper/hand geometry and joints are
removed. Galaxea's two separately mounted wrist RealSense sensors are also
removed because they become visually detached after the grippers are removed.
Each arm terminates in an empty fixed link named `left_eef_mount` or
`right_eef_mount`, placed at the vendor tool transform.

## Version provenance

Versions were checked against the official upstreams on 2026-07-25.

| Robot | Official source | Locked version |
| --- | --- | --- |
| Dexmate Vega | [dexmate-ai/dexmate-urdf](https://github.com/dexmate-ai/dexmate-urdf), distributed through [PyPI](https://pypi.org/project/dexmate-urdf/) | `0.8.4`, uploaded 2026-07-02, wheel SHA256 `0c04e8134e625fa13401268249ab3fead3ce7c54d267eb07950f2eb7deafe67c` |
| Rainbow Robotics RB-Y1 M | [RainbowRobotics/rby1-sdk](https://github.com/RainbowRobotics/rby1-sdk) | RBY1-M `model_v1.3.urdf` from `main` commit `38df3267e617d22644f6686e8a7e3c4eac3ce2ee`; the model's latest upstream fix is commit `9b7e2fa28b8ffcefbf2f17cfbe09927b402c807f` (2025-10-20) |
| Galaxea R1 | [userguide-galaxea/URDF](https://github.com/userguide-galaxea/URDF/tree/galaxea/main/R1) | `galaxea/main` commit `2e5d31e1784481a34d178006c0d0e18e0a84a82a`, model `r1_v2_1_0` |

RB-Y1 upstream is Apache-2.0 and its license is copied beside the source
model. The canonical local `source/model.urdf` is an exact copy of upstream
`models/rby1m/urdf/model_v1.3.urdf`; upstream's unversioned model files are
deliberately not used. Dexmate's package
metadata does not declare an SPDX license. The
Galaxea URDF repository does not publish a root license in the locked commit;
redistribution terms should therefore be confirmed before public release.

## Eligible robots without a public description

The following robots satisfy the wheeled mobile-manipulator requirement, but
were not added because no official whole-robot URDF or MJCF was publicly
downloadable when rechecked on 2026-07-24:

- **X Square Robot QUANTA X2 (自变量量子二号)**: the official product page
  describes a wheeled base, dual 7-DoF arms, and interchangeable grippers or
  dexterous hands. The official site exposes specifications and media, but no
  whole-robot description package or official GitHub model repository.
- **RobotEra Q5 (星动纪元 Q5)**: the official product/download pages identify
  it as a 44-DoF wheeled humanoid. The Q5 download entry exposes only the
  product-sheet PDF; its manual/model field is empty. The official
  [`roboterax/models`](https://github.com/roboterax/models) repository was
  checked at its sole branch HEAD, commit
  `e8660e664e39e5b80ebb1f252e1b77b0e6929cff`, and contains only the bipedal
  STAR1 description.

These models should be added only after the vendors publish an authoritative
description package or provide one directly; a third-party reconstruction
would not meet this library's latest-official-source requirement.

## Proposed additions (not downloaded)

- **Hello Robot Stretch 3**: widely used modern research baseline with
  official URDF/MuJoCo support and a mature manipulation ecosystem.
- **PAL Robotics TIAGo++**: mature dual-arm wheeled mobile manipulator with
  strong ROS, MoveIt, and simulator support.

These are proposals only and are not part of the current asset download.
