# Asset validation

The entry points in `registry.json` are the only supported project-facing model paths. Validation is intentionally layered so broken topology, missing meshes, malformed simulator files, and upstream provenance issues are caught separately.

## Completed checks

- Registry completeness: 14 hands, exactly 28 left/right entries.
- URDF topology and resource resolution: every link/joint tree is connected, joint limits are valid, all mesh paths resolve, and no preferred entry retains a ROS `package://` URI.
- URDF runtime parsing: every preferred URDF parses as a robot tree, and all 252 unique referenced mesh files load with Trimesh.
- MJCF runtime compilation: both sides of MANO, Schunk SVH, Shadow Hand E-Series, and Inspire RH56DFX compile with MuJoCo 3.10.
- USD stage opening: both sides of Sharpa Wave 01 and Tesollo DG-5F open with OpenUSD and resolve their composed layers.
- Generated handedness: MIDAS and RUKA-v2 left hands include reflected mesh vertices, corrected triangle winding, transformed poses and joint axes, and reflected inertia tensors.

WUJI's official native USD files are retained for provenance, but are not registered as preferred entries because the upstream stages contain unresolved fingertip visual prim references. The official left/right URDF files from the same release are used instead and pass the URDF and mesh checks.

SPIDER also contains Ability Hand, Allegro, Inspire, and XHAND assets. They are intentionally not copied because this registry already contains the requested current versions of those families. Ability Hand now comes from PSYONIC's official MIT-licensed repository at commit `34c9a9324d3739d976e6de441e56ccefafd0000b`. The retained SPIDER snapshots are pinned to commit `71238456bf97a7eeb3d0471aa31974e2d404d4ae` and retain SPIDER's CC BY-NC 4.0 license; they must not be used commercially.

## Run locally

Static/resource validation works without Isaac Sim:

```bash
python scripts/assets/prepare_robot_hands.py
python scripts/assets/validate_robot_hands.py
```

The final Isaac Lab importer smoke test must run on a supported NVIDIA Isaac Sim installation:

```bash
./isaaclab.sh -p scripts/assets/smoke_test_robot_hands.py --headless
```

This test imports each registered URDF/MJCF or opens its USD stage, then checks for rigid bodies, joints, and an articulation. Passing the parser and resource checks substantially reduces import risk, but the Isaac Lab smoke test is the definitive runtime gate and cannot be honestly guaranteed without running it in the target NVIDIA environment.
