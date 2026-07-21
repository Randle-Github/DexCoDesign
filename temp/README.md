# All-right-hands MuJoCo gallery (macOS)

This viewer loads every `right` entry from `assets/robot_hands/registry.json`, converts their visual geometry into one combined MJCF scene, and opens them together in MuJoCo's native viewer. MANO is the first model in the top-left.

```bash
temp/.venv/bin/mjpython temp/visualize_all_right_hands_mujoco.py
```

Controls:

- left mouse drag: rotate;
- right mouse drag: pan;
- mouse wheel: zoom;
- double-click a model: select it in MuJoCo's inspector;
- `Esc`: close.

Build and compile the combined scene without opening a window:

```bash
temp/.venv/bin/mjpython temp/visualize_all_right_hands_mujoco.py --check
```

`visualize_right_hands.py` contains the URDF/MJCF/USD geometry loaders used by the gallery. It is not the main visualizer.
