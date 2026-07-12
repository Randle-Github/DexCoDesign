# Isaac Lab upstream provenance

This repository contains a focused source snapshot of:

- Repository: <https://github.com/isaac-sim/IsaacLab>
- Tag: `v2.3.2`
- Commit: `37ddf626871758333d6ed89cf64ad702aef127d0`
- Retrieved: 2026-07-12

The Git remote named `upstream` points to the official repository. The original DexCoDesign repository remains `origin`.

## Update policy

Do not merge the complete upstream tree blindly: this repository intentionally omits unrelated components. To review a future stable release:

```bash
git fetch upstream --tags
git diff 37ddf626871758333d6ed89cf64ad702aef127d0 <new-tag> -- \
  apps source scripts isaaclab.sh pyproject.toml environment.yml
```

Port relevant changes selectively and record the new tag and commit in this file and in `README.md`.

## Licensing

The upstream BSD-3-Clause license is retained in `LICENSE`. Isaac Lab Mimic's Apache-2.0 license is retained in `LICENSE-mimic`. Third-party notices required by the retained code are under `docs/licenses`.
