# MiDas manufacturing-constrained morphology grammar

MiDas uses a dedicated source-local grammar. It does not replace or modify the
generic hand grammar or the WUJI morphology-search vector.

## Parameterization

The optimizer input is a 15-dimensional vector in `[-1, 1]`. The all-zero
vector resolves exactly to the source MiDas geometry. Negative and positive
halves map independently to the lower/source and source/upper physical ranges,
so asymmetric engineering bounds remain meaningful.

All resolved dimensions are stored in millimetres:

1. palm width and height;
2. common index/middle/ring base spacing and thumb-base distance;
3. common normal-finger DP length and PP length;
4. common normal-finger DP, MP, and PP widths;
5. thumb DP, MP, and PP lengths;
6. thumb DP, MP, and PP widths.

Normal-finger MP length is derived from DP length using the fixed source
MP/DP ratio, so the resolved graph contains 16 physical dimensions while the
search vector has 15 degrees of freedom.

## Constructive constraints

- Index, middle, and ring dimensions share one parameter set.
- Finger-base spacing is greater than both MP and PP widths.
- DP width is smaller than both MP and PP widths.
- The v1 polymer profile keeps MP/PP widths at or above 27.5 mm.
- Finger and thumb thicknesses remain source-exact.
- Joint count, joint types, axes, limits, motor ownership, and transmission
  semantics remain source-exact.
- Motor housings and proximal/distal connector caps remain rigid. Only the
  middle span of each phalanx is deformed, and distal joint centres and their
  complete downstream graphs move analytically with the new length.
- The PIP/DIP linkage branch is recalibrated from the resolved PP length while
  the required MP/DP ratio stays fixed.
- Palm width and height are continuous dimensions. No palm prototype bank or
  nearest-template quantization is used.
- The wrist/transmission region is fixed. Short palms construct deterministic
  motor-interface collars, rather than relying on a disconnected mesh repair.
- Visual and collision geometry use the same compiled mesh deformation.

## Generation

```bash
.venv-morphology/bin/python \
  scripts/dexcodesign/generate_midas_morphologies.py \
  --count 50 --workers 4
```

The default output is
`artifacts/hand_morphology/midas_constraints_50/`. Meshes are stored as binary
PLY without face reduction; this avoids duplicating many gigabytes of textual
OBJ data while preserving the same vertices and faces.

