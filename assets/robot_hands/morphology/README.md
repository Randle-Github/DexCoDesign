# Morphology source scaffold

`canonical_scaffold.json` contains only semantic roles, rigid-link grouping,
source joint names, and the shared upright coordinate convention for the 14
current right-hand sources. It contains no generated mesh and no `temp/`
dependency.

The complete geometry and kinematics are read from
`assets/robot_hands/direct_motor/<hand>/right/hand.urdf`.

When a new hand is added, first generate its bilateral direct-motor asset, then
add its semantic rigid-link grouping to this scaffold. The production graph
builder refuses to silently omit a registered source.
