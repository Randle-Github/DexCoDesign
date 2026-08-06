# Morphology Rules for Mechanical Review

This document summarizes the current design grammar in **DexCoDesign**. The goal is to preserve the source robot's mechanisms and interfaces while exposing a small, fast morphology vector for search. “Buildable” below means mechanically plausible under these grammar checks; it does not replace stress, tolerance, cable-routing, or actuator-load analysis.

## 1. Dexterous-hand transformation rules

- **Source-bound construction.** A generated hand uses the complete rigid-link, motor, and transmission modules of one source hand. Meshes are never mixed across source hands.
- **Kinematic semantics stay fixed.** Joint axes, limits, active/mimic relationships, wrist mount, palm thickness, and protected under-palm hardware do not change. Mimic joints remain affine followers and are not independent design or control variables.
- **Finger-link dimensions.** A whitelisted phalanx may change its longitudinal length and radial size in its own semantic link frame. Random-generation ranges are currently `0.87–1.17` for length and `0.88–1.14` for radius; wider explicit search bounds are available when requested. The current WUJI experiment ties all segments of one finger to one shared length/radius pair.
- **Palm geometry.** Palm in-plane width/depth and a bounded in-plane yaw may change. Palm thickness is fixed. `palm_expansion` is a continuous scalar: `0` preserves the source finger-root layout and `1` reaches the selected star/House-style layout. In the fast PhysX search it is represented by an ordered bank of 32 collision-mesh prototypes, while its optimization variable remains scalar and ordered.
- **Finger-root motion.** Palm interfaces and the complete attached finger roots move together. Small root displacement is allowed only along the palm edge/tangent direction; roots must retain clearance from motor footprints and protected hardware.
- **Topology.** Finger order is preserved, a source finger may be used at most once, and the hand has at most five fingers. A four-finger source is not given a synthetic fifth finger.
- **Mesh/contact rule.** Visual meshes retain the source vertices/faces and are deformed in the appropriate local frame. Collision geometry is generated separately but must cover every palm and finger rigid body. Hand–object and object–support contact are enabled; hand–table/ground contact is disabled for the current retargeting task.
- **Hard validity checks.** The output graph must be connected and acyclic, preserve the active/mimic DoF partition, maintain bilateral/interface consistency, contain no duplicate or floating rigid parts, and pass URDF export, collision-coverage, retargeting, and Isaac Lab smoke tests.

**Current WUJI search vector.** Palm expansion (`0–0.70`), palm in-plane X/Z scale, palm yaw, and independent length/radius scales for all five fingers (14 normalized continuous parameters; palm expansion selects an ordered prototype at simulation time).

## 2. Mobile-manipulator transformation rules

- **Base is immutable.** The root, mobile base, wheels, steering, torso-base sensors, and global base pose cannot change. The end effector is outside the grammar and is left empty.
- **Only two geometric productions are allowed.** (1) vertical link length and (2) shoulder width. Everything else is denied by default.
- **Vertical length.** A whitelisted body or arm segment may scale only along its original-pose world-Z direction (`0.5×` or `1.5×` in the current discrete grammar). The proximal connector cap is fixed, the distal cap and its complete downstream graph move together, and only the middle span is deformed. Terminal wrist/tool segments may lengthen but are not shortened if shortening would invert the source CAD.
- **Shoulder width.** Only the source shoulder carrier may scale laterally along original-pose world Y (`0.75×` or `1.25×`). Both shoulder attachments receive the same transform. Current carriers are Dexmate `torso_l3`, RB-Y1 `link_torso_5`, and Galaxea `torso_link4`.
- **Strict bilateral symmetry.** Every arm production is atomic: corresponding left/right links receive mirrored parameter changes, and both complete paths remain connected.
- **No joint reorientation.** Joint-origin RPY, joint axes, joint limits, and displayed joint values remain unchanged. No corrective IK is used to hide a structurally invalid design.
- **Physical consistency and speed.** Mass, center of mass, and inertia are updated with geometry. Source/link/factor mesh variants are precompiled; morphology search only selects cached variants and updates graph/URDF parameters.

## 3. Related co-design grammars

| Method | Mesh treatment | What changes and how | Buildability / validity rule |
|---|---|---|---|
| [Transformer Transformer](https://arxiv.org/abs/2607.25798) | **Simplified.** RoboToken represents links using primitive type, size, and inertia rather than full production CAD. | Curated spaces vary mounting pose, DoF count, joint orientation, link length, leg topology, sliding/fixed arm mounts, torso/spine type, and segment lengths. | Generated tokens must form a valid robot graph and stay inside a curated design space. A real ALOHA-family design is demonstrated, but the representation is not a general fabrication guarantee. |
| [House of Dextra](https://arxiv.org/html/2512.03743) — Wang et al. | **Parametric/modular.** A convex extruded palm and modular finger/fingertip collision parts are generated; CoACD is used for collision decomposition. | A fixed palm-plus-five-slot graph varies finger count, per-finger servo/joint grammar, segment scales, fingertip type, palm radius, and symmetric/asymmetric/anthropomorphic root layouts. | Minimum angular separation, motor-aware grip points, CAD joint limits, modular Dynamixel hardware, and direct grammar-to-printed-part mapping make the search strongly fabrication-aware; hardware is demonstrated. |
| [Learning to Design Soft Hands using Reward Models](https://arxiv.org/html/2510.17086) — Wang et al. | **Structured FEM mesh.** It optimizes a printable tendon-driven soft-finger template rather than preserving arbitrary source CAD. | Flexure length, block length/thickness, tendon waypoint height/routing, finger mount position, and finger orientation are varied within bounded ranges. | Geometry bounds and non-penetration checks reject invalid hands; the monolithic soft-hand template is 3D printable and final designs are physically tested, although manufacturability is template-specific. |
| [DiffHand](https://arxiv.org/abs/2107.07501) | **Detailed mesh can be retained.** Cage/deformation-based parameterization changes complex articulated geometry without reducing every link to a primitive. | Joint/kinematic parameters and rigid-body shape are optimized through differentiable contact simulation; cage handles deform the geometry continuously. | Bounded deformation of an assembly template helps preserve integrity and permits 3D printing, but there is no universal discrete manufacturing grammar or automatic stress/tolerance proof. |
| [RoboGrammar](https://people.csail.mit.edu/jiex/papers/robogrammar/index.html) | **Simplified components.** Robots are assembled from primitive links, joints, and wheels; detailed hand/end-effector CAD is not modeled. | Graph rewrite rules add paired limbs, impose symmetry, extend/terminate branches, choose joint type, and select discrete link lengths/wheels. | Its grammar is explicitly written to produce connected, symmetric, fabricable component assemblies. Feasibility is strong within that component library, not outside it. |

## 4. Questions for mechanical review

1. Are the proposed length/radius and palm-expansion bounds compatible with actuator packaging, tendon/cable routing, minimum wall thickness, and assembly access?
2. Is moving a finger root together with its complete motor/link module sufficient to preserve each real mounting interface?
3. Are the mobile-manipulator middle-span and shoulder-carrier deformations structurally plausible without redesigning bearings, harnesses, or covers?
4. Which designs require load, stiffness, fatigue, or collision-clearance analysis beyond the current geometric and simulation checks?
