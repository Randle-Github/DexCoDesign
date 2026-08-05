# GPU morphology search

The production WUJI search evaluates every sampled morphology with the same
Isaac Lab/PhysX rollout semantics as residual RL. There is no kinematic/contact
proxy and no top-k prefilter.

For every generation:

1. `update_gpu_wuji_cem.py` samples the complete reshape-vector population.
2. `gpu_wuji_retarget.py --save-trajectories --skip-proxy` retargets the
   446-frame Human/MANO command trajectory to every candidate on the GPU.
3. `prepare_wuji_morphology_physx_batch.py` generates every graph, mesh, URDF,
   and per-candidate reference. Mesh compilation and export are CPU-parallel.
4. `evaluate_wuji_morphology_physx_batch.py` creates one distinct hand asset per
   PhysX environment and rolls out all candidates concurrently. It calls the
   normal residual-RL environment reward/contact/termination implementation:
   C-error pose reward, binary thumb-versus-other-finger pinch contact, dynamic
   object physics, and 5 cm / 1.5 rad early termination.
5. CEM is updated only if the result proves that the entire sampled population
   completed physical evaluation. Proxy or partial summaries are rejected.

`launch_gpu_wuji_cem.py` submits one persistent Slurm allocation for all
generations. The allocation therefore does not return to the queue between
sampling, retargeting, asset generation, PhysX rollout, and distribution
update. Isaac may restart inside that allocation when the morphology set
changes, but the node and GPU remain assigned to the same workflow.

The default population is 1,024. It is evaluated in 128-environment physical
batches inside the same allocated job so that unique USD assets fit in memory.
Every one of the 1,024 candidates still receives a real rollout; batching is
not filtering. Search assets omit visual geometry and retain collision on every
rigid hand part. The selected morphology is regenerated with full visual meshes
for its final video.

The 64-candidate validation completed all 64 real rollouts concurrently; its
PhysX rollout itself took about five seconds. Asset generation and USD cooking,
not simulation, are the main remaining scaling costs.
