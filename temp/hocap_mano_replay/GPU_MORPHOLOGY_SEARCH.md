# GPU morphology search

The WUJI search is split into a high-throughput proposal stage and a small
exact stage. This avoids instantiating thousands of high-face-count collision
models while keeping the optimization objective equal to the physical
rollout reward.

1. `update_gpu_wuji_cem.py` samples 4,096 or more reshape vectors.
2. `gpu_wuji_retarget.py` reuses the exact source-hand 446-frame trajectory
   and runs batched Torch DLS on one GPU. It also computes a broad distal-link
   contact proxy for proposal ranking.
3. `evaluate_gpu_wuji_topk.py` compiles the top 32/64 graphs in one shared
   batch, exports them in parallel, uses the GPU trajectories as per-frame
   exact-IK warm starts, and recomputes the exact MuJoCo C-error plus binary
   pinch-contact reward.
4. The exact rollout summary, never the proxy score, updates CEM for the next
   generation.

Recommended defaults are `population=4096`, `top_k=32`, four GPU DLS
iterations, four exact correction iterations, and 16/32 CPU workers. Raise
the population to 16,384 without increasing GPU memory by keeping
`candidate_chunk=128`.

The implementation follows Isaac Lab's damped-least-squares differential IK
algorithm, but uses a custom Torch FK/Jacobian kernel because Isaac Lab's
standard controller batches environments of one robot model and does not
natively batch thousands of different morphology parameters.

Do not copy a full mesh-based MJX model per proposal. A 1,024-model WUJI test
requested about 131 GiB of device memory. MJX or MuJoCo is appropriate for a
small exact top-k; shared-topology mesh-free FK/IK is appropriate for the
thousands-wide proposal stage.
