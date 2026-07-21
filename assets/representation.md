# Grammar-Constrained Hand Morphology Co-Design

本文是 DexCoDesign 当前方法的权威设计文档。旧的自由 mesh regression、任意
motor/link 组合、跨 donor link 拼接和低维 full-hand decoder 不再属于正式方案。

项目研究的问题是：

> 给定人类示范、重定向方法和 residual PPO，哪种可实现的手部 morphology 能最有效地把人类动作变成机器人行为，并在固定训练预算内完成物体轨迹跟踪？

对 morphology `m`：

```text
human trajectory
  -> morphology-specific retargeting
  -> q_ref^m(t)
  -> morphology-specific residual PPO
  -> q_ref^m(t) + delta_q(t)
  -> object trajectory tracking
```

主要评价对象是物体参考轨迹，而不是手关节轨迹本身。

---

## 1. 不可违反的设计决定

1. 网络不直接输出 mesh vertex、URDF XML 或任意 adjacency matrix。
2. HandIR 是可编译的 source of truth；learned embedding 不是。
3. morphology node 对应相邻运动 joint/必要语义边界之间的一个最大 rigid functional link。
4. 同一个 rigid link 引用的全部 CAD visual 先变换到 link-local frame，再合并为一个 compound visual part。
5. movable joint 两侧的 link 永不合并。
6. visual mesh 与 collision geometry 分离。
7. graph 先于 mesh 存在；compiler 必须从 root/base 沿 graph 向 fingertip 逐 node 装配。
8. 每个 joint/motor instance 都有显式 `allowed_link_candidate_ids`，不能从全局 mesh 库任意采样。
9. link replacement 只替换当前 rigid link；绝不能平移、删除或整体替换它的 child subtree。
10. 离散结构合法性由 Design Grammar 保证，不依赖 reward 学会。
11. 第一版固定 topology，只搜索有界连续参数；稳定后才开放 topology edit。
12. 每种 morphology 使用独立 residual PPO checkpoint；parent-to-child 只做初始化迁移。
13. 全局最多 5 根 active finger；任何 grammar action 都不能创建第 6 根手指。

---

## 2. 总体表示

完整手表示为：

\[
H = \mathcal{C}(G, L, \Theta, \Phi)
\]

- `G`：typed kinematic graph；
- `L`：每个 motor instance 对应的合法 rigid-link candidate choices；
- `Theta`：几何、joint frame、finger attachment 等连续参数；
- `Phi`：物理、接触和允许搜索的硬件参数；
- `C`：确定性的 Hand Compiler。

```text
task/reference features + current HandIR + policy statistics
                           |
                           v
                 grammar-constrained search
                           |
                           v
                        HandIR
                           |
                  deterministic compiler
                           |
                           v
       URDF/USD/MJCF + visual + collision + dynamics
```

一个 HandIR 在 module database 和 compiler version 固定时，必须稳定生成同一个资产。

---

## 3. 正确的 rigid-link mesh integration

这里的 `integration` 不是传统 mesh simplification，也不是 decimation。

首先把由 fixed joint/helper frame 连接、且中间没有运动 DOF 的 source link 折叠成 maximal
rigid cluster。只有确实需要独立保留的安装接口、传感器或语义模块边界可以阻止折叠。然后对每个
rigid functional link：

1. 找到 cluster 内全部 source link 引用的 visual mesh；
2. 应用 fixed-link transform 以及各 visual 的 origin、scale 和 rotation；
3. 把顶点转换到该 functional link 的 canonical local frame；
4. 保留 material/submesh metadata；
5. 合并为一个 compound visual object；
6. 记录统一 bounds、connector frames、contact regions 和 source provenance。

```text
joint_i
   |
   |-- one rigid link
   |     |- shell_A.stl
   |     |- shell_B.stl
   |     |- bracket.obj
   |     `- cover.dae
   |
joint_(i+1)

=> one morphology LinkNode
=> one integrated compound visual part
```

螺丝、壳体、支架和 fixed helper link 不自动变成 morphology node。只有当它们被运动 joint
分隔，或者被明确标记为必须保留的机械、传感器或接口边界，才会成为单独 node。

第一版不对 integrated visual 做破坏性简化。collision 另行生成：普通 link 可使用 convex
parts、capsule 或 rounded box；fingertip 和 palm contact patch 保留更高精度。

---

## 4. Design Grammar：graph-first motor–link assembly

### 4.1 唯一正确的装配顺序

```text
先生成 typed graph
  -> 从 root/base 开始拓扑遍历
  -> 到达一个 joint/motor instance
  -> 查询这个 motor 的 allowed_link_candidate_ids
  -> 选择并安装一个 rigid link
  -> 沿 graph 继续安装 child joint
```

graph 决定 parent、child、joint origin、axis、limit 和零位；mesh 不能反过来改变 graph。替换一个
link 只允许改变当前 node 的 visual/collision、局部尺寸和 `template_to_link_pose`，不能把后续 joint
或整条 finger subtree 一起移动。

### 4.2 Motor instance 与有限 candidate 集

```yaml
MotorInstance:
  motor_instance_id: string
  motor_family_id: string
  parent_node_id: integer
  child_node_id: integer
  joint_template_id: string
  allowed_link_candidate_ids: [string]

LinkCandidate:
  candidate_id: string
  source_hand_id: string
  source_rigid_link: string
  integrated_visual_mesh: path
  proximal_connector: SE3
  distal_connectors: [SE3]
  deformation_spec_id: string
```

硬约束为：

\[
\operatorname{valid}(m,l)=1 \iff l\in\mathcal L(m)
\]

`L(m)` 必须显式存储。candidate 可以包括同一 motor family 在 source data 中真实连接过的多个 link，
以及这些 link 的 connector-preserving 变体；不能从全局 mesh library 做 nearest-neighbor 猜测。

### 4.3 相同 motor type

多个 joint 可以复用同一个物理 motor type。若 BOM、安装接口、输出轴和允许载荷证明它们属于同一
motor family，则这些 source instance 的合法 link candidate 可以合并。公开 URDF 没有 BOM 时，
`grammar_v1` 暂时用以下代理条件建立验证用 family：

- joint type 相同；
- thumb/normal-finger 类别一致；
- chain stage 一致；
- proximal connector 尺度和配准误差在阈值内。

这只是 visual/graph prototype 的 compatibility proxy，不是物理电机等价证明。最终数据库必须由
真实 motor part number、connector drawing 和 torque-speed 数据替换。

### 4.4 Tendon、mimic 与 differential

在 morphology graph 与 mesh 装配阶段，它们全部按普通 joint edge 处理。tendon/mimic/equality
只作为单独的 actuation mapping metadata 保存：

```yaml
ActuationMap:
  actuator_id: string
  joint_ids: [integer]
  ratios: [float]
  offsets: [float]
```

它们不参与 link candidate 选择，不允许改变 parent/child，也不允许触发“完整 digit bundle 替换”。
控制器或物理 compiler 后续再读取 actuation mapping。

---

## 5. HandIR

### 5.1 Global

```yaml
HandGlobal:
  schema_version: string
  compiler_version: string
  module_database_version: string
  handedness: left | right
  root_frame: SE3
  global_scale: float
  wrist_interface_id: string
  material_family: string
```

### 5.2 Rigid link node

```yaml
LinkNode:
  node_id: integer
  semantic_role: enum
  finger_slot: integer | null
  motor_instance_id: string
  link_candidate_id: string
  template_to_link_pose: SE3

  geometry:
    length_scale: float
    width_scale: float
    thickness_scale: float
    taper: float
    bend: float
    twist: float
    deformation_coefficients: vector

  contact:
    contact_patch_id: string | null
    patch_scale: vector3
    patch_pose: SE3
    friction: float

  physical:
    density: float
    mass_override: float | null
```

`link_candidate_id` 必须属于对应 motor instance 的 `allowed_link_candidate_ids`。不能出现孤立的
mesh ID，也不能因为更换当前 candidate 而改变 child node pose。

### 5.3 Joint edge

```yaml
JointEdge:
  joint_id: integer
  parent_node: integer
  child_node: integer
  joint_template_id: string
  joint_type: fixed | revolute
  origin_translation: vector3
  origin_rotation_6d: vector6
  axis_type: categorical
  axis_residual: vector3
  lower_limit: float
  upper_limit: float
  zero_position: float
  active: bool
  motor_instance_id: string | null
  coupling_template_id: string | null
```

joint origin、axis、limit 和 zero position 必须显式存在。

### 5.4 Finger slot

```yaml
FingerSlot:
  slot_id: integer
  active: bool
  role: thumb | normal | auxiliary
  palm_node_id: integer
  attachment_translation: vector3
  attachment_rotation_6d: vector6
  root_motor_instance_id: string
```

thumb opposition 主要由 attachment pose、root joint template 和 root motor/link 决定，不能只用 finger
length 表示。

### 5.5 Fingertip

```yaml
FingertipSpec:
  candidate_id: string
  scale_xyz: vector3
  curvature: vector
  patch_size: vector2
  patch_pose: SE3
  material_id: string
  local_deformation: vector
```

---

## 6. Module database 与 connector-preserving deformation

每个 candidate 是一个 source-derived rigid-link template：

```yaml
LinkCandidate:
  candidate_id: string
  compatible_motor_family_ids: [string]
  source_rigid_link: string
  semantic_role: string
  integrated_visual_mesh: path
  collision_recipe_id: string
  proximal_connector: SE3
  distal_connectors: [SE3]
  canonical_dimensions: vector3
  deformation_spec_id: string
  contact_regions: []
  content_hash: string
```

变形域分为：

```text
proximal rigid connector
deformable body
distal rigid connector
protected motor/transmission envelope
contact-critical region
```

基本规则：

- connector vertices 保持刚性；
- motor envelope 不允许被 deformable body 穿透；
- body deformation 在 source-specific bounds 内；
- fingertip contact region 使用独立高精度参数；
- 变形后重新计算 collision、mass、COM 和 inertia；
- deformation 失败时 compiler 报错，不能 fallback 到其他来源 mesh。

---

## 7. Strict Design Grammar

第一版 grammar：

```text
CREATE_WRIST(interface)
INSTANTIATE_GRAPH(template)
SELECT_MOTOR_INSTANCE(node, motor_family)
SELECT_ALLOWED_LINK(node, candidate)
SET_JOINT_TEMPLATE(edge, template)
SET_ACTUATION_METADATA(mapping)
EDIT_BOUNDED_GEOMETRY(target, delta)
END_FINGER
END_HAND
```

grammar 必须保证：

- exactly one root；
- 每个 active non-root node 有一个 parent；
- 无 disconnected component；
- 第一版无 closed loop；
- fingertip 只在 chain 末端；
- candidate 属于对应 motor instance 的显式 allow-list；
- 替换 link 不改变 child subtree 的 topology 或 pose；
- connector interface 匹配；
- q=0 位于 limits 内；
- active actuator 只引用 active joint；
- deformation 参数在 candidate-specific bounds 内。
- active finger 数量不超过 5。

这些是 action mask、schema validation 和 compiler assertion，不是 soft reward。

---

## 8. Deterministic Hand Compiler

```text
1. Validate HandIR schema and versions.
2. Validate graph grammar.
3. Resolve every motor instance and its allowed link candidate.
4. Instantiate integrated rigid-link visual candidates.
5. Apply connector-preserving bounded deformation.
6. Align parent connector -> joint frame -> child connector.
7. Build high-fidelity visual geometry.
8. Build stable collision geometry from collision recipes.
9. Validate motor envelopes and joint sweep clearance.
10. Compute volume, mass, COM and inertia.
11. Instantiate actuators, transmissions and couplings.
12. Generate self-collision filters and semantic metadata.
13. Export URDF/USD/MJCF.
14. Reload the exported asset and run articulation/FK checks.
```

任何阶段失败都返回结构化 `CompileError`，不能静默补 mesh、改 parent 或删除 joint。

---

## 9. Constraints

### 9.1 Grammar constraints

生成时禁止：结构、接口、motor allow-list、joint ordering 和 coupling 错误。

### 9.2 Analytic hard constraints

- motor/transmission envelope containment；
- connector compatibility；
- joint sweep clearance；
- static self-collision；
- torque-speed and gear-ratio limits；
- tendon stroke/routing；
- mass、inertia、wrist payload；
- minimum wall thickness and manufacturing bounds。

不满足时 reject 或 project，不进入昂贵 PPO evaluation。

### 9.3 Simulation constraints

- dynamic self-collision；
- contact penetration；
- torque saturation；
- slip/contact-loss；
- object tracking；
- robustness under friction、mass and geometry perturbation。

---

## 10. Human demonstration、retargeting 与 residual PPO

每种 morphology 都单独 retarget：

\[
q_{\mathrm{ref}}^m =
\operatorname{Retarget}(\tau_{\mathrm{human}},\tau_{\mathrm{obj}}^{\mathrm{ref}},m)
\]

控制命令：

\[
q_t^{\mathrm{cmd}} = q_{\mathrm{ref},t}^m + \Delta q_t
\]

每个 morphology 保存独立 policy checkpoint。所有 policy 可以使用相同的 semantic graph
architecture specification，但不使用同一个 universal trained checkpoint 作为主评估器。

parent-to-child transfer：

- same topology：完整复制 checkpoint；
- added/removed joint：按 semantic role 迁移匹配 module；
- changed finger count：迁移 global encoder 和匹配 finger module；
- major topology edit：只迁移 task/reference encoder。

最终 Top-K 必须独立从随机初始化训练，以区分 morphology quality 与 transfer convenience。

---

## 11. Hybrid SAC morphology search

SAC state：

```text
morphology graph embedding
policy learning/failure statistics
task-category statistics
retargeting feasibility
constraint margins
remaining budget
```

一次 action 只编辑一个局部目标：

```text
discrete:
  operation + target + grammar-valid motor/link/template choice

continuous:
  8-16 candidate-specific bounded deformation/joint parameters
```

第一版 operation：

```text
MODIFY_PALM_GEOMETRY
MODIFY_FINGER_ATTACHMENT
MODIFY_LINK_BODY
MODIFY_TIP
MODIFY_JOINT_LIMIT
SELECT_ALLOWED_LINK
STOP
```

`SELECT_ALLOWED_LINK` 只能从目标 motor instance 的 allow-list 选择，并且只修改当前 rigid link。

第一版固定 topology。后续才加入 `ADD/REMOVE_FINGER`、`ADD/REMOVE_JOINT` 和 coupling edit。
即使开放 topology edit，`ADD_FINGER` 也只能在 0–4 个 active slot 时执行，生成后最多 5 指。

---

## 12. Morphology evaluation

外层 reward 同时衡量：

\[
R_M =
-\lambda_e E_{\mathrm{early}}
-\lambda_a \operatorname{AUC}(E)
-\lambda_f E_{\mathrm{final}}
-\lambda_r C_{\mathrm{residual}}
-\lambda_h C_{\mathrm{hardware}}
-\lambda_c C_{\mathrm{constraint}}
\]

其中 object position/orientation tracking 是主任务信号。搜索阶段使用分层抽样的 8–16 条轨迹；
Top 5–10 使用约 100 条完整 benchmark、多 seed 和独立 PPO 训练。

---

## 13. 推荐实现数据结构

Python 层建议保持四个不可混淆的对象：

```python
@dataclass(frozen=True)
class MotorInstanceSpec:
    motor_instance_id: str
    motor_family_id: str
    parent_node_id: int
    child_node_id: int
    allowed_link_candidate_ids: tuple[str, ...]

@dataclass(frozen=True)
class LinkCandidateSpec:
    candidate_id: str
    source_hand_id: str
    visual_path: Path
    proximal_connector: Pose
    distal_connectors: tuple[Pose, ...]
    deformation_spec_id: str
    collision_recipe_id: str

@dataclass
class LinkNodeIR:
    node_id: int
    role: LinkRole
    motor_instance_id: str
    candidate_id: str
    template_to_link_pose: Pose
    geometry: GeometryParams

@dataclass
class HandIR:
    global_spec: HandGlobal
    nodes: list[LinkNodeIR]
    joints: list[JointEdgeIR]
    finger_slots: list[FingerSlotIR]
```

关键 validation：

```python
def validate_motor_link(node, motors, candidates):
    motor = motors[node.motor_instance_id]
    candidate = candidates[node.candidate_id]
    if node.candidate_id not in motor.allowed_link_candidate_ids:
        raise GrammarError("candidate is outside this motor instance allow-list")
```

compiler 只接受已经通过该 validation 的 HandIR。

---

## 14. 实施顺序

### Phase 1：source canonicalization

- parse source articulations；
- merge CAD visuals by rigid link；
- extract source motor instances and observed link bindings；
- annotate roles、connectors、contact regions；
- write versioned motor/link database。

### Phase 2：compiler

- HandIR schema；
- grammar validator；
- source-binding validator；
- integrated mesh instantiation；
- connector-preserving deformation；
- collision/inertia generation；
- URDF/USD/MJCF reload tests。

### Phase 3：fixed-topology experiment

- real seed hands only；
- graph topology and motor instances fixed；
- geometry and attachment edits only；
- morphology-specific retargeting/PPO；
- verify local policy inheritance。

### Phase 4：allowed per-link replacement

- allow `SELECT_ALLOWED_LINK` at one graph node；
- never move or replace the child subtree；
- add analytic hardware checks。

### Phase 5：limited topology and Hybrid SAC

- masked finger/joint activation；
- semantic policy transfer；
- search/promotion/final evaluation split。

---

## 15. 验收条件

在开始 SAC 前必须通过：

- source asset -> HandIR -> compiled source-equivalent asset；
- rigid-link node 数等于 movable/retained-boundary-separated rigid cluster 数，而不是 CAD file 或 fixed helper link 数；
- 每个 visual vertex 已正确变换到 link-local frame；
- 每个 motor instance 的 candidate allow-list 可追溯；
- 不存在 allow-list 之外的 motor–mesh pairing；
- connector frames 对齐；
- q=0 与 joint sweep 无明显 mesh 穿透；
- collision 与 visual 独立且可加载；
- mass/inertia 有限且物理一致；
- MuJoCo 与 Isaac Lab reload 后 topology、DOF、axis、limits 一致；
- reconstructed seed hand 的 workspace、retargeting 和 contact behavior 在容差内。

---

## 16. Legacy prototype 边界

`temp/exp1/strict_v2` 是历史 graph/crossover/MuJoCo 可视化实验，只用于记录失败模式和验证工具链。
它的跨 donor digit crossover、candidate mesh repair 和 VAE regression 不是正式 Design Grammar。

正式重构不得继续依赖：

- 整条 donor digit/subtree 搬运；
- allow-list 之外的 source link crossover；
- unrelated candidate mesh repair；
- motor 与 link candidate 独立采样；
- geometry descriptor 直接代表物理 mesh；
- reconstruction MAE 代表 contact fidelity；
- 由 visual bounds 猜 joint/motor interface。

下一阶段代码应从 versioned motor/link database、HandIR schema 和 deterministic compiler 开始。

---

## 17. 当前 `grammar_v1` 结构/视觉验证

`temp/exp1/grammar_v1` 当前采用 graph-first compiler：先实例化完整 source graph，再从 base 向外
逐 node 选择合法 rigid-link candidate。tendon/mimic/equality 不参与装配，只保留为以后控制器使用的
actuation metadata。

### 17.1 当前实现

- 保留 source graph 的 palm、base、掌内活动 joint 和 digit topology；
- 第一版不增加第 5 指，始终执行最多 5 指；
- 每个 motor instance 具有显式 `allowed_candidate_ids`；
- mixed design 只替换当前 rigid link，不搬运 donor digit bundle；
- `template_to_link_pose` 只配准当前 mesh，child joint/subtree 不随之平移；
- 若 source exporter 把 palm shell 挂在 digit-root body，该 carrier mesh 被保护；
- link length/radius 可做有界变化，graph attachment 从 base 向外重新计算。

### 17.2 当前数据和自动审计

```text
14 source hands
359 source link records
216 replaceable rigid-link candidates
19 provisional motor/interface families
100 generated hands
finger count: 4–5
DoF: 10–25
added fingers: 0
pure / mixed designs: 25 / 75
cross-source per-link replacements: 314
connected graphs: 100/100
acyclic graphs: 100/100
invalid motor-link bindings: 0
tendon metadata used by assembly: false
subtree mesh translations: 0
```

审计文件是 `temp/exp1/grammar_v1/outputs/grammar_audit.json`，MuJoCo 总览是
`temp/exp1/grammar_v1/outputs/grammar_100_hands.png`。

### 17.3 尚未声称完成的部分

公开 URDF 通常缺少真实 motor BOM，因此当前 19 个 family 是由 joint type、finger class、chain
stage 和 connector 几何形成的 prototype proxy，不是跨厂商物理电机等价证明。进入 Isaac Lab/SAC
前仍需补齐真实 motor family、collision recipe、mass/inertia、joint sweep、自碰撞、torque-speed、
导出和 Isaac Lab reload 验证。
