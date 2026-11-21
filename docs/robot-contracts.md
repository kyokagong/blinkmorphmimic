# Robot and sparse-state contracts

Nominal asset and control contracts. Runtime/checkpoint metadata defines the actual trained controller; the optimized runs also use native actuator delay.

| Robot | Native hinge count | Physics | Control | IMU body |
|---|---:|---:|---:|---|
| Q1 | 22 | 10 × 2 ms | 50 Hz | pelvis |
| Bumi | 21 | 10 × 2 ms | 50 Hz | base_link |
| G1 | 29 | 4 × 5 ms | 50 Hz | pelvis |

Only these robots are registered. Importing the package has no mjlab task-registry
side effects; `morphmimic.register_tasks()` provides an explicit optional registry.

Assets are read under `BLINKMORPH_DATA_ROOT`, default `/data/kk/data`. Q1 uses
`blinkmimic_q1/20261107/robot/q1.xml`; its XML, training-reference JSON and deployed
hardware JSON are SHA256-bound in the adapter. Bumi uses
`supermimic_bumi/robot/bumi.xml`. G1 uses the pinned mjlab asset package.
Original source projects and asset files are never edited.

## Native/canonical boundary

The29 canonical G1 slots are unchanged. Missing slots have `perm=-1` and explicit
`joint_mask=False`. Canonical q/qd/limits are zero there. Native output scatters
only real indices, so missing action values have zero effect and gradient.
Missing slots are excluded from action-rate and joint-limit penalties, WM joint
state losses and output state, sensor noise, normalized policy history/goals and
policy actions. WM and decoder physical masks are saved in their state dicts.
The full morphology is154 values: the original125 descriptors plus29 binary
mask values. The no-actor-geometry variant is145. Positive placeholders prevent
log(0), followed by zeroing of missing scaled actuator features.

Bumi's simulator native order is XML depth-first order: waist, left arm4, right
arm4, left leg6, right leg6. Its source CSV/SDK order is interleaved. Conversion
must reorder columns by name. Q1's source22 order matches its XML. Tests assert
exact native order, not just name-set equality. Bumi elbow native ranges are
[-2.26,0] around +Y, whereas canonical flexion is positive: both elbows use sign=-1.

The14 Bumi body slots follow the original canonical functional ordering. The
asset has no wrist joint or distinct hand body. Each elbow gets one massless,
fixed `l_hand_frame`/`r_hand_frame` child at local `(0,0,-0.18702218)` metres. The
z coordinate is the distal bound of each actual elbow STL; the frame lies on the
forearm's local longitudinal axis and inherits its orientation. It adds no joint,
collision, inertia or independently realizable wrist orientation. Both hands are
unique frames, never aliases of elbows. Foot collision names are assigned in the
in-memory MuJoCo spec to the real ankle-roll collision meshes for friction DR.

## Calibration scope

### Bumi collision correction at iteration15000

Final configuration preserves every collision geom that is active in the source
asset. A real `mjlab.Entity(...).spec.compile()` audit found the first Bumi adapter
kept only its2 foot geoms: `CollisionCfg` defaults `disable_other_geoms=True`.
This was an adapter defect. It is now explicitly False, retaining all13 originally
active geoms and every original0/0 visual geom. Foot condim3/priority1/friction0.6
are retained; native CPU contacts confirm torso/arms/legs can touch the floor.
Q1 retains all48 source-active geoms and G1 all33, with their masks unchanged.

The first15000 iterations and Bumi reports from runtime-v1..v4 use the old feet-only
model and do not qualify whole-body contacts. Fixed15000 checkpoints are retained
as warm-start sources. Corrected training continues in a separate run; later
checkpoints save the requested collision configuration in `robot_contract.collision_cfg`.
Data poses/FK, source XML, mass, inertia, joints and controller values are unchanged.

Q1 copies the hash-bound deployed home angles, kp/kd and software effort limits,
action scale0.25 and nominal armature0.002. This release uses ideal MuJoCo built-in
position PD. It does **not** implement BlinkMimic's measured HAL timing queue,
prediction reuse, command slew, tracking-error correction, speed taper or tilt
guards. Nominal geometry is not a dynamics identification. This is simulation
training, not a claim of execution-stack equivalence or hardware qualification.

Bumi copies the conservative source effort/kp/kd/armature/default-pose values and
0.25×effort/kp action scales from `supermimic_vibed/supermimic/bumi.py` and
`robot.py`; source10×2ms physics is preserved. Source speed limits are documented
in MOTORS but ideal built-in PD has no torque-speed curve or velocity controller.

## Physics diagnostics

Native `wp_data.overflow` is sampled after every substep. Process-local sticky
per-world OR bits and observation counters survive episode resets. Any rank's
nonzero log-window signal stops all training ranks; preview also stops playback.
Cross-rank MAX is a stop signal, not a union of distinct rank bit patterns.
Counters count flagged observations, not independent events. Older checkpoints
without diagnostics expose null training coverage; resume does not reconstruct
unobserved history.

Metadata binds engine versions, EPA horizon, type hashes and physics patches.
EPA face exhaustion has no native diagnostic bit. Zero overflow does not prove
all contacts or solver behavior correct. Nominal PD, geometry and simulation
checks do not qualify real hardware timing, torque-speed response or safety.
