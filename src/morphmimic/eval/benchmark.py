"""Benchmark evaluation (paper App. F).

Every clip of a subset is played exactly once from its first frame in the eval env (no
DR, no observation or reset noise, floor friction 1.0) with the deterministic policy.
The episode of a clip ends at the first training termination (anchor position /
orientation; the clip then counts as failed) or at the `clip_end` time-out on its last
frame.

Frames: the reset places the robot on frame 0 and the command shows frame 1, so the
transitions of a full clip are rewarded against frames 1 .. clip_len - 1, once each. The
normalised return is `sum r / (R_MAX * (clip_len - 1))`: 1 for perfect tracking of the
whole clip without penalties; a clip that fails early only collects the rewards up to
its failure, so it scores low. `source_*` separately measures the manifest's motion
body: ceil(entry*FPS) <= reference frame < ceil(dance*FPS), intersected with 1..L-1.
Its denominator includes the entire body even after an early failure; failure before
entry scores zero. Without phase metadata, source and full returns are identical.
Usage counters instead address transition starts and exclude the terminal frame.

Clips are handed out to the envs as they free up: every env plays its clip to the end,
then takes the next unplayed clip; envs left without one replay a dummy clip whose
results are ignored.

The policy acts through the motion goal or, for a human checkpoint, through the human
skeleton goal under a fixed mask (`evaluate`'s `interface`, `human_points`,
`kp_horizon`; `policy_loader.load_policy`); the benchmark is otherwise the same.

The same run also records the drift diagnostic (`drift.py`): per step the anchor xy,
heading and height error against the reference, summarised per clip into the
`drift.DRIFT_SUMMARY_COLUMNS` (a failed clip up to its failing step), as `run_drift`
gives them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from morphmimic.constants import (
  GOAL_DIM,
  GOAL_K,
  HIST_LEN,
  HUMAN_ENV_DIM,
  PROPRIO_DIM,
  R_MAX,
  GoalAnchor,
)
from morphmimic.data.manifest import load_subset
from morphmimic.env.commands import MultiMotionCommand, MultiMotionCommandCfg
from morphmimic.env.env_cfg import COMMAND_NAME, morphmimic_env_cfg
from morphmimic.env.morphmimic_env import MorphMimicEnv
from morphmimic.eval.drift import DRIFT_SUMMARY_COLUMNS, DriftTrace, summarize_drift
from morphmimic.eval.policy_loader import Interface, PolicyFn, load_policy
from morphmimic.model.human import HumanPoints
from morphmimic.train.collector import GOAL_GROUPS
from morphmimic.train.obs_history import ObsHistory

RESULT_COLUMNS = (
  "name",
  "frames",
  "failed",
  "fail_frame",
  "norm_return",
  "steps",
  "source_norm_return",
  "source_steps",
  "source_total_steps",
  "source_has_phases",
  *DRIFT_SUMMARY_COLUMNS,
)
"""Per clip: clip name, its frame count, whether a termination fired, the clip-local
frame the failing transition was compared against (-1 if none), the normalised return,
the number of rewarded steps played and the drift columns (`drift.summarize_drift`)."""
UNKNOWN_CATEGORY = "unknown"
PROGRESS_EVERY = 1000  # env steps between progress lines (verbose)


def make_eval_env(
  subset_yaml: str,
  num_envs: int,
  device: str,
  library_dtype: str = "float32",
  seed: int = 0,
  robot: str = "g1",
  builtin_pd: bool = False,
  joint_frame: str | None = None,
  imu_body: str | None = None,
  human_root: str | None = None,
  goal_anchor: GoalAnchor = "reference",
  allow_missing_human: bool = False,
  control_graph: bool = False,
) -> MorphMimicEnv:
  """The eval env (`morphmimic_env_cfg(robot, eval=True)`) over the clips of
  `subset_yaml` (which must be `robot`'s); `builtin_pd`, `joint_frame`, `imu_body`,
  `human_root` and `goal_anchor` as in `morphmimic_env_cfg`."""
  cfg = morphmimic_env_cfg(
    robot,
    subset_yaml=subset_yaml,
    num_envs=num_envs,
    eval=True,
    builtin_pd=builtin_pd,
    joint_frame=joint_frame,
    imu_body=imu_body,
    human_root=human_root,
    goal_anchor=goal_anchor,
  )
  cfg.seed = seed
  command_cfg = cfg.commands[COMMAND_NAME]
  assert isinstance(command_cfg, MultiMotionCommandCfg)
  command_cfg.library_dtype = library_dtype
  command_cfg.allow_missing_human = allow_missing_human
  if not control_graph:
    return MorphMimicEnv(cfg, device=device)
  if torch.device(device).type != 'cuda':
    raise ValueError('Evaluation control graph requires CUDA')
  from morphmimic.env.control_graph import PhysicsGraph
  stream = torch.cuda.Stream(device=device)
  env = MorphMimicEnv(cfg, device=device, graph_stream=stream)
  env._eval_physics_graph = PhysicsGraph(env)
  env._eval_graph_warmed = False
  return env


def _eval_graph_overflow(env,stage):
  if bool(env.physics_diagnostics.bits_tensor()!=0):
    raise RuntimeError(f'Native physics overflow after {stage}: {env.physics_diagnostics.report()}')


def prepare_eval_graph(env):
  graph=getattr(env,'_eval_physics_graph',None)
  if graph is None or env._eval_graph_warmed:return
  if 'reset' in env.event_manager.available_modes:
    raise ValueError('Evaluation graph warmup does not support reset-event side effects')
  command=env.command_manager.get_term(COMMAND_NAME)
  names=('G_b','past_softmax','global_frame','start_bin','disc_return','gamma_pow',
         'horizon_pow','episode_steps','time_left','command_counter','restart_frame','consecutive_local')
  state={name:getattr(command,name).clone() for name in names}
  metrics={name:value.clone() for name,value in command.metrics.items()}
  manager=env.termination_manager
  terminated=manager._terminated_buf.clone();truncated=manager._truncated_buf.clone()
  terms={name:value.clone() for name,value in manager._term_dones.items()}
  rewards={name:value.clone() for name,value in env.reward_manager._episode_sums.items()}
  pending=command._pending_forward;counters=env._sim_step_counter,env.common_step_counter
  devices=[] if torch.device(env.device).type!='cuda' else [torch.device(env.device)]
  try:
    with torch.random.fork_rng(devices=devices):
      env.reset()
      graph.warmup(torch.zeros(env.num_envs,29,device=env.device))
  finally:
    for name,value in state.items():getattr(command,name).copy_(value)
    for name,value in metrics.items():command.metrics[name].copy_(value)
    manager._terminated_buf.copy_(terminated);manager._truncated_buf.copy_(truncated)
    for name,value in terms.items():manager._term_dones[name].copy_(value)
    for name,value in rewards.items():env.reward_manager._episode_sums[name].copy_(value)
    command._pending_forward=pending
    env._sim_step_counter,env.common_step_counter=counters
  env._eval_graph_warmed=True


def eval_graph_step(env,action):
  if not bool(torch.isfinite(action).all()):raise ValueError('nonfinite action before evaluation physics graph')
  result=env._eval_physics_graph.step(action)
  _eval_graph_overflow(env,'evaluation graph step')
  return result


def source_reward_mask(library, frames: torch.Tensor) -> torch.Tensor:
  """Body reward refs for padded clips; whole clips keep all rewards 1..L-1."""
  clips = library.frame_clip[frames]
  local = frames - library.clip_start[clips]
  in_body = (local >= library.source_start[clips]) & (local < library.source_stop[clips])
  return (~library.source_has_phases[clips] | in_body) & (local >= 1) & (local < library.clip_len[clips])


def validate_windows(library,selected_ids,windows):
  """Clip-local (reset frame, last rewarded frame) bounds, with no truncation."""
  if len(windows)!=len(selected_ids):raise ValueError('Window count differs from selected clips')
  lengths=library.clip_len.cpu().tolist();result=[]
  for clip,bounds in zip(selected_ids,windows):
    if (not isinstance(bounds,(tuple,list)) or len(bounds)!=2 or
        any(type(x) is not int for x in bounds)):
      raise ValueError('Window bounds must be integer pairs')
    start,stop=bounds
    if not 0<=start<stop<lengths[clip]:raise ValueError('Window exceeds its source clip')
    result.append((start,stop))
  return result


@torch.no_grad()
def run_benchmark(
  env: MorphMimicEnv,
  policy_fn: PolicyFn,
  hist_len: int = HIST_LEN,
  verbose: bool = False,
  clip_ids: list[int] | None = None,
  deadline: float | None = None,
  windows: list[tuple[int,int]] | None = None,
) -> pd.DataFrame:
  """Plays every clip of `env`'s library once (see the module docstring) with
  `policy_fn(obs_hist_flat, goals) -> action` on the stacked `policy` obs (an
  `ObsHistory` of `hist_len` frames, refilled with the first obs at each clip start)
  and the goal obs (`GOAL_GROUPS`), and returns one row per clip, in library order, with
  RESULT_COLUMNS.

  `env` must be an eval env (`make_eval_env`); every env is reset first, so an env
  can be reused across calls.
  """
  command = env.command_manager.get_term(COMMAND_NAME)
  assert isinstance(command, MultiMotionCommand)
  if command.cfg.sampling != "start":
    raise ValueError("run_benchmark needs the eval env (sampling='start').")
  lib = command.library
  if (lib.clip_len < 2).any():
    raise ValueError("every clip needs at least 2 frames (one rewarded step)")
  device = env.device
  num_envs, num_clips = env.num_envs, lib.num_clips
  from morphmimic.train.probe_worker import validate_clip_ids
  selected_ids = list(range(num_clips)) if clip_ids is None else validate_clip_ids(clip_ids, num_clips)
  selected = torch.tensor(selected_ids, dtype=torch.long, device=device)
  num_selected = len(selected_ids)
  window_bounds = None if windows is None else validate_windows(lib,selected_ids,windows)
  window_start = None if window_bounds is None else torch.tensor([x[0] for x in window_bounds],device=device,dtype=torch.long)
  window_stop = None if window_bounds is None else torch.tensor([x[1] for x in window_bounds],device=device,dtype=torch.long)
  env_window_stop = None if window_bounds is None else torch.zeros(num_envs,device=device,dtype=torch.long)

  # Per env: the clip it plays (-1 while idle on a dummy clip), the reward sum and the
  # rewarded steps of its running episode.
  env_clip = torch.full((num_envs,), -1, dtype=torch.long, device=device)
  ep_return = torch.zeros(num_envs, dtype=torch.float64, device=device)
  ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
  ep_source_return = torch.zeros_like(ep_return)
  ep_source_steps = torch.zeros_like(ep_steps)
  trace = DriftTrace(command)
  history = ObsHistory(num_envs, hist_len, PROPRIO_DIM, device)
  goals = {
    "goal": torch.zeros(num_envs, GOAL_DIM, device=device),
    "goal_kp": torch.zeros(num_envs, HUMAN_ENV_DIM, device=device),
  }
  # Per clip results.
  failed = torch.zeros(num_clips, dtype=torch.bool, device=device)
  window_completed = torch.zeros(num_clips, dtype=torch.bool, device=device)
  fail_frame = torch.full((num_clips,), -1, dtype=torch.long, device=device)
  norm_return = torch.zeros(num_clips, dtype=torch.float64, device=device)
  steps = torch.zeros(num_clips, dtype=torch.long, device=device)
  source_norm_return = torch.zeros_like(norm_return)
  source_steps = torch.zeros_like(steps)
  source_total_steps = torch.where(lib.source_has_phases,
      (lib.source_stop - lib.source_start.clamp_min(1)).clamp_min(0), lib.clip_len - 1)
  drift: list[dict[str, float]] = [{}] * num_clips
  rewarded = (lib.clip_len - 1).double()  # frames 1 .. clip_len - 1
  dt = float(env.step_dt)

  next_clip = 0

  def start(env_ids: torch.Tensor) -> None:
    """Gives `env_ids` the next unplayed clips (in order) and resets them onto their
    first frame; envs beyond the last clip get a dummy clip."""
    nonlocal next_clip
    clips = next_clip + torch.arange(env_ids.numel(), device=device)
    next_clip += env_ids.numel()
    indices = clips.clamp(max=num_selected - 1)
    assigned = selected[indices]
    env_clip[env_ids] = torch.where(clips < num_selected, assigned, -1)
    if window_bounds is None:
      command.assign_clips(env_ids, assigned)
    else:
      command.assign_clips(env_ids, assigned, window_start[indices])
      env_window_stop[env_ids] = lib.clip_start[assigned] + window_stop[indices]
    obs, _ = env.reset(env_ids=env_ids)
    if getattr(env,'_eval_physics_graph',None) is not None:_eval_graph_overflow(env,'evaluation reset')
    history.reset(env_ids, obs["policy"][env_ids])
    for g in GOAL_GROUPS:
      goals[g][env_ids] = obs[g][env_ids]
    ep_return[env_ids] = 0.0
    ep_steps[env_ids] = 0
    ep_source_return[env_ids] = 0.0
    ep_source_steps[env_ids] = 0

  if deadline is not None:
    import time
    if time.monotonic() >= deadline:raise TimeoutError('Probe budget deadline reached')
  prepare_eval_graph(env)
  start(torch.arange(num_envs, device=device))
  num_done = 0
  step = 0
  while num_done < num_selected:
    if deadline is not None:
      import time
      if time.monotonic() >= deadline:raise TimeoutError('Probe budget deadline reached')
    frame = command.global_frame.clone()
    if lib.has_human:
      goals["human_joint_valid"] = lib.human_valid_at(frame)
    action = policy_fn(history.flat(), goals)
    obs, reward, terminated, truncated, _ = (env.step(action) if getattr(env,'_eval_physics_graph',None) is None
                                            else eval_graph_step(env,action))
    ep_return += reward.double()
    source = source_reward_mask(lib, frame)
    ep_source_return += reward.double() * source
    ep_source_steps += source.long()
    trace.record(ep_steps)
    ep_steps += 1
    history.push(obs["policy"])
    goals = {g: obs[g].clone() for g in GOAL_GROUPS}
    step += 1

    hit = torch.zeros_like(terminated) if window_bounds is None else (frame >= env_window_stop) & (env_clip >= 0)
    finished = (terminated | truncated | hit).nonzero().squeeze(-1)
    if finished.numel() > 0:
      ids = finished[env_clip[finished] >= 0]
      clips = env_clip[ids]
      term = terminated[ids]
      failed[clips] = term
      if window_bounds is not None:window_completed[clips] = hit[ids] & ~term
      local = frame[ids] - lib.clip_start[clips]
      fail_frame[clips] = torch.where(term, local, torch.full_like(local, -1))
      denominator = rewarded[clips] if window_bounds is None else ep_steps[ids].clamp_min(1).double()
      norm_return[clips] = ep_return[ids] / (R_MAX * denominator)
      steps[clips] = ep_steps[ids]
      source_norm_return[clips] = torch.where(source_total_steps[clips] > 0,
          ep_source_return[ids] / (R_MAX * source_total_steps[clips].clamp_min(1).double()), torch.nan)
      source_steps[clips] = ep_source_steps[ids]
      for c, series in zip(clips.tolist(), trace.take(ids, ep_steps[ids])):
        drift[c] = summarize_drift(*series, dt)
      num_done += ids.numel()
      if num_done < num_selected:
        start(finished)
    if verbose and step % PROGRESS_EVERY == 0:
      print(f"[morphmimic] eval step {step}: {num_done}/{num_selected} clips", flush=True)

  if window_bounds is not None:
    return pd.DataFrame({
      "name": [lib.clip_names[i] for i in selected_ids],
      "frames": lib.clip_len[selected].cpu().numpy(),
      "failed": failed[selected].cpu().numpy(),
      "fail_frame": fail_frame[selected].cpu().numpy(),
      "steps": steps[selected].cpu().numpy(),
      "window_start_frame": [x[0] for x in window_bounds],
      "window_end_frame": [x[1] for x in window_bounds],
      "window_completed": window_completed[selected].cpu().numpy(),
      "window_norm_return": norm_return[selected].cpu().numpy(),
      "metric_scope": ['probe_window'] * num_selected,
      **{k: [drift[i][k] for i in selected_ids] for k in DRIFT_SUMMARY_COLUMNS},
    })

  return pd.DataFrame(
    {
      "name": [lib.clip_names[i] for i in selected_ids],
      "frames": lib.clip_len[selected].cpu().numpy(),
      "failed": failed[selected].cpu().numpy(),
      "fail_frame": fail_frame[selected].cpu().numpy(),
      "norm_return": norm_return[selected].cpu().numpy(),
      "steps": steps[selected].cpu().numpy(),
      "source_norm_return": source_norm_return[selected].cpu().numpy(),
      "source_steps": source_steps[selected].cpu().numpy(),
      "source_total_steps": source_total_steps[selected].cpu().numpy(),
      "source_has_phases": lib.source_has_phases[selected].cpu().numpy(),
      **{k: [drift[i][k] for i in selected_ids] for k in DRIFT_SUMMARY_COLUMNS},
    },
    columns=list(RESULT_COLUMNS),
  )


def evaluate(
  ckpt: str | Path,
  subset_yaml: str,
  device: str,
  num_envs: int = 4096,
  library_dtype: str = "float32",
  verbose: bool = False,
  robot: str | None = None,
  interface: Interface = "motion",
  human_points: HumanPoints = "full",
  kp_horizon: int = GOAL_K,
  human_root: str | None = None,
  human_heading: bool = True,
  builtin_pd: bool = False,
  policy_joint_frame: bool = False,
  morph_robot: str | None = None,
  morph_mix: tuple[str, str, float] | None = None,
  morph_vector: torch.Tensor | None = None,
  norm_from: str | None = None,
  imu_body: str | None = None,
  goal_anchor: GoalAnchor = "reference",
  human_drift: bool = False,
  control_graph: bool = False,
) -> pd.DataFrame:
  """Benchmark of checkpoint `ckpt` on the clips of `subset_yaml` with up to
  `num_envs` parallel envs (never more than there are clips), on `robot` (default: the
  checkpoint's robot). Returns `run_benchmark`'s per-clip table; `df.attrs` holds
  `ckpt`, `iteration`, `subset_yaml` and `mode` ("motion" or
  "human-<points>-h<horizon>", "-noyaw" added without the relative orientation).

  `interface`, `human_points`, `kp_horizon`, `human_heading` and `human_drift` select
  the goal the policy acts on, as in `load_policy` (which also rejects an invalid selection, before
  any env is built).
  `human_root` is where the env finds the clips' human sidecars (`morphmimic_env_cfg`);
  the human interface raises ValueError when the clips have none (its goal would be
  zeros).

  Transfer: `builtin_pd`, implicit PD on every joint;
  `policy_joint_frame`, the canonical joints use the action scale and default pose of
  the checkpoint's robot, not `robot`'s; `morph_robot`, the actor morphology vector of
  that robot instead of the stored one; `morph_mix=(a, b, alpha)`, the vector
  `(1 - alpha) * vec(a) + alpha * vec(b)`; `morph_vector`, a given actor vector (all
  `load_policy`); `imu_body`, gyro and
  gravity read on that tracked body (`morphmimic_env_cfg`). `goal_anchor` is the
  motion goal's frame, the checkpoint's (`load_policy` refuses another)."""
  policy_fn, meta = load_policy(
    ckpt,
    device,
    interface,
    human_points,
    kp_horizon,
    morph_robot=morph_robot,
    morph_mix=morph_mix,
    morph_vector=morph_vector,
    norm_from=norm_from,
    goal_anchor=goal_anchor,
    human_heading=human_heading,
    human_drift=human_drift,
  )
  num_envs = min(num_envs, len(load_subset(subset_yaml)))
  env = make_eval_env(
    subset_yaml,
    num_envs,
    device,
    library_dtype,
    robot=robot or meta.robot,
    builtin_pd=builtin_pd,
    joint_frame=meta.robot if policy_joint_frame else None,
    imu_body=imu_body,
    human_root=human_root,
    goal_anchor=goal_anchor,
    control_graph=control_graph,
  )
  try:
    if interface == "human":
      require_human_sidecars(env, subset_yaml, human_root)
    df = run_benchmark(env, policy_fn, meta.hist_len, verbose=verbose)
    if interface == 'human':
      lib=env.command_manager.get_term(COMMAND_NAME).library
      counts,clips=lib.human_joint_valid.sum(-1).unique(return_counts=True)
      df.attrs['available_human_joints']={str(n):int(c) for n,c in zip(counts.tolist(),clips.tolist())}
    physics = env.physics_diagnostics.report()
    if physics["overflow_bits"]:
      raise RuntimeError(f"Invalid evaluation physics overflow: {physics}")
    df.attrs["physics"] = physics
  finally:
    env.close()
  df.attrs.update(
    ckpt=meta.path, iteration=meta.iteration, subset_yaml=subset_yaml, mode=meta.mode
  )
  return df


def require_human_sidecars(
  env: MorphMimicEnv, subset_yaml: str, human_root: str | None
) -> None:
  """Raise ValueError when the env's clips have no human sidecars: the `goal_kp` group
  is then zeros, and a policy acting through the human interface would act on an
  empty goal (shared by the benchmark and `scripts.play`)."""
  command = env.command_manager.get_term(COMMAND_NAME)
  assert isinstance(command, MultiMotionCommand)
  if not command.library.has_human:
    raise ValueError(
      f"the human interface needs human sidecars; the clips of {subset_yaml}"
      f" have none (human_root {human_root or '<clip root>/human50'})"
    )


def summarize(df: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
  """Aggregate benchmark numbers of a `run_benchmark` table. Categories come from the
  manifest's `category` column (joined on `name`); clips missing from the manifest
  count as UNKNOWN_CATEGORY. Per-category dicts cover every category in `df`. `drift`
  holds the means over the clips of `mean_xy`, `final_xy` and `mean_abs_yaw_deg` and
  the number of clips whose xy / heading error crossed `drift.XY_THRESHOLD` /
  `drift.YAW_THRESHOLD_DEG` (`clips_xy_over`, `clips_yaw_over`)."""
  categories = manifest.drop_duplicates("name").set_index("name")["category"]
  category = df["name"].map(categories).fillna(UNKNOWN_CATEGORY)
  category = category.where(category != "", UNKNOWN_CATEGORY).rename("category")
  groups = df.groupby(category)
  num_clips = len(df)
  source = None
  if {"source_norm_return", "source_steps", "source_total_steps"}.issubset(df.columns):
    evaluable = df["source_total_steps"] > 0
    body = df[evaluable]
    source = {
      "mean_norm_return": float(body["source_norm_return"].mean()),
      "median_norm_return": float(body["source_norm_return"].median()),
      "steps": int(df["source_steps"].sum()), "total_steps": int(df["source_total_steps"].sum()),
      "evaluable_clips": int(evaluable.sum()), "reached_clips": int((df["source_steps"] > 0).sum()),
      "completed_clips": int((evaluable & (df["source_steps"] == df["source_total_steps"])).sum()),
      "mean_step_fraction": float((body["source_steps"] / body["source_total_steps"]).mean()),
      "mean_norm_return_by_category": {str(k): float(v) for k, v in groups["source_norm_return"].mean().items()},
    }
  return {
    "num_clips": num_clips,
    "source": source,
    "failures": int(df["failed"].sum()),
    "failure_rate": float(df["failed"].mean()) if num_clips else math.nan,
    "mean_norm_return": float(df["norm_return"].mean()),
    "median_norm_return": float(df["norm_return"].median()),
    "clips_by_category": {str(k): int(v) for k, v in groups.size().items()},
    "failures_by_category": {str(k): int(v) for k, v in groups["failed"].sum().items()},
    "mean_norm_return_by_category": {
      str(k): float(v) for k, v in groups["norm_return"].mean().items()
    },
    "drift": {
      "mean_xy": float(df["mean_xy"].mean()),
      "final_xy": float(df["final_xy"].mean()),
      "mean_abs_yaw_deg": float(df["mean_abs_yaw_deg"].mean()),
      "clips_xy_over": int(df["t_xy_over"].notna().sum()),
      "clips_yaw_over": int(df["t_yaw_over"].notna().sum()),
    },
  }
