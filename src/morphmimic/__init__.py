"""MorphMimic. Register optional mjlab tasks explicitly with ``register_tasks``."""


def register_tasks() -> None:
  """Build task configs only when the mjlab registry is actually requested."""
  from mjlab.tasks.registry import register_mjlab_task
  from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
  from morphmimic.env.env_cfg import morphmimic_env_cfg
  from morphmimic.robots import ROBOTS

  for name in ROBOTS:
    register_mjlab_task(
      task_id=f'MorphMimic-Tracking-Flat-{name.upper()}',
      env_cfg=morphmimic_env_cfg(name),
      play_env_cfg=morphmimic_env_cfg(name, play=True),
      rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
    )
