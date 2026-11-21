"""Lazy robot registry: Q1, Bumi and G1. Importing compiles no model."""

from __future__ import annotations

from functools import cache

from morphmimic.robots.spec import RobotSpec, Translator

ROBOTS: tuple[str, ...] = ("q1", "bumi", "g1")
"""The training robots: the morphology scaling is fixed over these."""
VARIANTS: tuple[str, ...] = ()
"""Bodies outside the morphology scaling's training set; scaled against `ROBOTS`."""

__all__ = [
  "ROBOTS",
  "VARIANTS",
  "RobotSpec",
  "Translator",
  "check_imu_body",
  "get_robot",
]


@cache
def get_robot(name: str) -> RobotSpec:
  if name == "g1":
    from morphmimic.robots.g1 import g1_spec

    return g1_spec()
  if name == "q1":
    from morphmimic.robots.q1 import q1_spec
    return q1_spec()
  if name == "bumi":
    from morphmimic.robots.bumi import bumi_spec
    return bumi_spec()
  raise ValueError(f"unknown robot {name!r}; expected one of {ROBOTS + VARIANTS}")


def check_imu_body(ckpt: dict, path: object) -> None:
  """Raises ValueError when checkpoint `ckpt` (at `path`) was trained with its robot's
  gyro and gravity read on another body than `RobotSpec.imu_body`. A checkpoint
  without the `imu_body` entry read them on tracked slot 0: the same body on the G1
  and X2, the T2's waist link instead of its trunk."""
  robot = ckpt.get("cfg", {}).get("robot", "g1")
  spec = get_robot(robot)
  saved = ckpt.get("imu_body", spec.tracked_bodies[0])
  if saved != spec.imu_body:
    raise ValueError(
      f"checkpoint {path} was trained with the {robot} IMU on {saved}; this code reads"
      f" it on {spec.imu_body}, so its policy inputs differ"
    )
