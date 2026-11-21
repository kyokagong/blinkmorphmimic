"""Data-only metadata needed by the borrowed Bumi retargeter, without IsaacLab."""
import re

from morphmimic.constants import JOINT_NAMES as G1_JOINT_NAMES
from supermimic.bumi import DEFAULT_JOINT_POS, JOINT_NAMES


def default_joint_pos_list():
    """The retarget solver initializes Bumi's 21 SDK-order joints, regardless of env."""
    return [next((value for pattern, value in DEFAULT_JOINT_POS.items()
                  if re.fullmatch(pattern, name)), 0.) for name in JOINT_NAMES]
