"""Backend interface: the motion primitives every execution stack must provide.

Two implementations exist:
  * DirectBackend  - in-process MuJoCo + damped-least-squares IK (no ROS needed)
  * MoveItBackend  - ROS 2: MoveIt 2 move_group plans, the MuJoCo sim node executes

Higher-level skills (pick, place, ...) live in llm_arm_agent.skills and are
written once against this interface.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


@dataclass
class MotionResult:
    ok: bool
    message: str = ""
    plan_time: float = 0.0  # seconds spent planning (IK or MoveIt)
    exec_time: float = 0.0  # seconds spent executing (sim or wall time)
    extra: Dict = field(default_factory=dict)


class ArmBackend(abc.ABC):
    name = "abstract"

    @abc.abstractmethod
    def reset(self, scene: Optional[Dict] = None) -> None:
        """Reset the robot to its ready pose and lay out the scene."""

    @abc.abstractmethod
    def get_state(self) -> Dict:
        """Ground-truth state snapshot (see TabletopSim.state)."""

    @abc.abstractmethod
    def move_to(
        self,
        position: Sequence[float],
        yaw: float,
        linear: bool = False,
        ignore_objects: Sequence[str] = (),
    ) -> MotionResult:
        """Move the TCP to `position` with a top-down gripper rotated by `yaw` (rad).

        linear=True requests a straight-line Cartesian path (approach/retreat).
        ignore_objects lists objects the planner may touch (grasp target, support).
        """

    @abc.abstractmethod
    def move_home(self) -> MotionResult:
        """Joint-space move to the 'ready' configuration."""

    @abc.abstractmethod
    def set_gripper(self, open_: bool) -> MotionResult:
        """Open or close the gripper. On close, extra['held'] is the grasped object or None."""

    def close(self) -> None:  # pragma: no cover - optional hook
        pass
