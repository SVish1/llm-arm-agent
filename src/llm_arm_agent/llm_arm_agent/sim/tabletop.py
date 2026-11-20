"""MuJoCo tabletop simulation shared by the standalone backend and the ROS 2 sim node.

The sim owns physics, the gripper and grasp detection, and exposes a
ground-truth state snapshot. It knows nothing about planning: callers hand
it joint trajectories (from MoveIt or from the built-in IK) to execute.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from importlib import resources
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import mujoco
import numpy as np

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
READY_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

OBJECTS = {
    "red_cube": {"color": "red", "size": 0.04},
    "green_cube": {"color": "green", "size": 0.04},
    "blue_cube": {"color": "blue", "size": 0.04},
    "yellow_cube": {"color": "yellow", "size": 0.04},
}
ZONES = {
    "left_pad": {"xy": (0.45, 0.25), "radius": 0.06},
    "right_pad": {"xy": (0.45, -0.25), "radius": 0.06},
}
TABLE_TOP_Z = 0.0
TABLE_X = (-0.15, 0.95)
TABLE_Y = (-0.6, 0.6)
GRIPPER_OPEN = 0.04  # per-finger opening, metres (Panda max is 0.04)
# Where unused objects are parked so they do not affect a task.
PARK_POSITIONS = {
    "red_cube": (0.85, 0.5),
    "green_cube": (0.85, 0.42),
    "blue_cube": (0.85, -0.42),
    "yellow_cube": (0.85, -0.5),
}


def default_model_path() -> str:
    return str(resources.files("llm_arm_agent.sim").joinpath("assets/panda_tabletop.xml"))


def yaw_from_quat(q: Sequence[float]) -> float:
    """Yaw (rotation about world z) of a wxyz quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def topdown_quat(yaw: float) -> np.ndarray:
    """wxyz quaternion for a gripper pointing straight down, rotated by `yaw` about world z.

    Equal to Rz(yaw) * Rx(pi); with yaw=0 the hand x-axis is aligned with world x.
    """
    c, s = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    # Rz(yaw) = (c, 0, 0, s); Rx(pi) = (0, 1, 0, 0); product:
    return np.array([0.0, c, s, 0.0])


@dataclass
class Grasp:
    name: str
    rel_pos: np.ndarray  # object position in TCP frame
    rel_quat: np.ndarray  # object orientation relative to TCP (wxyz)


class TabletopSim:
    """Physics + gripper + grasp bookkeeping for the Panda tabletop scene."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = mujoco.MjModel.from_xml_path(model_path or default_model_path())
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        m = self.model
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.finger_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in FINGER_JOINTS])
        self.arm_act = np.array([m.actuator(f"act_joint{i}").id for i in range(1, 8)])
        self.finger_act = np.array([m.actuator(n).id for n in ("act_finger1", "act_finger2")])
        self.tcp_site = m.site("tcp").id
        self.obj_qadr = {n: m.jnt_qposadr[m.joint(f"{n}_joint").id] for n in OBJECTS}
        self.obj_dadr = {n: m.jnt_dofadr[m.joint(f"{n}_joint").id] for n in OBJECTS}
        self.arm_lower = np.array([m.jnt_range[m.joint(j).id][0] for j in ARM_JOINTS])
        self.arm_upper = np.array([m.jnt_range[m.joint(j).id][1] for j in ARM_JOINTS])
        self.grasp: Optional[Grasp] = None
        self.gripper_cmd = GRIPPER_OPEN
        self.reset()

    # ------------------------------------------------------------------ scene
    def reset(self, scene: Optional[Dict] = None) -> None:
        """Reset the arm to the ready pose and place objects.

        scene = {"objects": {"red_cube": [x, y, yaw_deg] or [x, y, z, yaw_deg], ...},
                 "arm_q": [7 joint angles]  (optional, defaults to the ready pose)}
        Objects not listed are parked at the far edge of the table.
        """
        scene = scene or {}
        placements = scene.get("objects", {})
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            self.grasp = None
            d = self.data
            arm_q = np.asarray(scene.get("arm_q", READY_Q), dtype=float)
            d.qpos[self.arm_qadr] = arm_q
            d.qpos[self.finger_qadr] = GRIPPER_OPEN
            d.ctrl[self.arm_act] = arm_q
            d.ctrl[self.finger_act] = GRIPPER_OPEN
            self.gripper_cmd = GRIPPER_OPEN
            for name in OBJECTS:
                if name in placements:
                    p = list(placements[name])
                    if len(p) == 3:
                        x, y, yaw = p
                        z = OBJECTS[name]["size"] / 2
                    else:
                        x, y, z, yaw = p
                else:
                    (x, y), z, yaw = PARK_POSITIONS[name], OBJECTS[name]["size"] / 2, 0.0
                a = self.obj_qadr[name]
                half = math.radians(yaw) / 2.0
                d.qpos[a:a + 7] = [x, y, z, math.cos(half), 0.0, 0.0, math.sin(half)]
            mujoco.mj_forward(self.model, d)
        self.step_for(0.3)  # let objects settle

    # --------------------------------------------------------------- stepping
    def step(self, n: int = 1) -> None:
        with self.lock:
            for _ in range(n):
                mujoco.mj_step(self.model, self.data)
                if self.grasp is not None:
                    self._carry_grasped()

    def step_for(self, seconds: float) -> None:
        self.step(max(1, int(round(seconds / self.model.opt.timestep))))

    @property
    def time(self) -> float:
        return float(self.data.time)

    # ------------------------------------------------------------- arm access
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    def set_arm_target(self, q: Sequence[float]) -> None:
        with self.lock:
            self.data.ctrl[self.arm_act] = np.clip(q, self.arm_lower, self.arm_upper)

    def joint_state(self) -> Dict[str, List[float]]:
        with self.lock:
            names = ARM_JOINTS + FINGER_JOINTS
            adr = [self.model.jnt_qposadr[self.model.joint(n).id] for n in names]
            dadr = [self.model.jnt_dofadr[self.model.joint(n).id] for n in names]
            return {
                "name": names,
                "position": self.data.qpos[adr].tolist(),
                "velocity": self.data.qvel[dadr].tolist(),
            }

    def tcp_pose(self) -> tuple:
        with self.lock:
            pos = self.data.site_xpos[self.tcp_site].copy()
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.tcp_site])
            return pos, quat

    def execute_joint_trajectory(
        self,
        times: Sequence[float],
        positions: Sequence[Sequence[float]],
        settle: float = 0.25,
        on_step: Optional[Callable[[], None]] = None,
    ) -> float:
        """Track a time-parameterised joint trajectory (linear interpolation of setpoints).

        Runs as fast as possible in sim time. Returns the simulated duration.
        """
        times = np.asarray(times, dtype=float)
        positions = np.asarray(positions, dtype=float)
        if len(times) == 0:
            return 0.0
        t0 = self.time
        dt = self.model.opt.timestep
        t = 0.0
        while t <= times[-1]:
            q = np.array([np.interp(t, times, positions[:, j]) for j in range(7)])
            self.set_arm_target(q)
            self.step(1)
            if on_step is not None:
                on_step()
            t += dt
        self.set_arm_target(positions[-1])
        self.step_for(settle)
        return self.time - t0

    # ---------------------------------------------------------------- gripper
    def command_gripper(self, width: float) -> None:
        """Set per-finger opening target (0 = closed, 0.04 = fully open)."""
        with self.lock:
            self.gripper_cmd = float(np.clip(width, 0.0, GRIPPER_OPEN))
            self.data.ctrl[self.finger_act] = self.gripper_cmd

    def gripper_width(self) -> float:
        """Distance between finger pads (metres)."""
        return float(self.data.qpos[self.finger_qadr].sum())

    def open_gripper(self, duration: float = 0.5) -> None:
        self.release()
        self.command_gripper(GRIPPER_OPEN)
        self.step_for(duration)

    def close_gripper(self, duration: float = 0.6) -> Optional[str]:
        """Close the fingers and attach an object if one is between them."""
        self.command_gripper(0.0)
        self.step_for(duration)
        return self.try_attach()

    def try_attach(self) -> Optional[str]:
        """Grasp assist: if a cube sits between closed pads, rigidly carry it.

        Pure friction grasps of light cubes are fragile at sim timesteps a
        benchmark can afford, so once the pads are closed on an object we pin
        it to the TCP until the gripper opens. Detection is geometric: the
        cube centre must be inside the pad volume and the fingers must have
        stopped at roughly the cube width.
        """
        with self.lock:
            if self.grasp is not None:
                return self.grasp.name
            tcp_pos, tcp_quat = self.tcp_pose()
            rot = self.data.site_xmat[self.tcp_site].reshape(3, 3)
            width = self.gripper_width()
            for name, info in OBJECTS.items():
                a = self.obj_qadr[name]
                p = self.data.qpos[a:a + 3]
                local = rot.T @ (p - tcp_pos)
                size = info["size"]
                inside = abs(local[0]) < 0.02 and abs(local[1]) < 0.02 and abs(local[2]) < 0.022
                if inside and size * 0.7 < width < size * 1.3:
                    inv = np.zeros(4)
                    mujoco.mju_negQuat(inv, tcp_quat)
                    rel_q = np.zeros(4)
                    mujoco.mju_mulQuat(rel_q, inv, self.data.qpos[a + 3:a + 7])
                    self.grasp = Grasp(name, local.copy(), rel_q)
                    return name
            return None

    def release(self) -> None:
        with self.lock:
            self.grasp = None

    def _carry_grasped(self) -> None:
        g = self.grasp
        rot = self.data.site_xmat[self.tcp_site].reshape(3, 3)
        tcp_pos = self.data.site_xpos[self.tcp_site]
        tcp_quat = np.zeros(4)
        mujoco.mju_mat2Quat(tcp_quat, self.data.site_xmat[self.tcp_site])
        a, v = self.obj_qadr[g.name], self.obj_dadr[g.name]
        self.data.qpos[a:a + 3] = tcp_pos + rot @ g.rel_pos
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, tcp_quat, g.rel_quat)
        self.data.qpos[a + 3:a + 7] = q
        self.data.qvel[v:v + 6] = 0.0

    # ------------------------------------------------------------------ state
    def state(self) -> Dict:
        """Ground-truth snapshot (JSON-serialisable)."""
        with self.lock:
            tcp_pos, tcp_quat = self.tcp_pose()
            objects = {}
            for name, info in OBJECTS.items():
                a = self.obj_qadr[name]
                p = self.data.qpos[a:a + 3]
                objects[name] = {
                    "position": [round(float(v), 4) for v in p],
                    "yaw_deg": round(math.degrees(yaw_from_quat(self.data.qpos[a + 3:a + 7])), 1),
                    "color": info["color"],
                    "size": info["size"],
                }
            return {
                "time": round(self.time, 3),
                "ee_position": [round(float(v), 4) for v in tcp_pos],
                "ee_yaw_deg": round(math.degrees(_gripper_yaw(tcp_quat)), 1),
                "gripper_width": round(self.gripper_width(), 4),
                "held_object": self.grasp.name if self.grasp else None,
                "arm_joints": [round(float(v), 4) for v in self.arm_q()],
                "objects": objects,
                "zones": {k: {"center": list(v["xy"]), "radius": v["radius"]} for k, v in ZONES.items()},
            }


def _gripper_yaw(quat: Iterable[float]) -> float:
    """Yaw of the hand x-axis projected on the table plane."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    x_axis = mat.reshape(3, 3)[:, 0]
    return math.atan2(x_axis[1], x_axis[0])
