"""Standalone backend: MuJoCo sim + damped-least-squares IK, no ROS required.

Used for fast benchmark iteration and CI. It has no collision-aware planning;
the skills layer compensates by always approaching from a safe height.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import mujoco
import numpy as np

from ..sim.tabletop import READY_Q, TabletopSim, topdown_quat
from .base import ArmBackend, MotionResult

MAX_JOINT_SPEED = 1.2  # rad/s used for joint-space time scaling
LINEAR_SPEED = 0.15  # m/s for Cartesian moves
POSITION_TOL = 0.01  # m, acceptable final TCP error


class IKSolver:
    """Position + orientation IK for the TCP site via damped least squares."""

    def __init__(self, sim: TabletopSim):
        self.sim = sim
        self.model = sim.model
        self.data = mujoco.MjData(sim.model)  # scratch copy, never stepped
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

    def solve(
        self,
        target_pos: Sequence[float],
        target_quat: np.ndarray,
        q_seed: np.ndarray,
        iters: int = 300,
        damping: float = 1e-2,
        pos_tol: float = 5e-4,
        rot_tol: float = 5e-3,
    ) -> Optional[np.ndarray]:
        sim, d = self.sim, self.data
        target_pos = np.asarray(target_pos, dtype=float)
        q = np.array(q_seed, dtype=float)
        cols = sim.arm_dadr
        site_quat = np.zeros(4)
        neg = np.zeros(4)
        err_quat = np.zeros(4)
        rot_err = np.zeros(3)
        for _ in range(iters):
            d.qpos[sim.arm_qadr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            pos_err = target_pos - d.site_xpos[sim.tcp_site]
            mujoco.mju_mat2Quat(site_quat, d.site_xmat[sim.tcp_site])
            mujoco.mju_negQuat(neg, site_quat)
            mujoco.mju_mulQuat(err_quat, target_quat, neg)
            if err_quat[0] < 0:  # shortest rotation
                err_quat *= -1
            mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)
            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
                return q
            mujoco.mj_jacSite(self.model, d, self.jacp, self.jacr, sim.tcp_site)
            J = np.vstack([self.jacp[:, cols], self.jacr[:, cols]])
            err = np.concatenate([pos_err, rot_err])
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            step = np.linalg.norm(dq)
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, sim.arm_lower, sim.arm_upper)
        return None


def _quintic(n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class DirectBackend(ArmBackend):
    name = "direct"

    def __init__(self, sim: Optional[TabletopSim] = None, viewer: bool = False):
        self.sim = sim or TabletopSim()
        self.ik = IKSolver(self.sim)
        self._viewer = None
        if viewer:
            import mujoco.viewer  # noqa: WPS433 - optional GUI dependency

            self._viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)

    # ------------------------------------------------------------ interface
    def reset(self, scene: Optional[Dict] = None) -> None:
        self.sim.reset(scene)
        self._sync_viewer()

    def get_state(self) -> Dict:
        return self.sim.state()

    def move_to(self, position, yaw, linear=False, ignore_objects=()) -> MotionResult:
        t0 = time.perf_counter()
        target_quat = topdown_quat(yaw)
        q_start = self.sim.arm_q()
        if linear:
            start_pos, _ = self.sim.tcp_pose()
            dist = float(np.linalg.norm(np.asarray(position) - start_pos))
            n = max(2, int(dist / 0.01) + 1)
            waypoints = [q_start]
            q = q_start
            for s in np.linspace(0.0, 1.0, n)[1:]:
                p = start_pos + s * (np.asarray(position) - start_pos)
                q = self.ik.solve(p, target_quat, q)
                if q is None:
                    return MotionResult(False, f"IK failed along linear path at {np.round(p, 3).tolist()}",
                                        plan_time=time.perf_counter() - t0)
                waypoints.append(q)
            duration = max(0.4, 1.875 * dist / LINEAR_SPEED)
            times = np.linspace(0.0, duration, len(waypoints))
            # Smooth start/stop: sample the IK waypoints at quintic-spaced arc length.
            idx = _quintic(len(waypoints)) * (len(waypoints) - 1)
            wp = np.array(waypoints)
            positions = np.array([[np.interp(i, np.arange(len(wp)), wp[:, j]) for j in range(7)] for i in idx])
        else:
            q_goal = self._solve_with_restarts(position, target_quat, q_start)
            if q_goal is None:
                return MotionResult(False, f"No IK solution for position {np.round(position, 3).tolist()} "
                                           f"(likely outside the reachable workspace)",
                                    plan_time=time.perf_counter() - t0)
            dq = np.abs(q_goal - q_start).max()
            duration = max(0.5, 1.875 * dq / MAX_JOINT_SPEED)
            n = max(2, int(duration / 0.02))
            s = _quintic(n)
            times = np.linspace(0.0, duration, n)
            positions = q_start + s[:, None] * (q_goal - q_start)
        plan_time = time.perf_counter() - t0
        exec_time = self._execute(times, positions)
        err = float(np.linalg.norm(self.sim.tcp_pose()[0] - np.asarray(position)))
        if err > POSITION_TOL:
            return MotionResult(False, f"Tracking error {err:.3f} m after motion", plan_time, exec_time)
        return MotionResult(True, "reached", plan_time, exec_time)

    def move_home(self) -> MotionResult:
        q_start = self.sim.arm_q()
        dq = np.abs(READY_Q - q_start).max()
        duration = max(0.5, 1.875 * dq / MAX_JOINT_SPEED)
        n = max(2, int(duration / 0.02))
        positions = q_start + _quintic(n)[:, None] * (READY_Q - q_start)
        exec_time = self._execute(np.linspace(0.0, duration, n), positions)
        return MotionResult(True, "at home", 0.0, exec_time)

    def set_gripper(self, open_: bool) -> MotionResult:
        t0 = self.sim.time
        if open_:
            self.sim.open_gripper()
            held = None
        else:
            held = self.sim.close_gripper()
        self._sync_viewer()
        return MotionResult(True, "opened" if open_ else f"closed (holding {held})",
                            exec_time=self.sim.time - t0, extra={"held": held})

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()

    # -------------------------------------------------------------- helpers
    def _solve_with_restarts(self, position, target_quat, q_start):
        seeds = [q_start, READY_Q]
        rng = np.random.default_rng(0)
        for _ in range(4):
            seeds.append(np.clip(READY_Q + rng.normal(0, 0.4, 7), self.sim.arm_lower, self.sim.arm_upper))
        for seed in seeds:
            q = self.ik.solve(position, target_quat, seed)
            if q is not None:
                return q
        return None

    def _execute(self, times, positions) -> float:
        if self._viewer is None:
            return self.sim.execute_joint_trajectory(times, positions)
        # Real-time playback for the viewer.
        counter = {"n": 0}
        wall0, sim0 = time.perf_counter(), self.sim.time

        def on_step():
            counter["n"] += 1
            if counter["n"] % 10 == 0:
                self._viewer.sync()
                lag = (self.sim.time - sim0) - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)

        return self.sim.execute_joint_trajectory(times, positions, on_step=on_step)

    def _sync_viewer(self):
        if self._viewer is not None:
            self._viewer.sync()
