"""ROS 2 node that runs the MuJoCo tabletop in real time and acts as the robot driver.

Interfaces (they mirror what moveit_resources_panda_moveit_config expects):
  pub  /joint_states                                         sensor_msgs/JointState
  pub  /sim/state                                            std_msgs/String (JSON ground truth)
  act  /panda_arm_controller/follow_joint_trajectory         control_msgs/FollowJointTrajectory
  act  /panda_hand_controller/gripper_cmd                    control_msgs/GripperCommand
  srv  /sim/reset                                            std_srvs/Trigger (uses param `scene_json`)
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ..sim.tabletop import ARM_JOINTS, GRIPPER_OPEN, TabletopSim


class _Trajectory:
    def __init__(self, times, positions, t_start):
        self.times = np.asarray(times)
        self.positions = np.asarray(positions)
        self.t_start = t_start
        self.done = threading.Event()
        self.cancelled = False

    def target(self, t):
        s = t - self.t_start
        return np.array([np.interp(s, self.times, self.positions[:, j]) for j in range(7)])

    def finished(self, t, settle=0.2):
        return t - self.t_start >= self.times[-1] + settle


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__("mujoco_sim")
        self.declare_parameter("scene_json", "{}")
        self.declare_parameter("real_time_factor", 1.0)
        self.declare_parameter("state_rate_hz", 30.0)
        self.declare_parameter("viewer", False)
        self.sim = TabletopSim()
        self.traj: _Trajectory | None = None
        self.gripper_wait_until: float | None = None
        self._lock = threading.RLock()

        physics_group = MutuallyExclusiveCallbackGroup()
        action_group = ReentrantCallbackGroup()
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.state_pub = self.create_publisher(String, "/sim/state", 10)
        self.create_service(Trigger, "/sim/reset", self._on_reset, callback_group=action_group)
        ActionServer(self, FollowJointTrajectory, "/panda_arm_controller/follow_joint_trajectory",
                     execute_callback=self._exec_traj, goal_callback=lambda g: GoalResponse.ACCEPT,
                     cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=action_group)
        ActionServer(self, GripperCommand, "/panda_hand_controller/gripper_cmd",
                     execute_callback=self._exec_gripper, goal_callback=lambda g: GoalResponse.ACCEPT,
                     callback_group=action_group)

        self.viewer = None
        if self.get_parameter("viewer").value:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)

        self._wall0 = time.perf_counter()
        self._sim0 = self.sim.time
        self._last_state_pub = 0.0
        self.create_timer(0.005, self._tick, callback_group=physics_group)
        self.get_logger().info("MuJoCo sim ready (Panda tabletop).")

    # ---------------------------------------------------------------- physics
    def _tick(self):
        rtf = self.get_parameter("real_time_factor").value
        with self._lock:
            target_sim = self._sim0 + (time.perf_counter() - self._wall0) * rtf
            dt = self.sim.model.opt.timestep
            n = int((target_sim - self.sim.time) / dt)
            for _ in range(min(max(n, 0), 50)):  # cap catch-up so we never stall
                if self.traj is not None:
                    self.sim.set_arm_target(self.traj.target(self.sim.time))
                    if self.traj.finished(self.sim.time):
                        self.traj.done.set()
                        self.traj = None
                self.sim.step(1)
            if n > 50:  # fell behind; re-anchor the clock instead of fast-forwarding
                self._wall0, self._sim0 = time.perf_counter(), self.sim.time
            self._publish()
        if self.viewer is not None:
            self.viewer.sync()

    def _publish(self):
        now = self.get_clock().now().to_msg()
        js = self.sim.joint_state()
        msg = JointState()
        msg.header.stamp = now
        msg.name, msg.position, msg.velocity = js["name"], js["position"], js["velocity"]
        self.js_pub.publish(msg)
        rate = self.get_parameter("state_rate_hz").value
        if time.perf_counter() - self._last_state_pub >= 1.0 / rate:
            self._last_state_pub = time.perf_counter()
            self.state_pub.publish(String(data=json.dumps(self.sim.state())))

    # --------------------------------------------------------------- services
    def _on_reset(self, request, response):
        try:
            scene = json.loads(self.get_parameter("scene_json").value or "{}")
        except json.JSONDecodeError as e:
            response.success, response.message = False, f"bad scene_json: {e}"
            return response
        with self._lock:
            if self.traj is not None:
                self.traj.cancelled = True
                self.traj.done.set()
                self.traj = None
            self.sim.reset(scene)
            self._wall0, self._sim0 = time.perf_counter(), self.sim.time
            self._publish()
        response.success, response.message = True, "reset"
        return response

    # ---------------------------------------------------------------- actions
    def _exec_traj(self, goal_handle):
        jt = goal_handle.request.trajectory
        result = FollowJointTrajectory.Result()
        try:
            order = [list(jt.joint_names).index(j) for j in ARM_JOINTS]
        except ValueError:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            result.error_string = f"expected joints {ARM_JOINTS}, got {list(jt.joint_names)}"
            return result
        times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in jt.points]
        positions = [[p.positions[i] for i in order] for p in jt.points]
        if times and times[0] > 0.0:  # start from where the arm is now
            times = [0.0] + times
            positions = [self.sim.arm_q().tolist()] + positions
        with self._lock:
            traj = _Trajectory(times, positions, self.sim.time)
            self.traj = traj
        while not traj.done.wait(0.05):
            if goal_handle.is_cancel_requested:
                with self._lock:
                    self.traj = None
                    self.sim.set_arm_target(self.sim.arm_q())
                goal_handle.canceled()
                result.error_string = "cancelled"
                return result
        if traj.cancelled:
            goal_handle.abort()
            result.error_string = "preempted by reset"
            return result
        err = float(np.abs(self.sim.arm_q() - np.asarray(positions[-1])).max())
        if err > 0.05:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            result.error_string = f"final joint error {err:.3f} rad"
            return result
        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    def _exec_gripper(self, goal_handle):
        target = float(np.clip(goal_handle.request.command.position, 0.0, GRIPPER_OPEN))
        closing = target < self.sim.gripper_width() / 2 - 1e-4
        with self._lock:
            if not closing:
                self.sim.release()
            self.sim.command_gripper(target)
            t_end = self.sim.time + 0.6
        while self.sim.time < t_end:
            time.sleep(0.01)
        held = None
        if closing:
            with self._lock:
                held = self.sim.try_attach()
        result = GripperCommand.Result()
        result.position = self.sim.gripper_width() / 2
        result.stalled = held is not None
        result.reached_goal = held is None
        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
