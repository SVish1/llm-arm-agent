"""ROS 2 / MoveIt 2 backend.

Planning goes through move_group:
  * free-space moves   -> MoveGroup action (/move_action), OMPL, pose goal on panda_hand
  * approach / retreat -> /compute_cartesian_path + ExecuteTrajectory action
  * home               -> MoveGroup action with joint constraints
Execution is done by move_group's controller manager, which forwards
trajectories to the MuJoCo sim node's FollowJointTrajectory server.
The gripper is commanded directly through control_msgs/GripperCommand.

Before each motion the planning scene is synchronised from simulator ground
truth: the table and cubes become collision objects, a held cube is attached
to the hand, and objects the skill expects to touch are left out.
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Dict, Optional, Sequence

import rclpy
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject, Constraints, JointConstraint,
                             MotionPlanRequest, MoveItErrorCodes, OrientationConstraint, PlanningOptions,
                             PlanningScene, PositionConstraint)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ..sim.tabletop import ARM_JOINTS, GRIPPER_OPEN, OBJECTS, READY_Q, topdown_quat
from .base import ArmBackend, MotionResult

FRAME = "world"
GROUP = "panda_arm"
EE_LINK = "panda_hand"
HAND_TO_TCP = 0.1034  # panda_hand origin -> fingertip centre along hand z
TOUCH_LINKS = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
ERROR_NAMES = {v: k for k, v in vars(MoveItErrorCodes).items() if k.isupper() and isinstance(v, int)}


class MoveItBackend(ArmBackend):
    name = "moveit"

    def __init__(self, planning_time: float = 5.0, velocity_scaling: float = 0.5,
                 server_timeout: float = 30.0):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("llm_arm_agent")
        cb = ReentrantCallbackGroup()
        n = self.node
        self.move_client = ActionClient(n, MoveGroup, "/move_action", callback_group=cb)
        self.exec_client = ActionClient(n, ExecuteTrajectory, "/execute_trajectory", callback_group=cb)
        self.grip_client = ActionClient(n, GripperCommand, "/panda_hand_controller/gripper_cmd", callback_group=cb)
        self.cartesian_srv = n.create_client(GetCartesianPath, "/compute_cartesian_path", callback_group=cb)
        self.scene_srv = n.create_client(ApplyPlanningScene, "/apply_planning_scene", callback_group=cb)
        self.reset_srv = n.create_client(Trigger, "/sim/reset", callback_group=cb)
        self.param_srv = n.create_client(SetParameters, "/mujoco_sim/set_parameters", callback_group=cb)
        self._state: Optional[Dict] = None
        self._state_seq = 0
        self._state_cv = threading.Condition()
        n.create_subscription(String, "/sim/state", self._on_state, 10, callback_group=cb)
        self.planning_time = planning_time
        self.velocity_scaling = velocity_scaling
        self._attached: Optional[str] = None
        self._in_world: set = set()  # collision objects currently in move_group's world

        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(n)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        self._wait_for_servers(server_timeout)

    # -------------------------------------------------------------- plumbing
    def _wait_for_servers(self, timeout: float) -> None:
        checks = [
            ("/move_action", lambda t: self.move_client.wait_for_server(timeout_sec=t)),
            ("/execute_trajectory", lambda t: self.exec_client.wait_for_server(timeout_sec=t)),
            ("/panda_hand_controller/gripper_cmd", lambda t: self.grip_client.wait_for_server(timeout_sec=t)),
            ("/compute_cartesian_path", lambda t: self.cartesian_srv.wait_for_service(timeout_sec=t)),
            ("/apply_planning_scene", lambda t: self.scene_srv.wait_for_service(timeout_sec=t)),
            ("/sim/reset", lambda t: self.reset_srv.wait_for_service(timeout_sec=t)),
            ("/mujoco_sim/set_parameters", lambda t: self.param_srv.wait_for_service(timeout_sec=t)),
        ]
        for name, wait in checks:
            if not wait(timeout):
                raise RuntimeError(f"{name} not available. Is `ros2 launch llm_arm_agent "
                                   f"sim_moveit.launch.py` running?")
        self._wait_state(timeout)

    @staticmethod
    def _await(future, timeout: float):
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            raise TimeoutError("ROS call timed out")
        return future.result()

    def _send_goal(self, client, goal, timeout: float = 120.0):
        handle = self._await(client.send_goal_async(goal), 10.0)
        if not handle.accepted:
            raise RuntimeError("goal rejected")
        return self._await(handle.get_result_async(), timeout).result

    def _on_state(self, msg: String) -> None:
        with self._state_cv:
            self._state = json.loads(msg.data)
            self._state_seq += 1
            self._state_cv.notify_all()

    def _wait_state(self, timeout: float = 5.0, fresh: int = 0) -> Dict:
        """Return the latest state, optionally waiting for `fresh` new messages."""
        with self._state_cv:
            target = self._state_seq + fresh
            ok = self._state_cv.wait_for(lambda: self._state is not None and self._state_seq >= target, timeout)
            if not ok:
                raise TimeoutError("no /sim/state received")
            return self._state

    # ------------------------------------------------------------- interface
    def reset(self, scene: Optional[Dict] = None) -> None:
        req = SetParameters.Request()
        req.parameters = [Parameter(name="scene_json", value=ParameterValue(
            type=ParameterType.PARAMETER_STRING, string_value=json.dumps(scene or {})))]
        self._await(self.param_srv.call_async(req), 5.0)
        res = self._await(self.reset_srv.call_async(Trigger.Request()), 10.0)
        if not res.success:
            raise RuntimeError(f"sim reset failed: {res.message}")
        self._wait_state(fresh=2)
        self._sync_scene(ignore=())

    def get_state(self) -> Dict:
        return self._wait_state(fresh=1)

    def move_to(self, position, yaw, linear=False, ignore_objects=()) -> MotionResult:
        t0 = time.perf_counter()
        self._sync_scene(ignore=ignore_objects)
        pose = self._hand_pose(position, yaw)
        if linear:
            res = self._cartesian(pose, t0)
            if res.ok:
                return res
            # Fall back to a sampled plan if the straight line is not fully feasible.
        return self._plan_and_execute(self._pose_goal(pose), t0)

    def move_home(self) -> MotionResult:
        t0 = time.perf_counter()
        self._sync_scene(ignore=())
        c = Constraints()
        for name, q in zip(ARM_JOINTS, READY_Q):
            c.joint_constraints.append(JointConstraint(joint_name=name, position=float(q),
                                                       tolerance_above=0.01, tolerance_below=0.01, weight=1.0))
        return self._plan_and_execute(c, t0)

    def set_gripper(self, open_: bool) -> MotionResult:
        t0 = time.perf_counter()
        goal = GripperCommand.Goal()
        goal.command.position = GRIPPER_OPEN if open_ else 0.0
        goal.command.max_effort = 40.0
        self._send_goal(self.grip_client, goal, timeout=10.0)
        state = self._wait_state(fresh=2)
        held = state["held_object"]
        msg = "opened" if open_ else f"closed (holding {held})"
        return MotionResult(True, msg, exec_time=time.perf_counter() - t0, extra={"held": held})

    def close(self) -> None:
        self._executor.shutdown()
        self.node.destroy_node()

    # ------------------------------------------------------------- planning
    @staticmethod
    def _hand_pose(tcp_position: Sequence[float], yaw: float) -> Pose:
        w, x, y, z = topdown_quat(yaw)
        p = Pose()
        # Gripper points down, so the hand frame sits HAND_TO_TCP above the TCP.
        p.position.x, p.position.y = float(tcp_position[0]), float(tcp_position[1])
        p.position.z = float(tcp_position[2]) + HAND_TO_TCP
        p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = map(float, (w, x, y, z))
        return p

    @staticmethod
    def _pose_goal(pose: Pose) -> Constraints:
        c = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = FRAME
        pc.link_name = EE_LINK
        sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.002])
        pc.constraint_region.primitives.append(sphere)
        center = Pose()
        center.position = pose.position
        center.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(center)
        pc.weight = 1.0
        oc = OrientationConstraint()
        oc.header.frame_id = FRAME
        oc.link_name = EE_LINK
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 0.02
        oc.weight = 1.0
        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        return c

    def _plan_and_execute(self, goal_constraints: Constraints, t0: float) -> MotionResult:
        req = MotionPlanRequest()
        req.group_name = GROUP
        req.pipeline_id = "ompl"
        req.num_planning_attempts = 4
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.velocity_scaling
        req.start_state.is_diff = True
        ws = req.workspace_parameters
        ws.header.frame_id = FRAME
        ws.min_corner.x, ws.min_corner.y, ws.min_corner.z = -1.0, -1.0, -0.1
        ws.max_corner.x, ws.max_corner.y, ws.max_corner.z = 1.0, 1.0, 1.5
        req.goal_constraints.append(goal_constraints)
        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        result = self._send_goal(self.move_client, goal)
        total = time.perf_counter() - t0
        plan_t = float(result.planning_time)
        code = result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return MotionResult(False, f"MoveIt failed: {ERROR_NAMES.get(code, code)}", plan_t, total - plan_t)
        self._wait_state(fresh=2)
        return MotionResult(True, "reached", plan_t, total - plan_t)

    def _cartesian(self, pose: Pose, t0: float) -> MotionResult:
        req = GetCartesianPath.Request()
        req.header.frame_id = FRAME
        req.start_state.is_diff = True
        req.group_name = GROUP
        req.link_name = EE_LINK
        req.waypoints = [pose]
        req.max_step = 0.005
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        if hasattr(req, "max_velocity_scaling_factor"):
            req.max_velocity_scaling_factor = self.velocity_scaling * 0.5
            req.max_acceleration_scaling_factor = self.velocity_scaling * 0.5
        res = self._await(self.cartesian_srv.call_async(req), 10.0)
        plan_t = time.perf_counter() - t0
        if res.error_code.val != MoveItErrorCodes.SUCCESS or res.fraction < 0.99:
            return MotionResult(False, f"Cartesian path {res.fraction * 100:.0f}% feasible", plan_t)
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        result = self._send_goal(self.exec_client, goal)
        total = time.perf_counter() - t0
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return MotionResult(False, f"execution failed: {ERROR_NAMES.get(result.error_code.val)}",
                                plan_t, total - plan_t)
        self._wait_state(fresh=2)
        return MotionResult(True, "reached", plan_t, total - plan_t)

    # --------------------------------------------------------- planning scene
    def _sync_scene(self, ignore: Sequence[str]) -> None:
        state = self._wait_state()
        held = state["held_object"]
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True

        table = CollisionObject()
        table.header.frame_id = FRAME
        table.id = "table"
        table.operation = CollisionObject.ADD
        table.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.1, 1.2, 0.05]))
        tp = Pose()
        tp.position.x, tp.position.z = 0.4, -0.027  # top 2 mm below z=0 so the base does not touch it
        tp.orientation.w = 1.0
        table.primitive_poses.append(tp)
        scene.world.collision_objects.append(table)

        # Detach a previously held object that has since been released.
        if self._attached and self._attached != held:
            aco = AttachedCollisionObject()
            aco.link_name = EE_LINK
            aco.object.id = self._attached
            aco.object.operation = CollisionObject.REMOVE
            scene.robot_state.attached_collision_objects.append(aco)
            self._attached = None

        for name in OBJECTS:
            obj = state["objects"][name]
            co = CollisionObject()
            co.header.frame_id = FRAME
            co.id = name
            if name == held:
                if self._attached != name:
                    aco = AttachedCollisionObject()
                    aco.link_name = EE_LINK
                    aco.touch_links = TOUCH_LINKS
                    aco.object = self._box(name, obj)
                    scene.robot_state.attached_collision_objects.append(aco)
                    self._attached = name
                    self._in_world.discard(name)  # attaching moves it out of the world
                continue
            if name in ignore:
                if name in self._in_world:  # removing a non-existent object rejects the whole diff
                    co.operation = CollisionObject.REMOVE
                    scene.world.collision_objects.append(co)
                    self._in_world.discard(name)
                continue
            scene.world.collision_objects.append(self._box(name, obj))
            self._in_world.add(name)
        res = self._await(self.scene_srv.call_async(ApplyPlanningScene.Request(scene=scene)), 5.0)
        if not res.success:
            self.node.get_logger().warn("planning scene update was not applied")

    @staticmethod
    def _box(name: str, obj: Dict) -> CollisionObject:
        co = CollisionObject()
        co.header.frame_id = FRAME
        co.id = name
        co.operation = CollisionObject.ADD
        s = obj["size"]
        co.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[s, s, s]))
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, obj["position"])
        half = math.radians(obj["yaw_deg"]) / 2
        p.orientation.w, p.orientation.z = math.cos(half), math.sin(half)
        co.primitive_poses.append(p)
        return co
