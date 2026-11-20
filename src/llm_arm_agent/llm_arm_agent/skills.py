"""Robot skills exposed to the agent, implemented once on top of ArmBackend.

Every skill returns a JSON-serialisable dict with an `ok` flag and a short
message, so failures are reported back to the LLM instead of raising.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backends.base import ArmBackend, MotionResult
from .sim.tabletop import TABLE_TOP_Z

APPROACH_HEIGHT = 0.12  # clearance above grasp/place height for approach and retreat
MIN_TCP_Z = 0.012  # never send the TCP lower than this (fingertips would hit the table)
MAX_REACH = 0.82  # metres from the shoulder axis; conservative Panda reach
PLACE_CLEARANCE = 0.004  # drop height above the support surface


class SkillError(Exception):
    pass


class Skills:
    def __init__(self, backend: ArmBackend):
        self.backend = backend
        self.plan_time = 0.0
        self.exec_time = 0.0

    # ----------------------------------------------------------- primitives
    def get_scene(self) -> Dict:
        s = self.backend.get_state()
        return {
            "ok": True,
            "ee_position": s["ee_position"],
            "ee_yaw_deg": s["ee_yaw_deg"],
            "gripper_width": s["gripper_width"],
            "held_object": s["held_object"],
            "objects": {n: {k: o[k] for k in ("position", "yaw_deg", "color", "size")}
                        for n, o in s["objects"].items()},
            "zones": s["zones"],
        }

    def move_to(self, x: float, y: float, z: float, yaw_deg: Optional[float] = None) -> Dict:
        pos = (x, y, z)
        self._check_target(pos)
        yaw = self._current_yaw() if yaw_deg is None else math.radians(yaw_deg)
        res = self._run(self.backend.move_to(pos, self._closest_equivalent_yaw(yaw), linear=False))
        return self._result(res, extra={"ee_position": self.backend.get_state()["ee_position"]})

    def move_relative(self, dx: float, dy: float, dz: float) -> Dict:
        cur = np.array(self.backend.get_state()["ee_position"])
        target = cur + np.array([dx, dy, dz])
        self._check_target(target)
        res = self._run(self.backend.move_to(tuple(target), self._current_yaw(), linear=True))
        return self._result(res, extra={"ee_position": self.backend.get_state()["ee_position"]})

    def go_home(self) -> Dict:
        return self._result(self._run(self.backend.move_home()))

    def open_gripper(self) -> Dict:
        return self._result(self._run(self.backend.set_gripper(True)))

    def close_gripper(self) -> Dict:
        res = self._run(self.backend.set_gripper(False))
        held = res.extra.get("held")
        return {"ok": True, "message": res.message, "held_object": held}

    # -------------------------------------------------------------- skills
    def pick(self, object_name: str) -> Dict:
        state = self.backend.get_state()
        if state["held_object"]:
            raise SkillError(f"Already holding {state['held_object']}; place it first.")
        obj = self._get_object(state, object_name)
        x, y, z = obj["position"]
        if z < TABLE_TOP_Z - 0.05:
            raise SkillError(f"{object_name} is not on the table (z={z:.3f}).")
        self._check_reach((x, y))
        yaw = self._grasp_yaw(state, object_name)
        grasp_z = max(z, MIN_TCP_Z)
        above = (x, y, grasp_z + APPROACH_HEIGHT)

        self._step(self.backend.set_gripper(True), "open gripper")
        self._step(self.backend.move_to(above, yaw), "move above object")
        self._step(self.backend.move_to((x, y, grasp_z), yaw, linear=True, ignore_objects=[object_name]),
                   "descend to grasp")
        res = self._run(self.backend.set_gripper(False))
        held = res.extra.get("held")
        if held != object_name:
            self._run(self.backend.set_gripper(True))
            self._run(self.backend.move_to(above, yaw, linear=True))
            raise SkillError(f"Grasp failed: gripper closed on {held or 'nothing'}.")
        self._step(self.backend.move_to(above, yaw, linear=True, ignore_objects=[object_name]), "lift")
        return {"ok": True, "message": f"Picked {object_name}.", "held_object": held}

    def place_at(self, x: float, y: float) -> Dict:
        state = self.backend.get_state()
        held = state["held_object"]
        if not held:
            raise SkillError("Not holding anything; pick an object first.")
        self._check_reach((x, y))
        support_name, support_top = self._support_below(state, (x, y), exclude=held)
        half = state["objects"][held]["size"] / 2
        place_z = support_top + half + PLACE_CLEARANCE
        if support_name:  # centre on the supporting object for a clean stack
            sx, sy, _ = state["objects"][support_name]["position"]
            x, y = sx, sy
        yaw = self._closest_equivalent_yaw(self._current_yaw())
        above = (x, y, place_z + APPROACH_HEIGHT)
        ignore = [held] + ([support_name] if support_name else [])

        self._step(self.backend.move_to(above, yaw, ignore_objects=[held]), "move above place location")
        self._step(self.backend.move_to((x, y, place_z), yaw, linear=True, ignore_objects=ignore), "descend")
        self._step(self.backend.set_gripper(True), "release")
        self._step(self.backend.move_to(above, yaw, linear=True, ignore_objects=ignore), "retreat")
        after = self.backend.get_state()["objects"][held]["position"]
        where = f"on {support_name}" if support_name else "on the table"
        return {"ok": True, "message": f"Placed {held} {where}.", "object_position": after}

    def place_on(self, target: str) -> Dict:
        state = self.backend.get_state()
        if target in state["zones"]:
            cx, cy = state["zones"][target]["center"]
            return self.place_at(cx, cy)
        if target in state["objects"]:
            if target == state["held_object"]:
                raise SkillError("Cannot place an object on itself.")
            tx, ty, _ = state["objects"][target]["position"]
            return self.place_at(tx, ty)
        raise SkillError(f"Unknown target '{target}'. Zones: {sorted(state['zones'])}; "
                         f"objects: {sorted(state['objects'])}.")

    # -------------------------------------------------------------- helpers
    def _run(self, res: MotionResult) -> MotionResult:
        self.plan_time += res.plan_time
        self.exec_time += res.exec_time
        return res

    def _step(self, res: MotionResult, what: str) -> MotionResult:
        self._run(res)
        if not res.ok:
            raise SkillError(f"{what} failed: {res.message}")
        return res

    @staticmethod
    def _result(res: MotionResult, extra: Optional[Dict] = None) -> Dict:
        out = {"ok": res.ok, "message": res.message}
        if extra:
            out.update(extra)
        return out

    @staticmethod
    def _get_object(state: Dict, name: str) -> Dict:
        if name not in state["objects"]:
            raise SkillError(f"Unknown object '{name}'. Known objects: {sorted(state['objects'])}.")
        return state["objects"][name]

    def _current_yaw(self) -> float:
        return math.radians(self.backend.get_state()["ee_yaw_deg"])

    def _closest_equivalent_yaw(self, yaw: float, symmetry: float = math.pi) -> float:
        """Pick the grasp yaw closest to 0 among symmetric equivalents.

        The parallel gripper is symmetric under 180 deg (cubes under 90 deg), so we
        choose the equivalent that keeps joint 7 near the middle of its range.
        """
        k = round(yaw / symmetry)
        return yaw - k * symmetry

    def _grasp_yaw(self, state: Dict, name: str) -> float:
        """Choose between the two cube-aligned grasps the one whose fingers open away from neighbours.

        The open fingers span ~8 cm along the hand y-axis, which for a top-down hand with
        yaw psi points along (sin psi, -cos psi) in the table plane. A neighbour lying
        along that axis would be hit by a finger on the way down.
        """
        obj = state["objects"][name]
        base = self._closest_equivalent_yaw(math.radians(obj["yaw_deg"]), symmetry=math.pi / 2)
        candidates = [base, self._closest_equivalent_yaw(base + math.pi / 2)]
        ox, oy, _ = obj["position"]

        def penalty(yaw: float) -> float:
            ax, ay = math.sin(yaw), -math.cos(yaw)
            total = 0.0
            for other, o in state["objects"].items():
                if other == name:
                    continue
                dx, dy = o["position"][0] - ox, o["position"][1] - oy
                dist = math.hypot(dx, dy)
                if dist < 0.12:
                    total += abs(dx * ax + dy * ay) / dist * (0.12 - dist)
            return total

        return min(candidates, key=lambda y: (round(penalty(y), 6), abs(y)))

    @staticmethod
    def _check_reach(xy: Tuple[float, float]) -> None:
        r = math.hypot(*xy)
        if r > MAX_REACH:
            raise SkillError(f"Target ({xy[0]:.2f}, {xy[1]:.2f}) is {r:.2f} m from the base, beyond "
                             f"the reachable radius of ~{MAX_REACH} m.")
        if r < 0.25:
            raise SkillError(f"Target ({xy[0]:.2f}, {xy[1]:.2f}) is too close to the robot base.")

    def _check_target(self, pos) -> None:
        x, y, z = pos
        if z < MIN_TCP_Z:
            raise SkillError(f"z={z:.3f} would drive the gripper into the table (min {MIN_TCP_Z}).")
        if math.dist((0, 0, 0.333), (x, y, z)) > MAX_REACH + 0.05:
            raise SkillError(f"Position ({x:.2f}, {y:.2f}, {z:.2f}) is outside the reachable workspace.")

    @staticmethod
    def _support_below(state: Dict, xy, exclude: str) -> Tuple[Optional[str], float]:
        """Highest object whose footprint contains xy, else the table."""
        best_name, best_top = None, TABLE_TOP_Z
        for name, o in state["objects"].items():
            if name == exclude:
                continue
            ox, oy, oz = o["position"]
            half = o["size"] / 2
            if abs(ox - xy[0]) < half + 0.01 and abs(oy - xy[1]) < half + 0.01:
                top = oz + half
                if top > best_top:
                    best_name, best_top = name, top
        return best_name, best_top


def tool_names() -> List[str]:
    return ["get_scene", "move_to", "move_relative", "go_home", "open_gripper", "close_gripper",
            "pick", "place_at", "place_on"]
