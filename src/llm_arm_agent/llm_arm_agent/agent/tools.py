"""Tool schemas exposed to the LLM and the dispatcher that runs them on Skills."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple

from ..skills import SkillError, Skills


def _obj(props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_NUM = {"type": "number"}

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_scene",
        "description": "Return the current end-effector position, gripper state, the object the gripper "
                       "holds (if any), every object's position/colour/size, and the named target zones. "
                       "Call this before acting and again to verify results.",
        "input_schema": _obj({}),
    },
    {
        "name": "move_to",
        "description": "Move the gripper tool-centre point to an absolute position (metres, robot base frame) "
                       "with the gripper pointing straight down. Collision-aware when the MoveIt backend is "
                       "used. Use for free-space moves such as hovering; prefer pick/place for manipulation.",
        "input_schema": _obj({
            "x": {**_NUM, "description": "Forward distance from the robot base (m)."},
            "y": {**_NUM, "description": "Leftward distance from the robot base (m); negative is right."},
            "z": {**_NUM, "description": "Height above the table top (m)."},
            "yaw_deg": {**_NUM, "description": "Gripper rotation about vertical axis in degrees; 0 is the default."},
        }),
    },
    {
        "name": "move_relative",
        "description": "Move the gripper in a straight line by an offset from its current position (metres).",
        "input_schema": _obj({"dx": _NUM, "dy": _NUM, "dz": _NUM}),
    },
    {
        "name": "go_home",
        "description": "Move the arm to its home (ready) configuration above the table.",
        "input_schema": _obj({}),
    },
    {
        "name": "open_gripper",
        "description": "Open the gripper, releasing anything it holds.",
        "input_schema": _obj({}),
    },
    {
        "name": "close_gripper",
        "description": "Close the gripper; reports which object, if any, ended up held.",
        "input_schema": _obj({}),
    },
    {
        "name": "pick",
        "description": "Pick up an object by name: opens the gripper, approaches from above, grasps, and "
                       "lifts it 12 cm. Fails if something is already held.",
        "input_schema": _obj({"object_name": {"type": "string", "description": "Exact object name from get_scene."}}),
    },
    {
        "name": "place_at",
        "description": "Place the held object at table coordinates (x, y). If another object is under that "
                       "point the held object is stacked on top of it. Releases and retreats upward.",
        "input_schema": _obj({"x": _NUM, "y": _NUM}),
    },
    {
        "name": "place_on",
        "description": "Place the held object on a named zone (e.g. 'left_pad') or on top of a named object.",
        "input_schema": _obj({"target": {"type": "string", "description": "Zone or object name from get_scene."}}),
    },
    {
        "name": "finish",
        "description": "End the task. Call exactly once when the instruction is complete, or when it cannot "
                       "be completed (unknown object, unreachable target, ambiguous request).",
        "input_schema": _obj({
            "success": {"type": "boolean", "description": "True only if the instruction was fully carried out."},
            "summary": {"type": "string", "description": "One or two sentences on what was done or why not."},
        }),
    },
]

for _t in TOOLS:
    _t["strict"] = True

TOOL_NAMES = [t["name"] for t in TOOLS]


class ToolExecutor:
    """Runs tool calls against Skills, timing each and catching skill errors."""

    def __init__(self, skills: Skills):
        self.skills = skills
        self.finished = False
        self.claimed_success = None
        self.summary = ""

    def __call__(self, name: str, args: Dict[str, Any]) -> Tuple[str, bool, float]:
        """Returns (json_result, is_error, seconds)."""
        t0 = time.perf_counter()
        try:
            result = self._dispatch(name, args)
            is_error = not result.get("ok", True)
        except SkillError as e:
            result, is_error = {"ok": False, "error": str(e)}, True
        except (TypeError, KeyError, ValueError) as e:
            result, is_error = {"ok": False, "error": f"Bad arguments for {name}: {e}"}, True
        return json.dumps(result), is_error, time.perf_counter() - t0

    def _dispatch(self, name: str, a: Dict[str, Any]) -> Dict[str, Any]:
        s = self.skills
        if name == "finish":
            self.finished = True
            self.claimed_success = bool(a["success"])
            self.summary = str(a["summary"])
            return {"ok": True, "message": "Episode finished."}
        if self.finished:
            return {"ok": False, "error": "Episode already finished."}
        if name == "get_scene":
            return s.get_scene()
        if name == "move_to":
            return s.move_to(float(a["x"]), float(a["y"]), float(a["z"]), float(a["yaw_deg"]))
        if name == "move_relative":
            return s.move_relative(float(a["dx"]), float(a["dy"]), float(a["dz"]))
        if name == "go_home":
            return s.go_home()
        if name == "open_gripper":
            return s.open_gripper()
        if name == "close_gripper":
            return s.close_gripper()
        if name == "pick":
            return s.pick(str(a["object_name"]))
        if name == "place_at":
            return s.place_at(float(a["x"]), float(a["y"]))
        if name == "place_on":
            return s.place_on(str(a["target"]))
        raise SkillError(f"Unknown tool '{name}'. Available: {TOOL_NAMES}")
