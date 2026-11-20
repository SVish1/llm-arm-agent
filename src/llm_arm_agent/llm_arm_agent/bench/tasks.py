"""Benchmark task suite: natural-language instructions with ground-truth success checks.

Each task defines
  * scene      - initial object poses (and optionally the arm configuration)
  * instruction- what the user types
  * check      - predicate on (initial state, final state, episode) -> (success, reason)
  * oracle     - a scripted tool sequence that solves the task; used to verify
                 that every task is solvable by the skill layer and to give an
                 LLM-free upper bound on success rate and a lower bound on latency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from ..sim.tabletop import TABLE_X, TABLE_Y, ZONES

ToolCall = Tuple[str, Dict]

HOME_TCP = (0.307, 0.0, 0.487)
ON_TABLE_Z = 0.02
Z_TOL = 0.008
XY_TOL = 0.025


# --------------------------------------------------------------------- helpers
def P(state: Dict, name: str) -> List[float]:
    return state["objects"][name]["position"]


def xy_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def on_table(state, name) -> bool:
    return abs(P(state, name)[2] - ON_TABLE_Z) < Z_TOL and state["held_object"] != name


def in_zone(state, name, zone) -> bool:
    z = ZONES[zone]
    return xy_dist(P(state, name), z["xy"]) <= z["radius"] and state["held_object"] != name


def stacked_on(state, top, bottom) -> bool:
    t, b = P(state, top), P(state, bottom)
    return (xy_dist(t, b) < 0.02 and abs((t[2] - b[2]) - 0.04) < Z_TOL
            and state["held_object"] != top)


def moved(init, final, name, tol=0.01) -> bool:
    a, b = P(init, name), P(final, name)
    return math.dist(a, b) > tol


def on_tabletop(state, name) -> bool:
    x, y, z = P(state, name)
    return TABLE_X[0] < x < TABLE_X[1] and TABLE_Y[0] < y < TABLE_Y[1] and z > 0.0


def result(ok: bool, why_not: str) -> Tuple[bool, str]:
    return (True, "ok") if ok else (False, why_not)


@dataclass
class Task:
    id: str
    category: str
    instruction: str
    scene: Dict
    check: Callable[[Dict, Dict, object], Tuple[bool, str]]
    oracle: Callable[[Dict], List[ToolCall]]
    feasible: bool = True  # False: the correct behaviour is to decline (finish success=false)


# ----------------------------------------------------------------------- tasks
def _t01():
    return Task(
        "T01_home", "motion", "Move the arm back to its home position.",
        {"arm_q": [0.5, -0.2, 0.2, -1.9, 0.1, 1.9, 0.9],
         "objects": {"red_cube": [0.55, 0.05, 0]}},
        lambda i, f, e: result(math.dist(f["ee_position"], HOME_TCP) < 0.02,
                               f"gripper {f['ee_position']} not at home {HOME_TCP}"),
        lambda s: [("go_home", {})],
    )


def _t02():
    def check(i, f, e):
        dz = f["ee_position"][2] - i["ee_position"][2]
        drift = xy_dist(f["ee_position"], i["ee_position"])
        return result(0.08 <= dz <= 0.12 and drift < 0.02, f"dz={dz:.3f}, xy drift={drift:.3f}")

    return Task("T02_raise", "motion", "Raise the gripper by 10 centimetres.",
                {"objects": {"red_cube": [0.55, 0.05, 0]}}, check,
                lambda s: [("move_relative", {"dx": 0, "dy": 0, "dz": 0.10})])


def _t03():
    def check(i, f, e):
        r = P(f, "red_cube")
        ee = f["ee_position"]
        h = ee[2] - r[2]
        return result(xy_dist(ee, r) < 0.015 and 0.07 <= h <= 0.15 and not moved(i, f, "red_cube"),
                      f"ee {ee} vs red {r} (height {h:.3f})")

    return Task("T03_hover", "motion", "Hover the gripper about 10 cm above the red cube.",
                {"objects": {"red_cube": [0.5, 0.12, 15], "blue_cube": [0.6, -0.1, 0]}}, check,
                lambda s: [("move_to", {"x": P(s, "red_cube")[0], "y": P(s, "red_cube")[1],
                                        "z": P(s, "red_cube")[2] + 0.10, "yaw_deg": 0})])


def _t04():
    return Task(
        "T04_pick", "pick-place", "Pick up the red cube.",
        {"objects": {"red_cube": [0.5, 0.05, 30], "green_cube": [0.45, -0.15, 0], "blue_cube": [0.62, -0.05, 0]}},
        lambda i, f, e: result(f["held_object"] == "red_cube" and P(f, "red_cube")[2] > 0.08,
                               f"held={f['held_object']}, red z={P(f, 'red_cube')[2]}"),
        lambda s: [("pick", {"object_name": "red_cube"})],
    )


def _t05():
    return Task(
        "T05_to_zone", "pick-place", "Put the blue cube on the right pad.",
        {"objects": {"blue_cube": [0.55, 0.1, 10], "red_cube": [0.4, 0.0, 0], "green_cube": [0.6, -0.08, 0]}},
        lambda i, f, e: result(in_zone(f, "blue_cube", "right_pad") and on_table(f, "blue_cube"),
                               f"blue at {P(f, 'blue_cube')}"),
        lambda s: [("pick", {"object_name": "blue_cube"}), ("place_on", {"target": "right_pad"})],
    )


def _t06():
    return Task(
        "T06_stack", "pick-place", "Stack the red cube on top of the green cube.",
        {"objects": {"red_cube": [0.55, 0.12, 0], "green_cube": [0.45, -0.1, 25], "yellow_cube": [0.62, -0.15, 0]}},
        lambda i, f, e: result(stacked_on(f, "red_cube", "green_cube"),
                               f"red {P(f, 'red_cube')} green {P(f, 'green_cube')}"),
        lambda s: [("pick", {"object_name": "red_cube"}), ("place_on", {"target": "green_cube"})],
    )


def _t07():
    def check(i, f, e):
        b, y = P(f, "blue_cube"), P(f, "yellow_cube")
        target = (b[0], b[1] + 0.10)
        return result(xy_dist(y, target) < XY_TOL and on_table(f, "yellow_cube") and not moved(i, f, "blue_cube"),
                      f"yellow {y} vs target {target}")

    return Task(
        "T07_relative", "spatial", "Place the yellow cube 10 cm to the left of the blue cube.",
        {"objects": {"blue_cube": [0.5, -0.1, 0], "yellow_cube": [0.4, 0.18, 0], "red_cube": [0.62, 0.1, 0]}},
        check,
        lambda s: [("pick", {"object_name": "yellow_cube"}),
                   ("place_at", {"x": P(s, "blue_cube")[0], "y": P(s, "blue_cube")[1] + 0.10})],
    )


def _t08():
    cubes = ["red_cube", "green_cube", "blue_cube", "yellow_cube"]

    def closest(state):
        return min(cubes, key=lambda n: math.hypot(*P(state, n)[:2]))

    def check(i, f, e):
        c = closest(i)
        others_still = not any(moved(i, f, n) for n in cubes if n != c)
        return result(in_zone(f, c, "left_pad") and on_table(f, c) and others_still,
                      f"closest was {c}; at {P(f, c)}; others still={others_still}")

    return Task(
        "T08_closest", "spatial", "Put the cube that is closest to the robot onto the left pad.",
        {"objects": {"red_cube": [0.6, 0.1, 0], "green_cube": [0.38, -0.12, 0],
                     "blue_cube": [0.55, -0.2, 0], "yellow_cube": [0.5, 0.04, 0]}},
        check,
        lambda s: [("pick", {"object_name": closest(s)}), ("place_on", {"target": "left_pad"})],
    )


def _t09():
    return Task(
        "T09_sort", "multi-step", "Put the red cube on the left pad and the blue cube on the right pad.",
        {"objects": {"red_cube": [0.55, -0.1, 0], "blue_cube": [0.5, 0.1, 0], "green_cube": [0.62, 0.0, 0]}},
        lambda i, f, e: result(in_zone(f, "red_cube", "left_pad") and in_zone(f, "blue_cube", "right_pad")
                               and on_table(f, "red_cube") and on_table(f, "blue_cube"),
                               f"red {P(f, 'red_cube')} blue {P(f, 'blue_cube')}"),
        lambda s: [("pick", {"object_name": "red_cube"}), ("place_on", {"target": "left_pad"}),
                   ("pick", {"object_name": "blue_cube"}), ("place_on", {"target": "right_pad"})],
    )


def _t10():
    def check(i, f, e):
        ok = (in_zone(f, "blue_cube", "left_pad") and on_table(f, "blue_cube")
              and stacked_on(f, "green_cube", "blue_cube") and stacked_on(f, "red_cube", "green_cube"))
        return result(ok, f"blue {P(f, 'blue_cube')} green {P(f, 'green_cube')} red {P(f, 'red_cube')}")

    return Task(
        "T10_tower", "multi-step",
        "Build a tower on the left pad: blue cube at the bottom, then the green cube, then the red cube on top.",
        {"objects": {"blue_cube": [0.55, 0.0, 0], "green_cube": [0.5, -0.15, 20], "red_cube": [0.4, -0.05, 0]}},
        check,
        lambda s: [("pick", {"object_name": "blue_cube"}), ("place_on", {"target": "left_pad"}),
                   ("pick", {"object_name": "green_cube"}), ("place_on", {"target": "blue_cube"}),
                   ("pick", {"object_name": "red_cube"}), ("place_on", {"target": "green_cube"})],
    )


def _t11():
    def check(i, f, e):
        ok = (xy_dist(P(f, "red_cube"), P(i, "green_cube")) < XY_TOL
              and xy_dist(P(f, "green_cube"), P(i, "red_cube")) < XY_TOL
              and on_table(f, "red_cube") and on_table(f, "green_cube"))
        return result(ok, f"red {P(f, 'red_cube')} green {P(f, 'green_cube')}")

    def oracle(s):
        r, g = P(s, "red_cube"), P(s, "green_cube")
        return [("pick", {"object_name": "red_cube"}), ("place_at", {"x": 0.4, "y": 0.0}),
                ("pick", {"object_name": "green_cube"}), ("place_at", {"x": r[0], "y": r[1]}),
                ("pick", {"object_name": "red_cube"}), ("place_at", {"x": g[0], "y": g[1]})]

    return Task("T11_swap", "multi-step", "Swap the positions of the red cube and the green cube.",
                {"objects": {"red_cube": [0.52, 0.12, 0], "green_cube": [0.52, -0.12, 0]}}, check, oracle)


def _t12():
    def check(i, f, e):
        names = ["red_cube", "yellow_cube"]
        cleared = not any(in_zone(f, n, "left_pad") for n in ["red_cube", "green_cube", "blue_cube", "yellow_cube"])
        landed = all(xy_dist(P(f, n), ZONES["right_pad"]["xy"]) <= ZONES["right_pad"]["radius"] + 0.01
                     and f["held_object"] != n for n in names)
        return result(cleared and landed, f"red {P(f, 'red_cube')} yellow {P(f, 'yellow_cube')}")

    return Task(
        "T12_clear_zone", "multi-step",
        "Clear the left pad by moving everything on it onto the right pad. Stacking is fine.",
        {"objects": {"red_cube": [0.45, 0.28, 0], "yellow_cube": [0.45, 0.215, 0], "blue_cube": [0.6, 0.0, 0]}},
        check,
        lambda s: [("pick", {"object_name": "red_cube"}), ("place_on", {"target": "right_pad"}),
                   ("pick", {"object_name": "yellow_cube"}), ("place_on", {"target": "red_cube"})],
    )


def _t13():
    return Task(
        "T13_lift_hold", "pick-place", "Lift the green cube to about 20 cm above the table and keep holding it.",
        {"objects": {"green_cube": [0.5, -0.05, 0], "red_cube": [0.45, 0.15, 0]}},
        lambda i, f, e: result(f["held_object"] == "green_cube" and 0.16 <= P(f, "green_cube")[2] <= 0.24,
                               f"held={f['held_object']} green z={P(f, 'green_cube')[2]}"),
        lambda s: [("pick", {"object_name": "green_cube"}), ("move_relative", {"dx": 0, "dy": 0, "dz": 0.06})],
    )


def _t14():
    cubes = ["red_cube", "green_cube", "blue_cube", "yellow_cube"]

    def check(i, f, e):
        still = not any(moved(i, f, n) for n in cubes)
        return result(still and e.claimed_success is False and f["held_object"] is None,
                      f"objects still={still}, claimed_success={e.claimed_success}")

    return Task(
        "T14_missing_object", "robustness", "Pick up the purple cube.",
        {"objects": {"red_cube": [0.5, 0.1, 0], "blue_cube": [0.5, -0.1, 0], "green_cube": [0.6, 0.0, 0]}},
        check,
        lambda s: [("get_scene", {}),
                   ("finish", {"success": False, "summary": "There is no purple cube in the scene."})],
        feasible=False,
    )


def _t15():
    cubes = ["red_cube", "green_cube", "blue_cube", "yellow_cube"]

    def check(i, f, e):
        safe = all(on_tabletop(f, n) for n in cubes) and f["held_object"] is None
        return result(safe and e.claimed_success is False,
                      f"all on table & released={safe}, claimed_success={e.claimed_success}")

    return Task(
        "T15_unreachable", "robustness", "Put the red cube at x = 1.5 m, y = 0.",
        {"objects": {"red_cube": [0.5, 0.05, 0], "green_cube": [0.45, -0.15, 0]}},
        check,
        lambda s: [("get_scene", {}),
                   ("finish", {"success": False, "summary": "x = 1.5 m is beyond the arm's reach."})],
        feasible=False,
    )


TASKS: List[Task] = [f() for f in (_t01, _t02, _t03, _t04, _t05, _t06, _t07, _t08, _t09, _t10,
                                   _t11, _t12, _t13, _t14, _t15)]
TASKS_BY_ID = {t.id: t for t in TASKS}
