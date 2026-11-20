#!/usr/bin/env python3
"""Render a benchmark episode (standalone backend) to an animated GIF.

    python3 scripts/record_gif.py --task T10_tower --out docs/tower.gif                 # scripted oracle
    python3 scripts/record_gif.py --task T10_tower --planner llm --out docs/tower.gif  # LLM agent

Needs an OpenGL context: set MUJOCO_GL=egl or osmesa on a headless machine.
"""

import argparse

import mujoco
from PIL import Image

from llm_arm_agent.agent.tools import ToolExecutor
from llm_arm_agent.backends.direct import DirectBackend
from llm_arm_agent.bench.tasks import TASKS_BY_ID
from llm_arm_agent.skills import Skills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T10_tower", choices=sorted(TASKS_BY_ID))
    ap.add_argument("--planner", choices=["oracle", "llm"], default="oracle")
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--speedup", type=float, default=3.0, help="playback speed relative to sim time")
    ap.add_argument("--width", type=int, default=440)
    ap.add_argument("--height", type=int, default=330)
    args = ap.parse_args()

    task = TASKS_BY_ID[args.task]
    backend = DirectBackend()
    backend.reset(task.scene)
    sim = backend.sim
    renderer = mujoco.Renderer(sim.model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.38, 0.0, 0.25]
    cam.distance, cam.azimuth, cam.elevation = 1.75, 150.0, -24.0

    frames = []
    every = max(1, int(round(args.speedup / (args.fps * sim.model.opt.timestep))))
    original_step = sim.step
    counter = {"n": 0}

    def recording_step(n=1):
        for _ in range(n):
            original_step(1)
            counter["n"] += 1
            if counter["n"] % every == 0:
                renderer.update_scene(sim.data, cam)
                frames.append(Image.fromarray(renderer.render()))

    sim.step = recording_step
    init = backend.get_state()

    if args.planner == "oracle":
        from llm_arm_agent.agent.oracle import OraclePlanner

        planner = OraclePlanner(task.oracle)
    else:
        from llm_arm_agent.agent.llm_agent import LLMAgent

        planner = LLMAgent(verbose=True)
    ep = planner.run(task.instruction, ToolExecutor(Skills(backend)))
    recording_step(int(0.8 / sim.model.opt.timestep))  # hold the final pose briefly

    frames = [f.quantize(colors=128, method=0) for f in frames]  # 0 = median cut (works on old Pillow)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps),
                   loop=0, optimize=True)
    ok, why = task.check(init, backend.get_state(), ep)
    print(f"{len(frames)} frames -> {args.out}  (task check: {ok} {why})")


if __name__ == "__main__":
    main()
