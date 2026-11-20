"""Command-line entry point.

  llm-arm run "stack the red cube on the green cube"     # one command
  llm-arm run                                            # interactive prompt
  llm-arm bench --planner llm --trials 3              # benchmark
  llm-arm bench --planner oracle                         # scripted baseline, no API key needed

Add --backend moveit to go through ROS 2 / MoveIt 2 (launch sim_moveit.launch.py first).
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent.llm_agent import DEFAULT_MODEL
from .bench.tasks import TASKS, TASKS_BY_ID


def make_backend(name: str, viewer: bool = False):
    if name == "direct":
        from .backends.direct import DirectBackend

        return DirectBackend(viewer=viewer)
    if name == "moveit":
        from .backends.moveit import MoveItBackend

        return MoveItBackend()
    raise SystemExit(f"unknown backend {name!r}")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["direct", "moveit"], default="direct")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
    p.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer (direct backend)")


def cmd_run(args) -> int:
    from .agent.llm_agent import LLMAgent
    from .agent.tools import ToolExecutor
    from .skills import Skills

    backend = make_backend(args.backend, args.viewer)
    backend.reset(TASKS_BY_ID[args.scene].scene if args.scene else None)
    agent = LLMAgent(model=args.model, effort=args.effort, verbose=True)
    try:
        instructions = [args.instruction] if args.instruction else None
        while True:
            if instructions is not None:
                if not instructions:
                    break
                text = instructions.pop()
            else:
                try:
                    text = input("\ncommand> ").strip()
                except EOFError:
                    break
                if text in ("", "quit", "exit"):
                    break
            ep = agent.run(text, ToolExecutor(Skills(backend)))
            status = "done" if ep.claimed_success else "not done"
            print(f"\n[{status}] {ep.summary or ep.final_text or ep.error}")
            print(f"  total {ep.total_s:.1f}s | LLM {ep.llm_s:.1f}s over {ep.n_llm_calls} calls | "
                  f"tools {ep.tool_s:.1f}s over {ep.n_tool_calls} calls")
    finally:
        backend.close()
    return 0


def cmd_bench(args) -> int:
    from .bench.runner import run_benchmark

    tasks = [TASKS_BY_ID[t] for t in args.tasks.split(",")] if args.tasks else TASKS
    backend = make_backend(args.backend, args.viewer)
    if args.planner == "oracle":
        from .agent.oracle import OraclePlanner

        def make_planner(task):
            return OraclePlanner(task.oracle)
    else:
        from .agent.llm_agent import LLMAgent

        agent = LLMAgent(model=args.model, effort=args.effort, verbose=args.verbose)

        def make_planner(task):
            return agent
    try:
        summary = run_benchmark(backend, make_planner, tasks, args.trials, args.out)
    finally:
        backend.close()
    return 0 if summary["n_episodes"] else 1


def cmd_list(_args) -> int:
    for t in TASKS:
        print(f"{t.id:22s} {t.category:11s} {t.instruction}")
    return 0


def cmd_report(args) -> int:
    from .bench.runner import render_markdown

    with open(args.summary) as f:
        print(render_markdown(json.load(f)))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="llm-arm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="send a natural-language command to the arm")
    p.add_argument("instruction", nargs="?", help="omit for an interactive prompt")
    p.add_argument("--scene", choices=sorted(TASKS_BY_ID), help="start from a benchmark task's scene")
    _add_common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("bench", help="run the benchmark suite")
    p.add_argument("--planner", choices=["llm", "oracle"], default="llm")
    p.add_argument("--tasks", help="comma-separated task ids (default: all)")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--out", help="output directory (default: results/<timestamp>)")
    p.add_argument("--verbose", action="store_true", help="print every tool call")
    _add_common(p)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("list", help="list benchmark tasks")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("report", help="render a summary.json as markdown")
    p.add_argument("summary")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
