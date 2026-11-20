"""Scripted 'oracle' planner: replays a task's hand-written tool sequence.

It goes through exactly the same ToolExecutor -> Skills -> backend path as
the LLM agent, so it validates the task suite and the execution stack, and
gives an LLM-free reference for success rate and execution latency.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Tuple

from .llm_agent import Episode
from .tools import ToolExecutor


class OraclePlanner:
    name = "oracle"

    def __init__(self, script_fn: Callable[[Dict], List[Tuple[str, Dict]]]):
        self.script_fn = script_fn

    def run(self, instruction: str, executor: ToolExecutor) -> Episode:
        ep = Episode(instruction=instruction, planner=self.name, model="scripted")
        t_start = time.perf_counter()
        calls = self.script_fn(executor.skills.backend.get_state())
        if not any(name == "finish" for name, _ in calls):
            calls = calls + [("finish", {"success": True, "summary": "scripted sequence done"})]
        ep.first_action_s = 0.0
        for name, args in calls:
            out, is_error, secs = executor(name, args)
            ep.add_tool(name, args, out, is_error, secs)
            if is_error:
                ep.error = f"{name} failed: {out}"
                break
        ep.total_s = time.perf_counter() - t_start
        ep.finished = executor.finished
        ep.claimed_success = executor.claimed_success
        ep.summary = executor.summary
        return ep
