"""LLM agent: turns a natural-language command into a sequence of robot tool calls.

The loop is a plain Messages-API tool-use loop (rather than a hosted agent)
because the tools must run locally against the ROS graph / simulator and
because the benchmark needs per-call latency for the LLM and for execution.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .tools import TOOLS, ToolExecutor

# Model id is supplied by the user: --model on the CLI or the LLM_MODEL environment variable.
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "")

SYSTEM_PROMPT = """\
You control a 7-DoF Franka Panda arm with a parallel gripper mounted at the edge of a table.
You act only through the provided tools; each call runs on the real controller stack \
(simulated), so take actions deliberately.

Frame conventions (robot base frame, metres):
- x points forward, away from the robot; y points to the robot's left (negative y is right); \
z is up, with the table top at z = 0.
- "Left"/"right", "front"/"behind" are from the robot's point of view unless the user says otherwise. \
"In front of X" means larger x than X.
- Objects are 4 cm cubes; a cube resting on the table has its centre at z = 0.02. \
The reachable workspace is roughly 0.3-0.8 m from the base.

How to work:
- Start with get_scene. Use exact object and zone names from it; map colours and descriptions to names yourself.
- Prefer the high-level pick / place_at / place_on skills for manipulation. place_at stacks automatically \
if another object is under the target point.
- If a tool returns an error, read it and adapt (a different order, a free spot, or report failure). \
Do not retry the identical call more than once.
- Before finishing, call get_scene to verify the outcome against the instruction.
- If the instruction refers to an object that does not exist, asks for something outside the workspace, \
or is otherwise impossible, do not move anything; call finish with success=false and explain.
- Always end by calling finish exactly once. Keep any text between tool calls to one short sentence.
"""


@dataclass
class Step:
    kind: str  # "llm" or "tool"
    seconds: float
    name: str = ""
    args: Optional[Dict[str, Any]] = None
    result: str = ""
    is_error: bool = False
    usage: Optional[Dict[str, int]] = None
    stop_reason: str = ""


@dataclass
class Episode:
    instruction: str
    planner: str
    model: str = ""
    steps: List[Step] = field(default_factory=list)
    total_s: float = 0.0
    llm_s: float = 0.0
    tool_s: float = 0.0
    first_action_s: Optional[float] = None
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    finished: bool = False
    claimed_success: Optional[bool] = None
    summary: str = ""
    error: str = ""
    final_text: str = ""

    def add_tool(self, name, args, result, is_error, seconds):
        self.steps.append(Step("tool", seconds, name, args, result[:2000], is_error))
        self.n_tool_calls += 1
        self.n_tool_errors += int(is_error)
        self.tool_s += seconds

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMAgent:
    """Manual tool-use loop on the Anthropic Messages API."""

    name = "llm"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_turns: int = 30,
        client=None,
        verbose: bool = False,
    ):
        import anthropic  # local import so the oracle planner works without the SDK

        self.anthropic = anthropic
        # max_retries covers 429/5xx/connection errors with backoff inside the SDK.
        self.client = client or anthropic.Anthropic(max_retries=4)
        if not model and client is None:
            raise ValueError("No model set: pass --model or export LLM_MODEL=<model id>.")
        self.model = model
        self.effort = effort
        self.max_turns = max_turns
        self.verbose = verbose

    def run(self, instruction: str, executor: ToolExecutor) -> Episode:
        ep = Episode(instruction=instruction, planner=self.name, model=self.model)
        messages: List[Dict[str, Any]] = [{"role": "user", "content": instruction}]
        t_start = time.perf_counter()
        try:
            for _ in range(self.max_turns):
                t0 = time.perf_counter()
                response = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                    output_config={"effort": self.effort},
                    # Tools + system prompt are identical across turns and episodes: cache them.
                    cache_control={"type": "ephemeral"},
                    # If a safety classifier declines, let the API retry on a fallback model.
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
                dt = time.perf_counter() - t0
                self._record_llm(ep, response, dt)
                # Append the full content (thinking + tool_use blocks) unchanged.
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "refusal":
                    ep.error = "model refused the request"
                    break
                if response.stop_reason == "max_tokens":
                    ep.error = "response hit max_tokens"
                    break

                tool_uses = [b for b in response.content if b.type == "tool_use"]
                texts = [b.text for b in response.content if b.type == "text"]
                if texts:
                    ep.final_text = texts[-1]
                if not tool_uses:
                    if response.stop_reason == "pause_turn":
                        continue
                    break  # end_turn without finish: treat as done, finish flag stays False

                if ep.first_action_s is None:
                    ep.first_action_s = time.perf_counter() - t_start
                results = []
                for block in tool_uses:
                    args = block.input if isinstance(block.input, dict) else json.loads(block.input)
                    out, is_error, secs = executor(block.name, args)
                    ep.add_tool(block.name, args, out, is_error, secs)
                    if self.verbose:
                        print(f"  -> {block.name}({json.dumps(args)}) [{secs:.2f}s] {out[:160]}")
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": out, "is_error": is_error})
                messages.append({"role": "user", "content": results})
                if executor.finished:
                    break
            else:
                ep.error = f"exceeded {self.max_turns} turns"
        except self.anthropic.AuthenticationError:
            ep.error = "authentication failed: set ANTHROPIC_API_KEY or run `ant auth login`"
        except self.anthropic.BadRequestError as e:
            ep.error = f"bad request: {e.message}"
        except self.anthropic.RateLimitError as e:
            ep.error = f"rate limited after retries: {e.message}"
        except self.anthropic.APIStatusError as e:
            ep.error = f"API error {e.status_code}: {e.message}"
        except self.anthropic.APIConnectionError:
            ep.error = "network error contacting the API"

        ep.total_s = time.perf_counter() - t_start
        ep.finished = executor.finished
        ep.claimed_success = executor.claimed_success
        ep.summary = executor.summary
        return ep

    def _record_llm(self, ep: Episode, response, dt: float) -> None:
        u = response.usage
        usage = {
            "input": u.input_tokens or 0,
            "output": u.output_tokens or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        ep.steps.append(Step("llm", dt, name=response.model, usage=usage, stop_reason=response.stop_reason or ""))
        ep.n_llm_calls += 1
        ep.llm_s += dt
        ep.input_tokens += usage["input"]
        ep.output_tokens += usage["output"]
        ep.cache_read_tokens += usage["cache_read"]
        ep.cache_write_tokens += usage["cache_write"]
        if self.verbose:
            print(f"  [llm {dt:.2f}s, out={usage['output']} tok, stop={response.stop_reason}]")
