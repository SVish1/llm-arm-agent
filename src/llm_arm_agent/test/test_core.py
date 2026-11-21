"""Tests that run without ROS or an API key: sim, skills, tasks, and the agent loop."""

import json
import math
from types import SimpleNamespace

import pytest

from llm_arm_agent.agent.llm_agent import LLMAgent
from llm_arm_agent.agent.oracle import OraclePlanner
from llm_arm_agent.agent.tools import TOOLS, ToolExecutor
from llm_arm_agent.backends.direct import DirectBackend
from llm_arm_agent.bench.runner import run_episode, summarize
from llm_arm_agent.bench.tasks import TASKS, TASKS_BY_ID
from llm_arm_agent.skills import SkillError, Skills


@pytest.fixture(scope="module")
def backend():
    return DirectBackend()


def test_ready_pose_matches_panda_fk(backend):
    backend.reset()
    s = backend.get_state()
    # Known Panda FK: hand at (0.307, 0, 0.590) in the ready pose; TCP is 0.1034 m below.
    assert s["ee_position"] == pytest.approx([0.307, 0.0, 0.4869], abs=2e-3)


def test_tool_schemas_are_strict():
    names = set()
    for t in TOOLS:
        schema = t["input_schema"]
        assert t["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        names.add(t["name"])
    assert "finish" in names and "pick" in names


def test_benchmark_has_10_to_15_tasks():
    assert 10 <= len(TASKS) <= 15
    assert len({t.id for t in TASKS}) == len(TASKS)


@pytest.mark.parametrize("task_id", [t.id for t in TASKS])
def test_oracle_solves_every_task(backend, task_id):
    task = TASKS_BY_ID[task_id]
    rec = run_episode(backend, OraclePlanner(task.oracle), task)
    assert rec["success"], rec["check_detail"]
    assert rec["claim_matches_outcome"]


def test_doing_nothing_fails_feasible_tasks(backend):
    """Checks must not pass trivially from the initial scene."""
    noop = OraclePlanner(lambda s: [("finish", {"success": True, "summary": "noop"})])
    for task in TASKS:
        if task.feasible:
            assert not run_episode(backend, noop, task)["success"], task.id


def test_skill_errors_are_reported(backend):
    backend.reset(TASKS_BY_ID["T04_pick"].scene)
    skills = Skills(backend)
    ex = ToolExecutor(skills)
    out, is_error, _ = ex("pick", {"object_name": "purple_cube"})
    assert is_error and "Unknown object" in json.loads(out)["error"]
    out, is_error, _ = ex("move_to", {"x": 1.5, "y": 0, "z": 0.2, "yaw_deg": 0})
    assert is_error
    out, is_error, _ = ex("place_at", {"x": 0.5, "y": 0.0})
    assert is_error and "Not holding" in json.loads(out)["error"]
    with pytest.raises(SkillError):
        skills.move_to(0.5, 0.0, 0.0)


# ------------------------------------------------------------ agent loop (fake API)
def _resp(blocks, stop="tool_use"):
    return SimpleNamespace(
        content=blocks, stop_reason=stop, model="fake-model",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                              cache_read_input_tokens=50, cache_creation_input_tokens=0),
    )


def _tool(i, name, args):
    return SimpleNamespace(type="tool_use", id=f"tu_{i}", name=name, input=args)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot history
        return self.responses.pop(0)


def test_agent_loop_executes_tools_and_finishes(backend):
    task = TASKS_BY_ID["T06_stack"]
    client = FakeClient([
        _resp([_tool(1, "get_scene", {})]),
        _resp([SimpleNamespace(type="text", text="Picking red."), _tool(2, "pick", {"object_name": "red_cube"})]),
        _resp([_tool(3, "place_on", {"target": "green_cube"})]),
        _resp([_tool(4, "finish", {"success": True, "summary": "Stacked."})]),
    ])
    agent = LLMAgent(client=client)
    rec = run_episode(backend, agent, task)
    assert rec["success"], rec["check_detail"]
    assert rec["n_llm_calls"] == 4 and rec["n_tool_calls"] == 4
    assert rec["claimed_success"] is True and rec["claim_matches_outcome"]
    assert rec["cache_read_tokens"] == 200
    # Every request carries the full history; tool results go back with matching ids.
    last = client.calls[-1]["messages"]
    assert last[-1]["role"] == "user" and last[-1]["content"][0]["tool_use_id"] == "tu_3"
    assert client.calls[0]["output_config"] == {"effort": "medium"}


def test_agent_loop_surfaces_tool_errors(backend):
    backend.reset(TASKS_BY_ID["T14_missing_object"].scene)
    client = FakeClient([
        _resp([_tool(1, "pick", {"object_name": "purple_cube"})]),
        _resp([_tool(2, "finish", {"success": False, "summary": "No purple cube."})]),
    ])
    ep = LLMAgent(client=client).run("Pick up the purple cube.", ToolExecutor(Skills(backend)))
    err_result = client.calls[1]["messages"][-1]["content"][0]
    assert err_result["is_error"] is True
    assert ep.n_tool_errors == 1 and ep.claimed_success is False


def test_agent_stops_on_refusal(backend):
    backend.reset()
    client = FakeClient([_resp([], stop="refusal")])
    ep = LLMAgent(client=client).run("x", ToolExecutor(Skills(backend)))
    assert "refused" in ep.error and ep.n_llm_calls == 1


def test_summary_metrics(backend):
    recs = [run_episode(backend, OraclePlanner(t.oracle), t) for t in TASKS[:3]]
    for r in recs:
        r["trial"] = 0
    s = summarize(recs)
    assert s["success_rate"] == 1.0
    assert not math.isnan(s["latency_s"]["episode_total"]["p50"])


def test_compare_flags_regressions(backend):
    from llm_arm_agent.bench.runner import render_comparison

    tasks = TASKS[:3]
    base = [run_episode(backend, OraclePlanner(t.oracle), t) for t in tasks]
    worse = [dict(r) for r in base]
    worse[1]["success"] = False
    for r in base + worse:
        r["trial"] = 0
    md = render_comparison(summarize(base), summarize(worse), "base", "new")
    assert "regressed" in md and tasks[1].id in md
    assert "-33.3 pp worse" in md
    assert "identical" in render_comparison(summarize(base), summarize(base))
