"""Benchmark runner: executes tasks, scores them against ground truth, logs metrics."""

from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from ..agent.tools import ToolExecutor
from ..backends.base import ArmBackend
from ..skills import Skills
from .tasks import TASKS, Task


def run_episode(backend: ArmBackend, planner, task: Task) -> Dict:
    backend.reset(task.scene)
    init = backend.get_state()
    skills = Skills(backend)
    executor = ToolExecutor(skills)
    ep = planner.run(task.instruction, executor)
    final = backend.get_state()
    success, reason = task.check(init, final, ep)
    rec = ep.to_dict()
    rec.update({
        "task_id": task.id,
        "category": task.category,
        "backend": backend.name,
        "success": bool(success),
        "check_detail": reason,
        # Did the agent's own finish(success=...) report agree with reality?
        # For infeasible tasks the instruction is never fulfilled, so the right report is False.
        "claim_matches_outcome": (ep.claimed_success is not None
                                  and ep.claimed_success == (success and task.feasible)),
        "motion_plan_s": skills.plan_time,
        "motion_exec_s": skills.exec_time,
        # The direct backend simulates faster than real time; this projects the latency a user
        # would see on a robot executing at real-time speed (MoveIt backend: ~= total_s).
        "realtime_projected_s": (ep.total_s - ep.tool_s + skills.plan_time + skills.exec_time
                                 if backend.name == "direct" else ep.total_s),
        "final_state": final,
    })
    return rec


def run_benchmark(
    backend: ArmBackend,
    make_planner: Callable[[Task], object],
    tasks: Optional[List[Task]] = None,
    trials: int = 1,
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    tasks = tasks or TASKS
    out_dir = out_dir or os.path.join("results", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    records = []
    with open(os.path.join(out_dir, "episodes.jsonl"), "w") as f:
        for task in tasks:
            for trial in range(trials):
                if verbose:
                    print(f"[{task.id} #{trial + 1}] {task.instruction}", flush=True)
                t0 = time.perf_counter()
                rec = run_episode(backend, make_planner(task), task)
                rec["trial"] = trial
                records.append(rec)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if verbose:
                    mark = "PASS" if rec["success"] else "FAIL"
                    err = f" error={rec['error']}" if rec["error"] else ""
                    print(f"    {mark} in {time.perf_counter() - t0:.1f}s  llm={rec['llm_s']:.1f}s "
                          f"calls={rec['n_llm_calls']} tools={rec['n_tool_calls']}  "
                          f"({rec['check_detail']}){err}", flush=True)
    summary = summarize(records)
    summary["meta"] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend.name,
        "planner": records[0]["planner"] if records else None,
        "model": records[0]["model"] if records else None,
        "trials": trials,
        "n_tasks": len(tasks),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(render_markdown(summary))
    if verbose:
        print(render_markdown(summary))
        print(f"Results written to {out_dir}/")
    return summary


# --------------------------------------------------------------------- metrics
def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _lat(xs: List[float]) -> Dict[str, float]:
    return {"mean": statistics.fmean(xs) if xs else float("nan"),
            "p50": _pct(xs, 0.5), "p90": _pct(xs, 0.9), "max": max(xs) if xs else float("nan")}


def summarize(records: List[Dict]) -> Dict:
    per_task: Dict[str, Dict] = {}
    for r in records:
        t = per_task.setdefault(r["task_id"], {"category": r["category"], "instruction": r["instruction"],
                                               "runs": 0, "successes": 0, "total_s": [], "llm_calls": []})
        t["runs"] += 1
        t["successes"] += int(r["success"])
        t["total_s"].append(r["total_s"])
        t["llm_calls"].append(r["n_llm_calls"])
    for t in per_task.values():
        t["success_rate"] = t["successes"] / t["runs"]
        t["mean_total_s"] = statistics.fmean(t.pop("total_s"))
        t["mean_llm_calls"] = statistics.fmean(t.pop("llm_calls"))

    by_cat: Dict[str, List[bool]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r["success"])

    llm_call_lat = [s["seconds"] for r in records for s in r["steps"] if s["kind"] == "llm"]
    n = len(records)
    return {
        "n_episodes": n,
        "success_rate": sum(r["success"] for r in records) / n if n else 0.0,
        "success_by_category": {c: sum(v) / len(v) for c, v in sorted(by_cat.items())},
        "claim_calibration": (sum(r["claim_matches_outcome"] for r in records) / n) if n else 0.0,
        "latency_s": {
            "episode_total": _lat([r["total_s"] for r in records]),
            "episode_llm": _lat([r["llm_s"] for r in records]),
            "episode_tool_exec": _lat([r["tool_s"] for r in records]),
            "time_to_first_action": _lat([r["first_action_s"] for r in records if r["first_action_s"] is not None]),
            "per_llm_call": _lat(llm_call_lat),
            "motion_exec_per_episode": _lat([r["motion_exec_s"] for r in records]),
            "realtime_projected": _lat([r["realtime_projected_s"] for r in records]),
        },
        "mean_llm_calls": statistics.fmean([r["n_llm_calls"] for r in records]) if n else 0,
        "mean_tool_calls": statistics.fmean([r["n_tool_calls"] for r in records]) if n else 0,
        "tool_error_rate": (sum(r["n_tool_errors"] for r in records) / max(1, sum(r["n_tool_calls"] for r in records))),
        "tokens": {
            "input_mean": statistics.fmean([r["input_tokens"] for r in records]) if n else 0,
            "output_mean": statistics.fmean([r["output_tokens"] for r in records]) if n else 0,
            "cache_read_mean": statistics.fmean([r["cache_read_tokens"] for r in records]) if n else 0,
        },
        "per_task": per_task,
    }


def render_markdown(s: Dict) -> str:
    meta = s.get("meta", {})
    lines = [
        f"## Benchmark results: {meta.get('planner', '?')} ({meta.get('model', '?')}) on `{meta.get('backend', '?')}`",
        "",
        f"- Episodes: {s['n_episodes']} ({meta.get('n_tasks', '?')} tasks x {meta.get('trials', '?')} trials)",
        f"- **Success rate: {s['success_rate'] * 100:.1f}%**",
        f"- Agent self-report matches ground truth: {s['claim_calibration'] * 100:.1f}%",
        f"- Mean LLM calls / episode: {s['mean_llm_calls']:.1f}; mean tool calls: {s['mean_tool_calls']:.1f}; "
        f"tool error rate: {s['tool_error_rate'] * 100:.1f}%",
        f"- Mean tokens / episode: in {s['tokens']['input_mean']:.0f} (+{s['tokens']['cache_read_mean']:.0f} cached), "
        f"out {s['tokens']['output_mean']:.0f}",
        "",
        "| Latency (s) | mean | p50 | p90 | max |",
        "|---|---:|---:|---:|---:|",
    ]
    names = {"episode_total": "Episode wall-clock", "episode_llm": "LLM time / episode",
             "episode_tool_exec": "Tool execution / episode", "time_to_first_action": "Time to first action",
             "per_llm_call": "Single LLM call", "motion_exec_per_episode": "Motion execution (sim/robot time)",
             "realtime_projected": "Episode at real-time robot speed"}
    for k, label in names.items():
        v = s["latency_s"].get(k)
        if v is None or v["mean"] != v["mean"]:  # missing or NaN (e.g. no LLM calls)
            continue
        lines.append(f"| {label} | {v['mean']:.2f} | {v['p50']:.2f} | {v['p90']:.2f} | {v['max']:.2f} |")
    lines += ["", "| Category | Success |", "|---|---:|"]
    for c, v in s["success_by_category"].items():
        lines.append(f"| {c} | {v * 100:.0f}% |")
    lines += ["", "| Task | Category | Instruction | Success | Mean time (s) | LLM calls |",
              "|---|---|---|---:|---:|---:|"]
    for tid, t in s["per_task"].items():
        lines.append(f"| {tid} | {t['category']} | {t['instruction']} | {t['successes']}/{t['runs']} | "
                     f"{t['mean_total_s']:.1f} | {t['mean_llm_calls']:.1f} |")
    return "\n".join(lines) + "\n"
