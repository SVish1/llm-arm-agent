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


# ------------------------------------------------------------------ comparison
def _fmt_delta(a: float, b: float, scale: float = 1.0, unit: str = "", lower_is_better: bool = False) -> str:
    if a != a or b != b:  # NaN on either side
        return "n/a"
    d = (b - a) * scale
    if abs(d) < 1e-9:
        return f"0{unit}"
    better = (d < 0) if lower_is_better else (d > 0)
    return f"{d:+.1f}{unit} {'better' if better else 'worse'}"


def _label(s: Dict, fallback: str) -> str:
    meta = s.get("meta", {})
    if not meta:
        return fallback
    return f"{meta.get('planner', '?')}/{meta.get('backend', '?')}"


def render_comparison(a: Dict, b: Dict, name_a: str = "A", name_b: str = "B") -> str:
    """Markdown diff of two benchmark summaries (B relative to A)."""
    la, lb = _label(a, name_a), _label(b, name_b)
    lines = [f"## Benchmark comparison: {name_a} ({la}) vs {name_b} ({lb})", "",
             f"| Metric | {name_a} | {name_b} | Δ ({name_b} − {name_a}) |", "|---|---:|---:|---|"]

    def row(label, va, vb, scale=1.0, unit="", fmt="{:.2f}", lower=False):
        sa = "n/a" if va != va else fmt.format(va * scale)
        sb = "n/a" if vb != vb else fmt.format(vb * scale)
        lines.append(f"| {label} | {sa} | {sb} | {_fmt_delta(va, vb, scale, unit, lower)} |")

    row("Success rate", a["success_rate"], b["success_rate"], 100, " pp", "{:.1f}%")
    row("Self-report calibration", a["claim_calibration"], b["claim_calibration"], 100, " pp", "{:.1f}%")
    row("Tool error rate", a["tool_error_rate"], b["tool_error_rate"], 100, " pp", "{:.1f}%", lower=True)
    row("Mean LLM calls / episode", a["mean_llm_calls"], b["mean_llm_calls"], lower=True)
    row("Mean tool calls / episode", a["mean_tool_calls"], b["mean_tool_calls"], lower=True)
    for key, label in (("episode_total", "Episode p50 (s)"), ("per_llm_call", "LLM call p50 (s)"),
                       ("time_to_first_action", "Time to first action p50 (s)"),
                       ("realtime_projected", "Real-time projected p50 (s)")):
        va = a["latency_s"].get(key, {}).get("p50", float("nan"))
        vb = b["latency_s"].get(key, {}).get("p50", float("nan"))
        row(label, va, vb, lower=True)
    row("Output tokens / episode", a["tokens"]["output_mean"], b["tokens"]["output_mean"], fmt="{:.0f}", lower=True)

    cats = sorted(set(a["success_by_category"]) | set(b["success_by_category"]))
    lines += ["", f"| Category | {name_a} | {name_b} |", "|---|---:|---:|"]
    for c in cats:
        ca, cb = a["success_by_category"].get(c), b["success_by_category"].get(c)
        lines.append(f"| {c} | {'-' if ca is None else f'{ca * 100:.0f}%'} | {'-' if cb is None else f'{cb * 100:.0f}%'} |")

    changed = []
    for tid in sorted(set(a["per_task"]) | set(b["per_task"])):
        ta, tb = a["per_task"].get(tid), b["per_task"].get(tid)
        ra = None if ta is None else ta["success_rate"]
        rb = None if tb is None else tb["success_rate"]
        if ra != rb:
            fa = "-" if ta is None else f"{ta['successes']}/{ta['runs']}"
            fb = "-" if tb is None else f"{tb['successes']}/{tb['runs']}"
            tag = "only in one run" if ra is None or rb is None else ("improved" if rb > ra else "regressed")
            changed.append(f"| {tid} | {fa} | {fb} | {tag} |")
    lines += [""]
    if changed:
        lines += [f"| Task with changed success | {name_a} | {name_b} | |", "|---|---:|---:|---|", *changed]
    else:
        lines.append("Per-task success is identical in both runs.")
    return "\n".join(lines) + "\n"
