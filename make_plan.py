"""Phase 1 of the cross-layer motion demo: generate the task plan as JSON.

The "brain" (this repo) runs a scenario through the safety supervisor
(CheckSafety + FindObstacle) and writes a plain JSON file that the motion layer
(Phase 2, the `motion_executor` ROS 2 node) consumes. The two halves stay
decoupled: this script needs only `spot` (+ optionally Ollama); the executor
needs only ROS 2 / MoveIt. They communicate through the JSON file alone.

Two plans are written so the motion side can show the with/without-supervisor
contrast (paper Fig. 6 (a) vs (b)):

    <scenario>_with.json     run_closed_loop  -> safety-corrected actions
                                                 + per-step obstacle regions
    <scenario>_without.json  run_open_loop    -> raw actions, no obstacles

plan.json carries only WHAT to do (action names + which regions to avoid);
WHERE each region is in space lives in the motion side's layout.yaml. Keeping
them separate means you can move regions around without touching the plan.

Usage:
    python make_plan.py                                # scripted (deterministic)
    python make_plan.py --scenario sorting --out-dir plans
    python make_plan.py --llm                          # use the local Ollama planner
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from task_layer import (
    DONE,
    OllamaPlanner,
    ScriptedPlanner,
    run_closed_loop,
    run_open_loop,
)
from scenarios import get_scenario, make_supervisor

# Plan-file schema version, so the executor can check compatibility.
PLAN_FORMAT = 1


def _region_name(obstacle_action: str) -> str:
    """Map a region action id to a region name, e.g. 'go_to_white_area' -> 'white_area'.

    FindObstacle returns the *actions* that are unsafe (the ``go_to_<region>``
    ids); the motion side keys its layout by region name, so we strip the prefix.
    """
    prefix = "go_to_"
    return obstacle_action[len(prefix):] if obstacle_action.startswith(prefix) else obstacle_action


def build_plan_with_supervisor(scenario, planner, task: str) -> Dict:
    """Run the closed loop (CheckSafety + FindObstacle) and serialise the result."""
    supervisor = make_supervisor(scenario)
    result = run_closed_loop(scenario.env, planner, supervisor, task)

    steps: List[Dict] = []
    for action, obstacles in result.plan_tuple:
        if action == DONE:
            continue
        steps.append(
            {
                "action": action,
                "obstacles": sorted(_region_name(o) for o in obstacles),
            }
        )

    return {
        "plan_format": PLAN_FORMAT,
        "scenario": scenario.name,
        "task": task,
        "supervisor": True,
        "goal_reached": result.goal_reached,
        "safe": result.safe,
        "aborted": result.aborted,
        "steps": steps,
    }


def build_plan_without_supervisor(scenario, planner, task: str) -> Dict:
    """Run the open loop (no supervisor): raw actions, no obstacles (the baseline)."""
    actions = run_open_loop(scenario.env, planner, task)
    steps = [{"action": a, "obstacles": []} for a in actions if a != DONE]

    # Record whether this raw sequence is actually safe (for the comparison).
    supervisor = make_supervisor(scenario)
    return {
        "plan_format": PLAN_FORMAT,
        "scenario": scenario.name,
        "task": task,
        "supervisor": False,
        "goal_reached": None,  # the open loop does not track the goal the same way
        "safe": supervisor.is_sequence_safe(actions),
        "aborted": False,
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="sorting", help="scenario name")
    parser.add_argument("--task", default=None,
                        help="override the task instruction (free-form; great with --llm)")
    parser.add_argument("--out-dir", default="plans", help="where to write the JSON")
    parser.add_argument("--llm", action="store_true", help="use the Ollama planner")
    args = parser.parse_args()

    scenario = get_scenario(args.scenario)
    task = args.task or scenario.task
    os.makedirs(args.out_dir, exist_ok=True)

    def fresh_planner():
        # a fresh planner per run so the scripted queue restarts
        return OllamaPlanner() if args.llm else ScriptedPlanner(scenario.scripted_plan)

    plans = {
        "with": build_plan_with_supervisor(scenario, fresh_planner(), task),
        "without": build_plan_without_supervisor(scenario, fresh_planner(), task),
    }

    for suffix, plan in plans.items():
        path = os.path.join(args.out_dir, f"{scenario.name}_{suffix}.json")
        with open(path, "w") as f:
            json.dump(plan, f, indent=2)
        n_go = sum(1 for s in plan["steps"] if s["action"].startswith("go_to_"))
        print(
            f"wrote {path}  ({len(plan['steps'])} steps, {n_go} go-moves, "
            f"safe={plan['safe']})"
        )


if __name__ == "__main__":
    main()
