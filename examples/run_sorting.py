"""Reproduce the sorting scenario (Table II / Fig. 4).

    python examples/run_sorting.py           # deterministic ScriptedPlanner
    python examples/run_sorting.py --llm     # use local Ollama as the planner

Shows the dual-layer behaviour: CheckSafety redirects the unsafe "go to white
area" to "go to red area", and FindObstacle annotates each step with the
regions to avoid (the task plan tuple).

Requires spot. The --llm mode also requires a running Ollama server.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_layer import OllamaPlanner, ScriptedPlanner, run_closed_loop, run_open_loop
from scenarios import SORTING, make_supervisor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="use the Ollama planner")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="max planning steps before giving up (default 30)")
    parser.add_argument("--max-retries", type=int, default=8,
                        help="max regenerations per step on a violation (default 8)")
    parser.add_argument("--stop-at-goal", action="store_true",
                        help="stop as soon as the goal is reached")
    args = parser.parse_args()

    scenario = SORTING
    supervisor = make_supervisor(scenario)

    print(f"Scenario : {scenario.name}")
    print(f"Task     : {scenario.task}")
    print("Constraints:")
    for c in scenario.constraints:
        print(f"  - {c.name}: {c.ltl}")
        print(f"      ({c.description})")
    print()

    planner = OllamaPlanner() if args.llm else ScriptedPlanner(scenario.scripted_plan)

    print("Closed loop (task plan tuple = action, obstacle):")
    result = run_closed_loop(
        scenario.env, planner, supervisor, scenario.task,
        max_steps=args.max_steps, max_retries=args.max_retries,
        stop_at_goal=args.stop_at_goal,
    )
    print(result.format(scenario.env))
    print()

    if not args.llm:
        seq = run_open_loop(scenario.env, ScriptedPlanner(scenario.scripted_plan),
                            scenario.task)
        labels = scenario.labels
        print("Open loop (no supervisor) executed:")
        print("  " + " -> ".join(labels.get(a, a) for a in seq))
        print(f"  safe = {supervisor.is_sequence_safe(seq)}")


if __name__ == "__main__":
    main()
