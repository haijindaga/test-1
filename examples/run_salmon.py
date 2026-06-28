"""Reproduce the salmon scenario (Table I / Fig. 3).

    python examples/run_salmon.py            # deterministic ScriptedPlanner
    python examples/run_salmon.py --llm      # use local Ollama as the planner

Requires spot. The --llm mode also requires a running Ollama server.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_layer import OllamaPlanner, ScriptedPlanner, run_closed_loop, run_open_loop
from scenarios import SALMON, make_supervisor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="use the Ollama planner")
    args = parser.parse_args()

    scenario = SALMON
    supervisor = make_supervisor(scenario)

    print(f"Scenario : {scenario.name}")
    print(f"Task     : {scenario.task}")
    print("Constraints:")
    for c in scenario.constraints:
        print(f"  - {c.name}: {c.ltl}")
        print(f"      ({c.description})")
    print()

    planner = OllamaPlanner() if args.llm else ScriptedPlanner(scenario.scripted_plan)

    print("Closed loop (with cross-layer safety supervisor):")
    result = run_closed_loop(scenario.env, planner, supervisor, scenario.task)
    print(result.format(scenario.env))
    print()

    # Baseline contrast (open loop, no supervisor) -- only meaningful for the
    # deterministic planner, which proposes the unsafe order.
    if not args.llm:
        seq = run_open_loop(scenario.env, ScriptedPlanner(scenario.scripted_plan),
                            scenario.task)
        labels = scenario.labels
        print("Open loop (no supervisor) executed:")
        print("  " + " -> ".join(labels.get(a, a) for a in seq))
        print(f"  safe = {supervisor.is_sequence_safe(seq)}")


if __name__ == "__main__":
    main()
