"""Concrete scenarios from the Cross-Layer paper.

Each scenario bundles everything needed to run the closed loop:

    * the action space A (clean atomic-proposition ids + human labels),
    * the LTL safety constraints Phi,
    * a ``SymbolicEnv`` (state transition system T and goal),
    * the region actions used by FindObstacle (motion layer),
    * a deterministic ``scripted_plan`` reproducing the paper's figure with the
      ``ScriptedPlanner`` (no LLM required).

Two scenarios are provided:

    SALMON   -- Table I / Fig. 3, "Put salmon in the microwave"
                (task layer only; no motion / obstacles).
    SORTING  -- Table II / Fig. 4, "Place white cylinder over white area"
                (dual layer: CheckSafety + FindObstacle).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from task_layer import Constraint, State, SymbolicEnv


@dataclass
class Scenario:
    name: str
    task: str
    action_space: List[str]
    labels: Dict[str, str]
    constraints: List[Constraint]
    env: SymbolicEnv
    region_actions: List[str]
    scripted_plan: List[str]


# --------------------------------------------------------------------------- #
#  Scenario 1 : salmon in the microwave  (Table I / Fig. 3)                    #
# --------------------------------------------------------------------------- #
def _salmon_scenario() -> Scenario:
    actions = [
        "find_microwave",
        "open_microwave",
        "find_salmon",
        "grab_salmon",
        "put_salmon_in_microwave",
        "close_microwave",
        "done",
    ]
    labels = {
        "find_microwave": "Find microwave",
        "open_microwave": "Open microwave",
        "find_salmon": "Find salmon",
        "grab_salmon": "Grab salmon",
        "put_salmon_in_microwave": "Put salmon in microwave",
        "close_microwave": "Close microwave",
        "done": "DONE",
    }

    constraints = [
        Constraint(
            name="no_grab_before_open",
            ltl="(!grab_salmon) U open_microwave",
            description="Do not grab the salmon until the microwave is open.",
            paper_formula="G(F(!grab_salmon U open_microwave))",
        )
    ]

    def transition(state: State, action: str) -> State:
        s = set(state)
        if action == "find_microwave":
            s.add("found:microwave")
        elif action == "open_microwave":
            s.add("open:microwave")
            s.discard("closed:microwave")
        elif action == "find_salmon":
            s.add("found:salmon")
        elif action == "grab_salmon":
            s.add("holding:salmon")
        elif action == "put_salmon_in_microwave":
            if "holding:salmon" in s and "open:microwave" in s:
                s.add("in_microwave:salmon")
                s.discard("holding:salmon")
        elif action == "close_microwave":
            if "open:microwave" in s:
                s.add("closed:microwave")
                s.discard("open:microwave")
        return frozenset(s)

    def goal(state: State) -> bool:
        return "in_microwave:salmon" in state and "closed:microwave" in state

    env = SymbolicEnv(
        initial=frozenset(),
        transition_fn=transition,
        goal_fn=goal,
        action_space=actions,
        labels=labels,
    )

    # An LLM-like plan that tries the unsafe "grab salmon" before opening the
    # microwave; the supervisor forces the open-then-grab order (Fig. 3).
    scripted_plan = [
        "find_microwave",
        "find_salmon",
        "grab_salmon",   # UNSAFE here -> discarded
        "open_microwave",  # regenerated
        "grab_salmon",   # now safe
        "put_salmon_in_microwave",
        "close_microwave",
        "done",
    ]

    return Scenario(
        name="salmon",
        task="Put the salmon in the microwave",
        action_space=actions,
        labels=labels,
        constraints=constraints,
        env=env,
        region_actions=[],  # task layer only -> no obstacles
        scripted_plan=scripted_plan,
    )


# --------------------------------------------------------------------------- #
#  Scenario 2 : place white cylinder over white area  (Table II / Fig. 4)      #
# --------------------------------------------------------------------------- #
def _sorting_scenario() -> Scenario:
    regions = ["scan_pose", "white_area", "yellow_area", "blue_area", "red_area"]
    go_actions = [f"go_to_{r}" for r in regions]
    actions = go_actions + [
        "find_white_cylinder",
        "grab_white_cylinder",
        "place_white_cylinder",
        "done",
    ]
    labels = {f"go_to_{r}": f"Go to {r.replace('_', ' ')}" for r in regions}
    labels.update(
        {
            "find_white_cylinder": "Find white cylinder",
            "grab_white_cylinder": "Grab white cylinder",
            "place_white_cylinder": "Place white cylinder",
            "done": "DONE",
        }
    )

    # Operative safety cores of the three constraints in Fig. 4 (the printed
    # G(F(...)) wrappers are recorded in `paper_formula`).
    constraints = [
        Constraint(
            name="phi1_white_until_red",
            ltl="(!go_to_white_area) U go_to_red_area",
            description="Do not go to the white area until the red area is visited.",
            paper_formula="G(F(!Go_to_white_area U Go_to_red_area))",
        ),
        Constraint(
            name="phi2_place_then_yellow",
            ltl="G(place_white_cylinder -> X(go_to_yellow_area))",
            description="After placing the white cylinder, go to the yellow area next.",
            paper_formula="G(F(Place_white_cylinder X Go_to_yellow_area))",
        ),
        Constraint(
            name="phi3_find_not_yellow_next",
            ltl="G(find_white_cylinder -> X(!go_to_yellow_area))",
            description="After finding the white cylinder, do not go to the yellow area next.",
            paper_formula="G(F(Find_white_cylinder X !Go_to_yellow_area))",
        ),
    ]

    def transition(state: State, action: str) -> State:
        s = set(state)
        if action.startswith("go_to_"):
            region = action[len("go_to_"):]
            s = {p for p in s if not p.startswith("robot_at:")}
            s.add(f"robot_at:{region}")
            s.add(f"visited:{region}")
        elif action == "find_white_cylinder":
            s.add("found:white_cylinder")
        elif action == "grab_white_cylinder":
            if "found:white_cylinder" in s:
                s.add("holding:white_cylinder")
        elif action == "place_white_cylinder":
            if "holding:white_cylinder" in s and "robot_at:white_area" in s:
                s.discard("holding:white_cylinder")
                s.add("placed:white_cylinder@white_area")
        return frozenset(s)

    def goal(state: State) -> bool:
        return "placed:white_cylinder@white_area" in state

    env = SymbolicEnv(
        initial=frozenset({"robot_at:scan_pose", "visited:scan_pose"}),
        transition_fn=transition,
        goal_fn=goal,
        action_space=actions,
        labels=labels,
    )

    # Reproduces Fig. 4: the planner first tries "go to white area" (violates
    # phi1 -> redirected to red area), then completes via white area; after
    # placing it must visit yellow (phi2) before finishing.
    scripted_plan = [
        "go_to_scan_pose",
        "find_white_cylinder",
        "grab_white_cylinder",
        "go_to_white_area",   # UNSAFE: phi1 (red not visited) -> discarded
        "go_to_red_area",     # regenerated
        "go_to_white_area",   # now safe (red visited)
        "place_white_cylinder",
        "go_to_yellow_area",  # required next by phi2
        "done",
    ]

    return Scenario(
        name="sorting",
        task="Place the white cylinder over the white area",
        action_space=actions,
        labels=labels,
        constraints=constraints,
        env=env,
        region_actions=go_actions,
        scripted_plan=scripted_plan,
    )


SALMON = _salmon_scenario()
SORTING = _sorting_scenario()

SCENARIOS: Dict[str, Scenario] = {SALMON.name: SALMON, SORTING.name: SORTING}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {list(SCENARIOS)}")
    return SCENARIOS[name]


def make_supervisor(scenario: Scenario):
    """Build a :class:`SafetySupervisor` for ``scenario`` (requires spot)."""
    from task_layer import SafetySupervisor

    return SafetySupervisor(
        constraints=scenario.constraints,
        atomic_propositions=scenario.action_space,
        region_actions=scenario.region_actions,
    )
