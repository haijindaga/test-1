"""Tests for the symbolic environment and loop control flow.

These do NOT require spot or Ollama: the supervisor is replaced by a stub, so
they exercise transitions, goal detection and the loop plumbing on any machine.
"""

from task_layer import ScriptedPlanner, run_closed_loop, run_open_loop
from scenarios import SALMON, SORTING


class _AllowAllSupervisor:
    """Stub supervisor that approves everything (no spot needed)."""

    def check_safety(self, action_sequence, next_action):
        return []

    def find_obstacle(self, action_sequence, current_action):
        return frozenset()

    def is_sequence_safe(self, action_sequence):
        return True


def test_salmon_transitions_reach_goal():
    env = SALMON.env
    state = env.initial_state()
    for action in [
        "find_microwave",
        "find_salmon",
        "open_microwave",
        "grab_salmon",
        "put_salmon_in_microwave",
        "close_microwave",
    ]:
        state = env.transition(state, action)
    assert env.is_goal(state)


def test_salmon_grab_without_holding_blocks_put():
    env = SALMON.env
    state = env.initial_state()
    # put without holding/open does nothing
    state = env.transition(state, "put_salmon_in_microwave")
    assert "in_microwave:salmon" not in state


def test_sorting_place_requires_holding_and_location():
    env = SORTING.env
    state = env.initial_state()
    state = env.transition(state, "place_white_cylinder")  # nothing held
    assert "placed:white_cylinder@white_area" not in state

    state = env.initial_state()
    state = env.transition(state, "find_white_cylinder")
    state = env.transition(state, "grab_white_cylinder")
    state = env.transition(state, "go_to_white_area")
    state = env.transition(state, "place_white_cylinder")
    assert env.is_goal(state)


def test_robot_location_is_unique():
    env = SORTING.env
    state = env.transition(env.initial_state(), "go_to_red_area")
    state = env.transition(state, "go_to_white_area")
    at = {p for p in state if p.startswith("robot_at:")}
    assert at == {"robot_at:white_area"}


def test_closed_loop_with_stub_supervisor_runs():
    # With an allow-all supervisor the scripted plan executes verbatim.
    result = run_closed_loop(
        SALMON.env, ScriptedPlanner(SALMON.scripted_plan), _AllowAllSupervisor(),
        SALMON.task,
    )
    assert "grab_salmon" in result.action_sequence
    assert result.goal_reached
