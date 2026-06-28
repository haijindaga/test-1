"""End-to-end tests for the closed-loop correction mechanism (Fig. 2).

These drive the loop with the deterministic ``ScriptedPlanner`` so they need
no LLM.  They reproduce the safety-rate contrast of Tables I/II: with the
supervisor the executed action sequence is always safe; without it the
unsafe action gets executed.

Requires `spot`.
"""

import pytest

spot = pytest.importorskip("spot")

from task_layer import ScriptedPlanner, run_closed_loop, run_open_loop  # noqa: E402
from scenarios import SALMON, SORTING, make_supervisor  # noqa: E402


# --------------------------------------------------------------------------- #
#  salmon (Table I / Fig. 3)                                                   #
# --------------------------------------------------------------------------- #
def test_salmon_closed_loop_is_safe_and_succeeds():
    sup = make_supervisor(SALMON)
    planner = ScriptedPlanner(SALMON.scripted_plan)
    result = run_closed_loop(SALMON.env, planner, sup, SALMON.task)

    assert result.safe
    assert result.goal_reached
    assert not result.aborted
    # the unsafe "grab_salmon" must have been discarded at least once
    assert any(rec.rejected for rec in result.steps)
    # in the committed sequence, open precedes grab
    seq = result.action_sequence
    assert seq.index("open_microwave") < seq.index("grab_salmon")


def test_salmon_open_loop_baseline_is_unsafe():
    # Without the supervisor the scripted plan grabs salmon before opening.
    sup = make_supervisor(SALMON)
    planner = ScriptedPlanner(SALMON.scripted_plan)
    seq = run_open_loop(SALMON.env, planner, SALMON.task)
    assert not sup.is_sequence_safe(seq)


# --------------------------------------------------------------------------- #
#  sorting (Table II / Fig. 4)                                                 #
# --------------------------------------------------------------------------- #
def test_sorting_closed_loop_is_safe_and_succeeds():
    sup = make_supervisor(SORTING)
    planner = ScriptedPlanner(SORTING.scripted_plan)
    result = run_closed_loop(SORTING.env, planner, sup, SORTING.task)

    assert result.safe
    assert result.goal_reached
    assert not result.aborted
    # phi1 forced a redirect from white area to red area
    assert any("phi1" in f for rec in result.steps for f in rec.feedback)
    # the recorded task-plan tuple carries obstacle information (motion layer)
    assert any(obs for _, obs in result.plan_tuple)


def test_sorting_first_step_obstacle_matches_paper():
    sup = make_supervisor(SORTING)
    planner = ScriptedPlanner(SORTING.scripted_plan)
    result = run_closed_loop(SORTING.env, planner, sup, SORTING.task)
    # step 0 = go to scan pose, obstacle = white area (Fig. 4 obs0)
    first_action, first_obs = result.plan_tuple[0]
    assert first_action == "go_to_scan_pose"
    assert first_obs == frozenset({"go_to_white_area"})


def test_sorting_open_loop_baseline_is_unsafe():
    sup = make_supervisor(SORTING)
    planner = ScriptedPlanner(SORTING.scripted_plan)
    seq = run_open_loop(SORTING.env, planner, SORTING.task)
    assert not sup.is_sequence_safe(seq)
