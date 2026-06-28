"""Unit tests for SafetySupervisor.find_obstacle (Algorithm 2).

Reproduces the obstacle sets annotated in Fig. 4 of the paper.  ``find_obstacle``
takes the history *before* the current action (Sigma_{i-1}) and the current
action sigma_i, and returns the regions that are unsafe to enter at this step.

Requires `spot`; does NOT require Ollama.
"""

import pytest

spot = pytest.importorskip("spot")

from scenarios import SORTING, make_supervisor  # noqa: E402


def test_obs0_scan_pose_is_white_area():
    # sigma0 = go to scan pose, history = []. Before visiting red, the white
    # area is forbidden (phi1). Fig. 4 "obs0: white area".
    sup = make_supervisor(SORTING)
    obstacles = sup.find_obstacle([], "go_to_scan_pose")
    assert obstacles == frozenset({"go_to_white_area"})


def test_obs1_find_step_is_white_area_only():
    # sigma1 = find white cylinder, history = [scan]. phi3 ("not yellow right
    # after find") is not active yet, so only white is forbidden.
    # Fig. 4 "obs1: white area".
    sup = make_supervisor(SORTING)
    obstacles = sup.find_obstacle(["go_to_scan_pose"], "find_white_cylinder")
    assert obstacles == frozenset({"go_to_white_area"})


def test_obs2_grab_step_is_white_and_yellow():
    # sigma2 = grab white cylinder, history = [scan, find]. Now going to yellow
    # would be the move right after find -> phi3 forbids it; white still
    # forbidden by phi1. Fig. 4 "obs2: white area, yellow area".
    sup = make_supervisor(SORTING)
    obstacles = sup.find_obstacle(
        ["go_to_scan_pose", "find_white_cylinder"], "grab_white_cylinder"
    )
    assert obstacles == frozenset({"go_to_white_area", "go_to_yellow_area"})


def test_no_obstacles_after_red_visited():
    # Once red has been visited, phi1 is satisfied: the white area is free and
    # (no find in the immediate past) nothing else is forbidden.
    sup = make_supervisor(SORTING)
    history = ["go_to_scan_pose", "go_to_red_area"]
    obstacles = sup.find_obstacle(history, "go_to_blue_area")
    assert obstacles == frozenset()
