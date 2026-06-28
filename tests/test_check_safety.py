"""Unit tests for SafetySupervisor.check_safety (Algorithm 1).

These require `spot` (LTL backend) and do NOT require Ollama.  On a machine
without spot they are skipped.
"""

import pytest

spot = pytest.importorskip("spot")

from task_layer import SafetySupervisor  # noqa: E402
from scenarios import SALMON, SORTING, make_supervisor  # noqa: E402


# --------------------------------------------------------------------------- #
#  salmon : (!grab_salmon) U open_microwave                                    #
# --------------------------------------------------------------------------- #
def test_grab_before_open_is_rejected():
    sup = make_supervisor(SALMON)
    feedback = sup.check_safety(["find_microwave", "find_salmon"], "grab_salmon")
    assert feedback, "grabbing salmon before opening the microwave must be flagged"
    assert feedback[0].name == "no_grab_before_open"


def test_open_then_grab_is_safe():
    sup = make_supervisor(SALMON)
    seq = ["find_microwave", "find_salmon", "open_microwave"]
    assert sup.check_safety(seq, "grab_salmon") == []


def test_grab_first_action_is_rejected():
    sup = make_supervisor(SALMON)
    assert sup.check_safety([], "grab_salmon")


def test_unrelated_actions_are_safe():
    sup = make_supervisor(SALMON)
    assert sup.check_safety([], "find_microwave") == []
    assert sup.check_safety(["find_microwave"], "find_salmon") == []


# --------------------------------------------------------------------------- #
#  sorting : phi1 / phi2 / phi3                                                #
# --------------------------------------------------------------------------- #
def test_phi1_white_before_red_is_rejected():
    sup = make_supervisor(SORTING)
    feedback = sup.check_safety(["go_to_scan_pose"], "go_to_white_area")
    names = {c.name for c in feedback}
    assert "phi1_white_until_red" in names


def test_phi1_white_after_red_is_safe():
    sup = make_supervisor(SORTING)
    seq = ["go_to_scan_pose", "go_to_red_area"]
    assert sup.check_safety(seq, "go_to_white_area") == []


def test_phi3_yellow_right_after_find_is_rejected():
    sup = make_supervisor(SORTING)
    feedback = sup.check_safety(["find_white_cylinder"], "go_to_yellow_area")
    names = {c.name for c in feedback}
    assert "phi3_find_not_yellow_next" in names


def test_phi3_non_yellow_after_find_is_safe():
    sup = make_supervisor(SORTING)
    assert sup.check_safety(["find_white_cylinder"], "grab_white_cylinder") == []


def test_phi2_done_right_after_place_is_rejected():
    # phi2 = G(place -> X yellow): finishing immediately after placing leaves the
    # "next must be yellow" obligation unmet -> bad prefix.
    sup = make_supervisor(SORTING)
    seq = ["go_to_red_area", "go_to_white_area", "grab_white_cylinder",
           "place_white_cylinder"]
    feedback = sup.check_safety(seq, "done")
    names = {c.name for c in feedback}
    assert "phi2_place_then_yellow" in names


def test_phi2_yellow_after_place_is_safe():
    sup = make_supervisor(SORTING)
    seq = ["go_to_red_area", "go_to_white_area", "grab_white_cylinder",
           "place_white_cylinder"]
    assert sup.check_safety(seq, "go_to_yellow_area") == []
