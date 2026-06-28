"""Cross-Layer Sequence Supervision Mechanism -- task planning layer.

Faithful re-implementation of the task-planning side of:

    Z. Wang, Q. Liu, J. Qin, M. Li,
    "Ensuring Safety in LLM-Driven Robotics: A Cross-Layer Sequence
     Supervision Mechanism", IROS 2024.

The pieces implemented here mirror the paper directly:

    * ``SafetySupervisor``  -- the set of NBAs B_Phi (Sec. IV-A).
        - ``check_safety``   = Algorithm 1 (CheckSafety, task layer).
        - ``find_obstacle``  = Algorithm 2 (FindObstacle, motion layer).
    * ``SymbolicEnv``       -- the state transition system (S, A, T, s0)
                               from Sec. III-B.
    * ``Planner`` family    -- the LLM task planner (Sec. IV-B), either a
                               local-LLM ``OllamaPlanner`` or a deterministic
                               ``ScriptedPlanner`` for offline reproduction.
    * ``run_closed_loop``   -- the closed-loop correction mechanism of Fig. 2.

Formal verification is delegated to the `spot` library (LTL -> Buchi/monitor),
exactly as in RoboGuard's ``synthesis.py``.  The edge-matching idiom
(``spot.contains(edge_cond, letter)``) is borrowed from that proven code.

`spot` is imported lazily: only :class:`SafetySupervisor` needs it, so the
environment / planner / loop can be imported and exercised (with a stub
supervisor) on machines without spot.

Atomic-proposition model (paper Sec. III-B, "A subset of AP"):
each planning step emits exactly ONE action, which is the single atomic
proposition that is true at that step; every other proposition is false.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

# Sentinel action meaning "the task plan is complete".
DONE = "done"


# --------------------------------------------------------------------------- #
#  Safety constraints                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Constraint:
    """A single LTL safety constraint (one phi_i of the set Phi).

    Parameters
    ----------
    name:
        Short identifier, e.g. ``"no_grab_before_open"``.
    ltl:
        The *operative* LTL formula in `spot` syntax over action propositions,
        e.g. ``"(!grab_salmon) U open_microwave"``.
    description:
        Natural-language meaning.  This is the ``SFeedback`` text returned to
        the LLM planner when the constraint is violated (Eq. (3)).
    paper_formula:
        Optional: the formula exactly as printed in the paper, kept for
        traceability.  The paper wraps several constraints in ``G(F(...))``;
        that wrapper is pure liveness and produces no finite "bad prefix",
        so for runtime monitoring we encode the intended *safety* core in
        ``ltl`` and record the printed version here.  See README for details.
    """

    name: str
    ltl: str
    description: str
    paper_formula: Optional[str] = None


# --------------------------------------------------------------------------- #
#  Safety supervisor  (set of NBAs, Sec. IV-A)                                 #
# --------------------------------------------------------------------------- #
class _Monitor:
    """Runtime monitor for one constraint, built from spot.

    A *monitor* (``spot.translate(f, 'monitor')``) is an automaton whose
    states are all accepting and which recognises every finite prefix that is
    **not** a bad prefix -- i.e. a prefix that can still be extended into a word
    satisfying ``f``.  When, after feeding a prefix, no run survives, the prefix
    is a bad prefix and the safety property is violated.

    This is the rigorous prefix-safety test the handoff asked for: unlike the
    "is the final state accepting" approximation, it is correct for ``U``
    (until) and ``X`` (next) operators within the safety fragment.
    """

    def __init__(self, constraint: Constraint, spot_module):
        self.constraint = constraint
        self._spot = spot_module
        # All states of a monitor are accepting; a missing transition marks a
        # bad prefix.  We do not rely on determinism (we keep a set of live
        # states), so plain 'monitor' is enough.
        self.aut = spot_module.translate(constraint.ltl, "monitor")
        self.bdict = self.aut.get_dict()
        self.aps: List[str] = [str(ap) for ap in self.aut.ap()]

    def _letter(self, action: str):
        """The single-action letter as a boolean formula over this monitor's APs.

        Exactly the proposition ``action`` is true (iff it is one of this
        constraint's APs); all other APs of this constraint are false.
        """
        spot = self._spot
        if not self.aps:
            return spot.formula("1")  # true: constraint references no action
        terms = [ap if ap == action else f"!{ap}" for ap in self.aps]
        return spot.formula(" & ".join(terms))

    def survives(self, actions: Sequence[str]) -> bool:
        """Return True iff ``actions`` is **not** a bad prefix for this constraint.

        Powerset simulation over the (possibly nondeterministic) monitor:
        the prefix is safe as long as at least one run remains alive.
        """
        spot = self._spot
        states: Set[int] = {self.aut.get_init_state_number()}
        for action in actions:
            letter = self._letter(action)
            nxt: Set[int] = set()
            for s in states:
                for e in self.aut.out(s):
                    cond = spot.formula(spot.bdd_format_formula(self.bdict, e.cond))
                    # `letter` (a single minterm) satisfies edge `cond`
                    # iff L(letter) is contained in L(cond).
                    if spot.contains(cond, letter):
                        nxt.add(e.dst)
            if not nxt:
                return False  # bad prefix -> constraint violated
            states = nxt
        return True


class SafetySupervisor:
    """The cross-layer safety supervisor B_Phi (Sec. IV-A).

    Parameters
    ----------
    constraints:
        The LTL safety constraints Phi = {phi_1, ..., phi_n}.
    atomic_propositions:
        The full atomic-proposition set AP (here equal to the action space A,
        since A subset of AP).  Used by :meth:`find_obstacle`.
    region_actions:
        Subset of AP corresponding to "go to <region>" actions.  Only these are
        considered candidate "obstacles" in :meth:`find_obstacle`.  Defaults to
        every proposition starting with ``"go_to_"``.
    """

    def __init__(
        self,
        constraints: Sequence[Constraint],
        atomic_propositions: Sequence[str],
        region_actions: Optional[Sequence[str]] = None,
    ):
        import spot  # lazy: only the supervisor needs the formal backend

        self._spot = spot
        self.constraints: List[Constraint] = list(constraints)
        self.atomic_propositions: List[str] = list(atomic_propositions)
        if region_actions is None:
            region_actions = [a for a in self.atomic_propositions if a.startswith("go_to_")]
        self.region_actions: List[str] = list(region_actions)
        self._monitors: List[_Monitor] = [_Monitor(c, spot) for c in self.constraints]

    # -- Algorithm 1 : CheckSafety (task planning layer) -------------------- #
    def check_safety(
        self, action_sequence: Sequence[str], next_action: str
    ) -> List[Constraint]:
        """Algorithm 1.  Evaluate appending ``next_action`` to ``action_sequence``.

        Returns ``SFeedback``: the list of constraints that the candidate
        sequence violates.  An **empty** list means the action is safe
        (Algorithm 1 returns ``True``).
        """
        candidate = list(action_sequence) + [next_action]
        feedback: List[Constraint] = []
        for monitor in self._monitors:
            if not monitor.survives(candidate):
                feedback.append(monitor.constraint)
        return feedback

    # -- Algorithm 2 : FindObstacle (motion planning layer) ---------------- #
    def find_obstacle(
        self, action_sequence: Sequence[str], current_action: str
    ) -> FrozenSet[str]:
        """Algorithm 2.  Regions that must be avoided while executing ``current_action``.

        For every region action ``ap`` other than ``current_action``, if going
        to ``ap`` *now* (given the history ``action_sequence``) would violate a
        constraint, then the region behind ``ap`` is unreachable and is added to
        the obstacle set ``obs_i``.
        """
        obstacles: Set[str] = set()
        for ap in self.region_actions:
            if ap == current_action:
                continue
            if self.check_safety(action_sequence, ap):
                obstacles.add(ap)
        return frozenset(obstacles)

    # -- Helpers ----------------------------------------------------------- #
    def is_sequence_safe(self, action_sequence: Sequence[str]) -> bool:
        """True iff the whole sequence never violated any constraint.

        Bad prefixes are monotone (once dead, always dead), so this equals
        "every monitor survives the full sequence".
        """
        return all(m.survives(action_sequence) for m in self._monitors)

    def validate_sequence(
        self, action_sequence: Sequence[str]
    ) -> Tuple[bool, List[Tuple[str, bool]]]:
        """RoboGuard-style report: overall safety + per-step safety flag.

        At step ``i`` the flag is True iff the prefix ``actions[:i+1]`` is still
        safe under all constraints.
        """
        results: List[Tuple[str, bool]] = []
        seq: List[str] = []
        ok_so_far = True
        for action in action_sequence:
            seq.append(action)
            step_ok = self.is_sequence_safe(seq)
            ok_so_far = ok_so_far and step_ok
            results.append((action, step_ok))
        return ok_so_far, results


# --------------------------------------------------------------------------- #
#  Symbolic environment  (S, A, T, s0 ; Sec. III-B)                            #
# --------------------------------------------------------------------------- #
# A symbolic state is a frozenset of string predicates, e.g.
#   {"robot_at:white_area", "holding:white_cylinder", "open:microwave"}.
State = FrozenSet[str]


@dataclass
class SymbolicEnv:
    """A purely symbolic state-transition system (sim-independent).

    This is the ``T : S x A -> S`` of the paper, kept abstract so it can later
    be swapped for RLBench / MoveIt2 without touching the supervisor or loop.
    """

    initial: State
    transition_fn: Callable[[State, str], State]
    goal_fn: Callable[[State], bool]
    action_space: List[str]
    labels: Dict[str, str] = field(default_factory=dict)

    def initial_state(self) -> State:
        return self.initial

    def available_actions(self, state: State) -> List[str]:
        # The paper allows A to depend on s_env; by default the whole space.
        return list(self.action_space)

    def transition(self, state: State, action: str) -> State:
        return self.transition_fn(state, action)

    def is_goal(self, state: State) -> bool:
        return self.goal_fn(state)

    def label(self, action: str) -> str:
        return self.labels.get(action, action)


# --------------------------------------------------------------------------- #
#  Planners  (LLM task planner, Sec. IV-B)                                     #
# --------------------------------------------------------------------------- #
class Planner:
    """Interface for the LLM-driven task planner."""

    def generate(self, state: State, actions: Sequence[str], task: str) -> str:
        raise NotImplementedError

    def regenerate(
        self,
        state: State,
        actions: Sequence[str],
        task: str,
        rejected: str,
        feedback: Sequence[Constraint],
    ) -> str:
        raise NotImplementedError


class ScriptedPlanner(Planner):
    """A deterministic planner that replays a fixed action queue.

    Both :meth:`generate` and :meth:`regenerate` pop the next scripted action.
    This simulates an LLM that proposes actions one by one (some unsafe); when
    the supervisor rejects one, the "regeneration" simply yields the next
    scripted action.  It lets us reproduce the paper's scenarios with zero
    dependence on an LLM (deterministic safety-rate experiments).
    """

    def __init__(self, plan: Sequence[str]):
        self._queue: List[str] = list(plan)
        self._i = 0

    def _next(self) -> str:
        if self._i >= len(self._queue):
            return DONE
        action = self._queue[self._i]
        self._i += 1
        return action

    def generate(self, state, actions, task) -> str:
        return self._next()

    def regenerate(self, state, actions, task, rejected, feedback) -> str:
        return self._next()


class OllamaPlanner(Planner):
    """LLM task planner backed by a local Ollama OpenAI-compatible endpoint.

    The drop-in pattern from the handoff::

        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        client.chat.completions.create(model="qwen2.5:7b", ...)

    The model and endpoint can be overridden via the ``OLLAMA_MODEL`` and
    ``OLLAMA_BASE_URL`` environment variables.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        from openai import OpenAI  # lazy import

        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1"
        )
        self.temperature = temperature
        self.client = OpenAI(base_url=self.base_url, api_key="ollama")

    # -- prompt construction ---------------------------------------------- #
    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are the task planner of a robot. At each step you choose the "
            "single best next action that makes progress on the task, given the "
            "current environment state and the set of executable actions.\n"
            "Reply with ONLY a JSON object of the form {\"action\": \"<action_id>\"}, "
            "where <action_id> is exactly one of the provided actions. Use the "
            'action "done" when the task is complete. Do not add any other text.'
        )

    @staticmethod
    def _state_text(state: State) -> str:
        if not state:
            return "(empty)"
        return ", ".join(sorted(state))

    def _user_prompt(
        self,
        state: State,
        actions: Sequence[str],
        task: str,
        feedback: Optional[Sequence[Constraint]] = None,
        rejected: Optional[str] = None,
    ) -> str:
        lines = [
            f"Task: {task}",
            f"Current state: {self._state_text(state)}",
            "Executable actions: " + ", ".join(actions),
        ]
        if feedback:
            lines.append("")
            lines.append(
                f"Your previous choice '{rejected}' was REJECTED because it "
                "violates these safety constraints:"
            )
            for c in feedback:
                lines.append(f"  - {c.description}")
            lines.append("Choose a different, safe action.")
        return "\n".join(lines)

    def _ask(self, system: str, user: str, actions: Sequence[str]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        content = resp.choices[0].message.content or ""
        return self._parse_action(content, actions)

    @staticmethod
    def _parse_action(content: str, actions: Sequence[str]) -> str:
        content = content.strip()
        # 1) try strict JSON
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            obj = json.loads(content[start:end])
            cand = str(obj.get("action", "")).strip()
            if cand in actions:
                return cand
        except (ValueError, json.JSONDecodeError):
            pass
        # 2) fall back to substring match (longest id first to avoid prefixes)
        lowered = content.lower()
        for action in sorted(actions, key=len, reverse=True):
            if action.lower() in lowered:
                return action
        # 3) give up -> signal done so the loop can terminate gracefully
        return DONE

    # -- Planner interface ------------------------------------------------- #
    def generate(self, state, actions, task) -> str:
        return self._ask(
            self._system_prompt(), self._user_prompt(state, actions, task), actions
        )

    def regenerate(self, state, actions, task, rejected, feedback) -> str:
        return self._ask(
            self._system_prompt(),
            self._user_prompt(state, actions, task, feedback, rejected),
            actions,
        )


# --------------------------------------------------------------------------- #
#  Closed-loop correction mechanism  (Fig. 2)                                  #
# --------------------------------------------------------------------------- #
@dataclass
class StepRecord:
    """Trace of a single committed planning step."""

    index: int
    committed: str
    rejected: List[str]
    feedback: List[str]  # descriptions of constraints violated by rejected actions
    obstacles: FrozenSet[str]


@dataclass
class ClosedLoopResult:
    plan_tuple: List[Tuple[str, FrozenSet[str]]]  # Pi = (action, obstacle) sequence
    action_sequence: List[str]
    final_state: State
    goal_reached: bool
    safe: bool
    aborted: bool
    steps: List[StepRecord]

    def format(self, env: Optional[SymbolicEnv] = None) -> str:
        def lbl(a: str) -> str:
            return env.label(a) if env else a

        lines = []
        for rec in self.steps:
            obs = ", ".join(sorted(lbl(o) for o in rec.obstacles)) or "None"
            if rec.rejected:
                rej = "; ".join(
                    f"{lbl(r)} [violates: {f}]"
                    for r, f in zip(rec.rejected, rec.feedback)
                )
                lines.append(
                    f"  step {rec.index}: {lbl(rec.committed):<24} "
                    f"obstacle=({obs})   <- discarded: {rej}"
                )
            else:
                lines.append(
                    f"  step {rec.index}: {lbl(rec.committed):<24} obstacle=({obs})"
                )
        status = (
            f"goal_reached={self.goal_reached} safe={self.safe} aborted={self.aborted}"
        )
        return "\n".join(lines + [f"  --> {status}"])


def run_closed_loop(
    env: SymbolicEnv,
    planner: Planner,
    supervisor: SafetySupervisor,
    task: str,
    *,
    max_steps: int = 30,
    max_retries: int = 12,
    stop_at_goal: bool = False,
    verbose: bool = False,
) -> ClosedLoopResult:
    """Run the closed-loop correction mechanism of Fig. 2.

    For each step:
      1. the planner generates an action sigma_i;
      2. ``CheckSafety`` (Alg. 1) tests appending sigma_i to the sequence;
      3. on violation the action is discarded and the planner regenerates with
         the supervisor's ``SFeedback`` (Eq. (3)) -- the unsafe action does NOT
         change the environment state;
      4. once safe, ``FindObstacle`` (Alg. 2) computes obs_i and the task plan
         tuple pi_i = (sigma_i, obs_i) is recorded;
      5. only the safe action updates the environment state.

    Termination follows Fig. 2: the loop runs until the planner emits ``DONE``
    (or ``max_steps`` is hit).  This lets "next"-style obligations (e.g. phi2,
    "go to yellow after placing") complete even once the goal predicate already
    holds.  Set ``stop_at_goal=True`` to stop as soon as the goal is reached
    (useful to keep a real LLM from wandering past completion).
    """
    state = env.initial_state()
    action_sequence: List[str] = []
    plan_tuple: List[Tuple[str, FrozenSet[str]]] = []
    steps: List[StepRecord] = []
    aborted = False

    for index in range(max_steps):
        if stop_at_goal and env.is_goal(state):
            break

        actions = env.available_actions(state)
        action = planner.generate(state, actions, task)

        rejected: List[str] = []
        feedback_text: List[str] = []
        feedback = supervisor.check_safety(action_sequence, action)
        retries = 0
        while feedback and retries < max_retries:
            rejected.append(action)
            feedback_text.append(", ".join(c.name for c in feedback))
            action = planner.regenerate(state, actions, task, action, feedback)
            feedback = supervisor.check_safety(action_sequence, action)
            retries += 1

        if feedback:
            # No safe action could be found within the retry budget.
            steps.append(
                StepRecord(index, action, rejected, feedback_text, frozenset())
            )
            aborted = True
            break

        if action == DONE:
            plan_tuple.append((DONE, frozenset()))
            action_sequence.append(DONE)
            steps.append(StepRecord(index, DONE, rejected, feedback_text, frozenset()))
            break

        obstacles = supervisor.find_obstacle(action_sequence, action)
        plan_tuple.append((action, obstacles))
        action_sequence.append(action)
        steps.append(StepRecord(index, action, rejected, feedback_text, obstacles))

        # Only safe actions alter the environment state (Sec. IV-B).
        state = env.transition(state, action)

        if verbose:
            print(steps[-1])

    committed = [a for a in action_sequence if a != DONE]
    return ClosedLoopResult(
        plan_tuple=plan_tuple,
        action_sequence=action_sequence,
        final_state=state,
        goal_reached=env.is_goal(state),
        safe=supervisor.is_sequence_safe(committed),
        aborted=aborted,
        steps=steps,
    )


def run_open_loop(
    env: SymbolicEnv,
    planner: Planner,
    task: str,
    *,
    max_steps: int = 30,
) -> List[str]:
    """Baseline without the supervisor (Fig. 1(a) / NL-only ablation).

    Every proposed action is committed and executed; there is no CheckSafety
    feedback.  Used to contrast safety rate against :func:`run_closed_loop`.
    """
    state = env.initial_state()
    action_sequence: List[str] = []
    for _ in range(max_steps):
        if env.is_goal(state):
            break
        actions = env.available_actions(state)
        action = planner.generate(state, actions, task)
        if action == DONE:
            break
        action_sequence.append(action)
        state = env.transition(state, action)
    return action_sequence
