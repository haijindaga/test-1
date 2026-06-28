# Cross-Layer Sequence Supervision — local-LLM reproduction

A faithful, runnable reproduction of the **task-planning side** of

> Z. Wang, Q. Liu, J. Qin, M. Li,
> *Ensuring Safety in LLM-Driven Robotics: A Cross-Layer Sequence Supervision
> Mechanism*, **IROS 2024**.

The safety guarantee comes from **formal methods (`spot`)**, not from the LLM, so
the GPT-4 task planner of the paper can be swapped for a small **local model via
Ollama** without weakening the verification. Parts of the LTL/automaton plumbing
are adapted from the MIT-licensed
[RoboGuard](https://arxiv.org/abs/2503.07885) (`synthesis.py`, `generator.py`).

> **Please cite both papers** (Cross-Layer *and* RoboGuard) in any write-up — a
> README acknowledgement is not sufficient academically.

---

## What is implemented

| Paper element | Where |
|---|---|
| Safety supervisor `B_Φ` (set of NBAs, Sec. IV-A) | `task_layer.py::SafetySupervisor` |
| **Algorithm 1 — CheckSafety** (task layer) | `SafetySupervisor.check_safety` |
| **Algorithm 2 — FindObstacle** (motion layer) | `SafetySupervisor.find_obstacle` |
| State-transition system `(S, A, T, s₀)` (Sec. III-B) | `task_layer.py::SymbolicEnv` |
| LLM task planner (Sec. IV-B, Eq. (1)/(3)) | `task_layer.py::OllamaPlanner` |
| Closed-loop correction (Fig. 2) | `task_layer.py::run_closed_loop` |
| Table I / Fig. 3 "salmon" scenario | `scenarios.py::SALMON` |
| Table II / Fig. 4 "sorting" scenario | `scenarios.py::SORTING` |
| RoboGuard-style LTL generator (local LLM + spot retry) | `generator.py` |

`spot` is imported lazily, so the environment / planner / loop can be used (with a
stub supervisor) even where spot is unavailable; only the supervisor needs it.

---

## Quick start (Ubuntu, clone → run)

`spot` is **not** a pip package (the PyPI project named `spot` is unrelated).
Use conda-forge (recommended) or apt.

### Option A — conda (recommended)

```bash
git clone <your-repo-url> crosslayer && cd crosslayer
conda env create -f environment.yml      # installs python, spot, openai, pytest
conda activate crosslayer

pytest -q                                # run the test suite (needs spot only)
python examples/run_salmon.py            # Table I / Fig. 3
python examples/run_sorting.py           # Table II / Fig. 4
```

### Option B — apt + venv

```bash
git clone <your-repo-url> crosslayer && cd crosslayer
sudo apt-get install python3-spot        # spot from the official apt repo
# --system-site-packages lets the venv see the apt-installed python3-spot:
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements.txt

python -c "import spot; print('spot', spot.version())"   # sanity check
pytest -q
python examples/run_salmon.py
```

The examples run with a **deterministic `ScriptedPlanner`** by default (no LLM
required) and print a safety contrast against the no-supervisor baseline.

### Using a local LLM (Ollama)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b        # fits 8 GB VRAM; or gemma3:12b / gemma3:27b
ollama serve &

python examples/run_salmon.py --llm
python examples/run_sorting.py --llm
```

Configure via env vars: `OLLAMA_MODEL` (default `qwen2.5:7b`) and
`OLLAMA_BASE_URL` (default `http://localhost:11434/v1`).

---

## How CheckSafety works (and a note on the paper's formulas)

A finite action prefix is checked against each constraint with a **spot monitor**
(`spot.translate(φ, "monitor")`). A monitor recognises exactly the prefixes that
are *not* "bad prefixes" — i.e. prefixes that can still be extended to satisfy φ.
When no run of the monitor survives the prefix, the property is violated. This is
the rigorous prefix-safety test (correct for `U` and `X`), generalising the
"is the final state accepting?" approximation that only works for `G(!…)` rules.

The single-action-per-step alphabet of the paper (`A ⊆ AP`) is honoured: at each
step exactly one proposition (the chosen action) is true, all others false.

**Formula caveat.** The paper prints constraints wrapped in `G(F(…))`, e.g.
`G(F(¬Grab_salmon U Open_microwave))`. A `G(F(…))` formula is pure *liveness* and
has **no** finite bad prefix, so taken literally it could never trigger
CheckSafety. We therefore encode the intended **safety core** in each
`Constraint.ltl` (e.g. `(!grab_salmon) U open_microwave`) and keep the printed
version in `Constraint.paper_formula` for traceability. The implemented
constraints reproduce the figures exactly — e.g. for the sorting task the
computed obstacle sets match Fig. 4 (`obs₀ = {white area}`,
`obs₂ = {white area, yellow area}`).

---

## Layout

```
task_layer.py     Cross-Layer task layer: supervisor, env, planners, closed loop
scenarios.py      SALMON (Table I) and SORTING (Table II) scenarios
generator.py      RoboGuard-style LTL generator on a local LLM + spot validation
examples/         runnable demos (ScriptedPlanner by default, --llm for Ollama)
tests/            pytest suite
  test_env.py            env + loop control flow      (no spot needed)
  test_check_safety.py   Algorithm 1                  (needs spot)
  test_find_obstacle.py  Algorithm 2 / Fig. 4 obstacles (needs spot)
  test_closed_loop.py    full loop + safety contrast  (needs spot)
environment.yml   conda environment (includes spot)
requirements.txt  pip deps (openai, pytest) — spot installed separately
```

---

## Roadmap (from the project handoff)

- [x] CheckSafety + unit tests (rigorous prefix monitoring via spot).
- [x] Table II scenarios: action space, `SymbolicEnv`, safety-rate contrast.
- [ ] Motion layer: feed `FindObstacle` obstacles to a real planner
      (RLBench/CoppeliaSim or ROS2 Humble + MoveIt2) as collision objects.
- [ ] A third, novel safety layer (the paper's extension / contribution).
