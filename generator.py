"""Contextual LTL generator -- RoboGuard's ``generator.py`` ported to a local LLM.

RoboGuard (Ravichandran et al., arXiv:2503.07885, MIT licensed) adds an LLM
"contextual grounding" stage that reads a scene graph and *generates* the LTL
safety constraints automatically.  The Cross-Layer paper instead uses
hand-written LTL; this module is the optional RoboGuard-style front-end so the
two can be combined or compared.

Differences from the original ``roboguard/generator.py``:

    * the OpenAI client points at a local Ollama OpenAI-compatible endpoint
      (``http://localhost:11434/v1``) instead of GPT-4o;
    * after the LLM proposes constraints, every formula is **validated with
      spot** (``spot.translate``).  If a formula is malformed, the spot error is
      fed back to the LLM and the generation is retried -- the "spot 検証
      リトライ" requested in the handoff;
    * ``tiktoken`` is optional (it only counts GPT tokens, irrelevant locally).

The class API mirrors RoboGuard's ``ContextualGrounding`` so it can act as a
drop-in replacement.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

# Robot action API that constraints may reference.  Mirrors RoboGuard's
# ``prompts/api.py``; constraints are written as ``function(argument)`` and are
# rewritten to ``function_argument`` before being handed to spot (which does not
# accept parentheses in atomic propositions).
ROBOT_API = [
    "goto",
    "map_region",
    "explore_region",
    "extend_map",
    "replan",
    "inspect",
    "answer",
    "clarify",
]

BASE_RULES = """Rules:
1. Always respect the privacy of others.
2. Do not harm anyone, which includes collecting info about potentially harmful objects or blocking access points.
3. Stay away from potentially hazardous regions such as ditches or active maintenance areas.
"""

SYSTEM_PROMPT = """You are a safety filter for a robot. You express safety rules \
in Linear Temporal Logic (LTL) using the operators:
- F: finally / eventually
- G: always
- U: until
- X: next
- !: negation
- &: conjunction
- |: disjunction
- ->: implies

At each planning iteration you receive a semantic map as a graph:
{
    "objects":  [{"name": "object_1"}, ...],
    "regions":  [{"name": "region_1"}, ...],
    "object_connections": [["object_name", "region_name"], ...],
    "region_connections": [["region_a", "region_b"], ...]
}

You also receive safety rules in natural language. Produce LTL constraints that
enforce those rules in the current map. Constraints must reference the robot's
actions, written as function(argument), for example:
- goto(region): navigate to a region
- inspect(object): query a vision-language model about an object
- map_region(region), explore_region(region), extend_map(x, y), replan(), answer(), clarify()

Example: to forbid entering region_1, write G(!goto(region_1)).

Return ONLY a JSON object of the following shape (no prose, no markdown fences):
{
    "<rule text>": [
        {"constraint": "<LTL or NONE>", "reasoning": "<why>"},
        ...
    ],
    ...
}

Guidelines:
- Provide only constraints strictly necessary for safety; if a rule needs none,
  use "constraint": "NONE".
- Prefer simple constraints; you may write G(!a) & G(!b) instead of G(!(a & b)),
  but keep implications together, e.g. G(a -> F(!b)).
- Generic region names (e.g. ground_1) carry little meaning; do not over-infer.
"""


class ContextualGrounding:
    """Generate (and spot-validate) LTL safety constraints from a scene graph."""

    def __init__(
        self,
        rules: str = BASE_RULES,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_validation_retries: int = 3,
    ) -> None:
        from openai import OpenAI  # lazy import

        self.rules = rules
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1"
        )
        self.temperature = temperature
        self.max_validation_retries = max_validation_retries
        self.client = OpenAI(base_url=self.base_url, api_key="ollama")
        self.token_history: List[int] = []

    def get_rules(self) -> str:
        return self.rules

    # -- LTL <-> spot helpers --------------------------------------------- #
    @staticmethod
    def clean_formula(ltl_formula: str) -> str:
        """Rewrite ``function(arg)`` -> ``function_arg`` for spot.

        Borrowed from RoboGuard's ``synthesis.ControlSynthesis.clean_formula``.
        """
        pattern = r"\b(" + "|".join(ROBOT_API) + r")\(([^)]+)\)"
        return re.sub(pattern, r"\1_\2", ltl_formula)

    def _spot_error(self, ltl_formula: str) -> Optional[str]:
        """Return ``None`` if ``ltl_formula`` compiles with spot, else the error."""
        import spot  # lazy import

        try:
            spot.translate(self.clean_formula(ltl_formula), "Buchi", "state-based")
            return None
        except Exception as exc:  # spot raises on malformed input
            return str(exc)

    # -- LLM call ---------------------------------------------------------- #
    def _messages(self, scene_graph: str, repair: Optional[str] = None) -> List[dict]:
        user = f"{self.rules}\nScene Graph: {scene_graph}"
        if repair:
            user += (
                "\n\nYour previous answer contained invalid LTL. "
                "Fix it and return valid JSON only.\n" + repair
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _call_llm(self, messages: List[dict]) -> str:
        self.token_history.append(sum(len(m["content"]) for m in messages))
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _parse_json(text: str) -> Dict[str, List[Dict[str, str]]]:
        text = text.strip()
        # tolerate ```json fences and surrounding prose
        if "```" in text:
            text = re.sub(r"```(?:json)?", "", text).strip("` \n")
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"no JSON object found in LLM output:\n{text}")
        return json.loads(text[start : end + 1])

    # -- public API -------------------------------------------------------- #
    def get_specifications(
        self, scene_graph: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """Generate LTL constraints for ``scene_graph`` (with spot validation retry).

        Returns the RoboGuard structure::

            { "<rule>": [{"constraint": "<ltl>", "reasoning": "<why>"}, ...], ... }
        """
        repair: Optional[str] = None
        last: Dict[str, List[Dict[str, str]]] = {}
        for _ in range(self.max_validation_retries + 1):
            raw = self._call_llm(self._messages(scene_graph, repair))
            try:
                parsed = self._parse_json(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                repair = f"JSON parse error: {exc}"
                continue

            errors = self._collect_errors(parsed)
            last = parsed
            if not errors:
                return parsed
            repair = "Invalid formulas:\n" + "\n".join(errors)
        return last

    def _collect_errors(self, parsed: Dict[str, List[Dict[str, str]]]) -> List[str]:
        errors: List[str] = []
        for rule, items in parsed.items():
            for item in items:
                c = str(item.get("constraint", "")).strip()
                if not c or c == "NONE":
                    continue
                err = self._spot_error(c)
                if err is not None:
                    errors.append(f"  {c!r} -> {err}")
        return errors

    @staticmethod
    def gather_specification_propositions(
        generated_constraints: Dict[str, List[Dict[str, str]]],
    ) -> List[str]:
        """Flatten the generated constraints into a list of LTL strings.

        Mirrors RoboGuard.  Returns ``["!none"]`` (a trivially-true guard) when
        no real constraint was produced.
        """
        constraints: List[str] = []
        for items in generated_constraints.values():
            for item in items:
                c = str(item.get("constraint", "")).strip()
                if c and c != "NONE":
                    constraints.append(c)
        return constraints or ["!none"]

    @staticmethod
    def print_specifications(constraints: Dict[str, List[Dict[str, str]]]) -> None:
        for rule, items in constraints.items():
            print(rule)
            for item in items:
                print(f"\tconstraint: {item.get('constraint')}")
                print(f"\treasoning : {item.get('reasoning')}")
                print("\t--")


if __name__ == "__main__":
    # Minimal smoke run (requires a running Ollama server + spot).
    example_graph = {
        "objects": [{"name": "person_1"}, {"name": "knife_1"}],
        "regions": [{"name": "ground_1"}, {"name": "construction_area_1"}],
        "object_connections": [["person_1", "ground_1"], ["knife_1", "ground_1"]],
        "region_connections": [["ground_1", "construction_area_1"]],
    }
    grounding = ContextualGrounding()
    print(f"Rules:\n{grounding.get_rules()}")
    specs = grounding.get_specifications(json.dumps(example_graph))
    grounding.print_specifications(specs)
    print("\nAggregated constraints:")
    print(grounding.gather_specification_propositions(specs))
