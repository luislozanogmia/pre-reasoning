"""Pre-Reasoning V4: external structural analysis for a second forward pass.

The calling AI interprets the user's language and writes the structured form.
This module validates that form, binds its entities to the checkpoint's learned
operator vocabulary, runs neural operations, restores the original entities,
and returns a structural trace for the AI's next forward pass.
"""
from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

from .inference import MODEL_PARAMS, generate_completion, load_model

_ENTITY_A = "amber project one"
_ENTITY_B = "blue relay two"
_ENTITY_C = "coral gate three"

FORM_TEMPLATE = """DEPENDENCIES
<dependent> depends on <prerequisite>.

CONFLICTS
<entity> conflicts with <entity>.

REQUIREMENTS
<entity> must be at least <number>.
<entity> must be at most <number>.
<entity> must be exactly <number>.

CONDITIONALS
If <condition> passes, then <consequence> can proceed.
If <condition> passes, then <consequence> can proceed, otherwise <alternative> must proceed.
"""

_SECTION_NAMES = {
    "dependencies",
    "conflicts",
    "requirements",
    "conditionals",
}

MIN_FORM_BLOCKS = 5
SHORT_FORM_ALARM = "PRE-REASONING ALARM: INSUFFICIENT STRUCTURED BLOCKS"
FOCUS_INTERVAL_MINUTES = 10
FOCUS_REMINDER = (
    "Pause active work and run a fresh pre-reasoning pass. Re-read the original "
    "problem, identify what changed, and express the currently supported "
    "dependencies, conflicts, numeric requirements, and conditional branches "
    "in at least five valid structured blocks. Review the returned root blockers "
    "and derived assumptions before choosing the next move."
)


def _focus_scheduler_request(interval_minutes: float) -> dict:
    cadence = f"{interval_minutes:g}"
    task_prompt = (
        "Run a Focus Mode pre-reasoning pulse for the substantive work currently "
        "active in this chat. Re-read the original goal and latest work state. "
        "Translate only supported facts into at least five valid Pre-Reasoning "
        "blocks covering the applicable dependencies, conflicts, numeric "
        "requirements, and conditional branches. Run a fresh pulse through the "
        "local 1M bundle, read its trace, and reconsider the next move before "
        "continuing. Keep the pulse internal unless it changes the plan, reveals "
        "a blocker or conflict, completes the work, or requires user input. If no "
        "substantive work is active, stay quiet."
    )
    return {
        "action": "create_or_update_recurring_task",
        "destination": "current_chat",
        "schedule": f"every {cadence} minutes",
        "interval_minutes": interval_minutes,
        "instruction": (
            "Use the host's scheduling or automation capability now to create or "
            f"update one recurring task in this current chat every {cadence} "
            "minutes. Reuse an existing matching Focus Mode task instead of "
            "creating a duplicate. Pause or remove it when the work ends."
        ),
        "task_prompt": task_prompt,
    }


FOCUS_SCHEDULER_PROMPT = _focus_scheduler_request(
    float(FOCUS_INTERVAL_MINUTES)
)["instruction"]


class FormError(ValueError):
    """Hard failure carrying the submitted form so the AI can correct it."""

    def __init__(self, message: str, *, form_text: str | None = None):
        super().__init__(message)
        self.form_text = form_text

    @property
    def form(self) -> str | None:
        """The exact submitted form, available for correction and retry."""
        return self.form_text


class NeuralContractError(RuntimeError):
    """Raised when the checkpoint violates a required learned operator contract."""


def _clean(value: str) -> str:
    """Normalize whitespace and punctuation without destroying entity spelling."""
    return " ".join(value.strip(" \t\r\n.,:;").split())


def _form_statements(text: str) -> Iterable[str]:
    body = re.sub(
        r"return\s+the\s+complete\s+reasoning\s+structure\s*\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    for fragment in re.split(r";|(?<!\d)\.|\.(?!\d)|[\n]+", body):
        statement = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", fragment).strip()
        if not statement:
            continue
        label = statement.rstrip(":").strip().lower()
        if label in _SECTION_NAMES or label == "none":
            continue
        yield statement


def _parse_form(text: str):
    """Parse the explicit AI-authored form; never infer meaning from raw prose."""
    dependencies: set[tuple[str, str]] = set()
    conflicts: set[tuple[str, str]] = set()
    requirements: list[list[str]] = []
    conditionals: list[list[str]] = []
    rejected: list[str] = []

    for statement in _form_statements(text):
        conditional = re.fullmatch(
            r"if\s+(.+?)\s+passes,\s*then\s+(.+?)\s+can proceed"
            r"(?:,\s*otherwise\s+(.+?)\s+must proceed)?",
            statement,
            re.IGNORECASE,
        )
        if conditional:
            conditionals.append(
                [
                    _clean(conditional.group(1)),
                    _clean(conditional.group(2)),
                    _clean(conditional.group(3)) if conditional.group(3) else "",
                ]
            )
            continue

        requirement = re.fullmatch(
            r"(.+?)\s+must be\s+(at least|at most|exactly)\s+"
            r"(-?\d+(?:\.\d+)?)",
            statement,
            re.IGNORECASE,
        )
        if requirement:
            requirements.append(
                [
                    _clean(requirement.group(1)),
                    {
                        "at least": "GEQ",
                        "at most": "LEQ",
                        "exactly": "EQ",
                    }[requirement.group(2).lower()],
                    requirement.group(3),
                ]
            )
            continue

        conflict = re.fullmatch(
            r"(.+?)\s+conflicts with\s+(.+)", statement, re.IGNORECASE
        )
        if conflict:
            conflicts.add((_clean(conflict.group(1)), _clean(conflict.group(2))))
            continue

        dependency = re.fullmatch(
            r"(.+?)\s+depends on\s+(.+)", statement, re.IGNORECASE
        )
        if dependency:
            dependencies.add(
                (_clean(dependency.group(1)), _clean(dependency.group(2)))
            )
            continue

        rejected.append(statement)

    if rejected:
        preview = "; ".join(repr(item) for item in rejected[:3])
        more = f" (+{len(rejected) - 3} more)" if len(rejected) > 3 else ""
        raise FormError(
            "Input is not a valid pre-reasoning form. "
            f"Unrecognized statement(s): {preview}{more}. "
            "Call get_form() for the exact contract and retry the returned form.",
            form_text=text,
        )
    return dependencies, conflicts, requirements, conditionals


def _adjacency(dependencies: Iterable[tuple[str, str]]):
    graph: dict[str, set[str]] = defaultdict(set)
    for dependent, prerequisite in dependencies:
        graph[dependent].add(prerequisite)
    return graph


def _root_blockers(dependencies: set[tuple[str, str]]) -> list[str]:
    dependents = {source for source, _ in dependencies}
    return sorted({target for _, target in dependencies if target not in dependents})


def _unlock_plan(
    dependencies: set[tuple[str, str]],
) -> tuple[list[dict], list[dict]]:
    """Render an exact topological schedule from model-verified direct edges."""
    nodes = {entity for edge in dependencies for entity in edge}
    prerequisites: dict[str, set[str]] = {node: set() for node in nodes}
    dependents: dict[str, set[str]] = defaultdict(set)
    for dependent, prerequisite in dependencies:
        prerequisites[dependent].add(prerequisite)
        dependents[prerequisite].add(dependent)

    remaining = {node: set(values) for node, values in prerequisites.items()}
    ready = sorted(node for node, values in remaining.items() if not values)
    sequence: list[dict] = []
    parallel: list[dict] = []
    emitted: set[str] = set()
    step = 1
    while ready:
        layer = [node for node in ready if node not in emitted]
        if not layer:
            break
        if len(layer) > 1:
            parallel.append({"step": step, "entities": layer})
        for node in layer:
            sequence.append({"step": step, "entity": node, "name": node})
            emitted.add(node)
        next_ready: set[str] = set()
        for node in layer:
            for dependent in dependents.get(node, ()):
                remaining[dependent].discard(node)
                if not remaining[dependent] and dependent not in emitted:
                    next_ready.add(dependent)
        ready = sorted(next_ready)
        step += 1
    return sequence, parallel


def form_spec() -> dict:
    """Return the stable contract intended for an outer AI."""
    return {
        "engine": "Pre-Reasoning V4",
        "version": V4ReasoningEngine.VERSION,
        "purpose": (
            "The calling AI externalizes its interpretation, receives an "
            "independent structural trace, and uses that trace in a new forward pass."
        ),
        "workflow": [
            "Interpret the user's language yourself.",
            "Write only facts supported by the user's context in the structured form.",
            "Call analyze_form(form_text).",
            "Read the returned trace alongside the original problem before answering.",
        ],
        "rules": [
            "Preserve entity names exactly and use them consistently.",
            "Use one structured statement per line.",
            "Do not send raw prose, explanations, or guessed relationships.",
            "Omit an uncertain relationship instead of inventing one.",
        ],
        "minimum_blocks": MIN_FORM_BLOCKS,
        "minimum_block_alarm": SHORT_FORM_ALARM,
        "template": FORM_TEMPLATE,
        "families": {
            "dependency": "<dependent> depends on <prerequisite>.",
            "conflict": "<entity> conflicts with <entity>.",
            "requirement_geq": "<entity> must be at least <number>.",
            "requirement_leq": "<entity> must be at most <number>.",
            "requirement_eq": "<entity> must be exactly <number>.",
            "conditional": (
                "If <condition> passes, then <consequence> can proceed, "
                "otherwise <alternative> must proceed."
            ),
        },
    }


class V4ReasoningEngine:
    """External pre-reasoner backed by the strictly loaded 1M checkpoint."""

    VERSION = "4.0.1"
    params = MODEL_PARAMS

    def __init__(self, checkpoint_path=None, device: str = "auto"):
        self.model, self.config, self.model_meta = load_model(
            checkpoint_path, device=device
        )
        self.device = str(self.model.get_device())
        self.checkpoint_path = self.model_meta["checkpoint"]
        self._inference_cache: dict[tuple[str, str], str] = {}
        self._model_calls: list[dict[str, str]] = []
        self._operation_counts: Counter[str] = Counter()
        self._execution_counts: Counter[str] = Counter()
        self._last_result: dict | None = None

    @property
    def mode(self) -> str:
        return "form"

    def engine_info(self) -> dict:
        return {
            "engine": "Pre-Reasoning V4",
            "version": self.VERSION,
            "mode": self.mode,
            "model": self.model_meta["variant_id"],
            "params": self.params,
            "checkpoint": self.checkpoint_path,
            "device": self.device,
            "strict_checkpoint_load": True,
        }

    def get_form(self) -> dict:
        return form_spec()

    def format_for_narrator(self) -> str:
        if self._last_result is None:
            return "--- PRE-REASONING TRACE (V4) ---\nNo form has been analyzed yet."
        return self._last_result["trace"]

    def _minimum_form_alarm(self, form_text: str, block_count: int) -> dict:
        reprompt = (
            f"Reprompt with at least {MIN_FORM_BLOCKS} valid structured blocks. "
            "Use the attached form template and call analyze_form(form_text) again."
        )
        trace = "\n".join(
            [
                "--- PRE-REASONING ALARM ---",
                SHORT_FORM_ALARM,
                (
                    f"Only {block_count} valid structured block(s) were submitted; "
                    f"at least {MIN_FORM_BLOCKS} are required."
                ),
                "",
                "REPROMPT REQUIRED:",
                reprompt,
                "",
                "ATTACHED FORM TEMPLATE:",
                FORM_TEMPLATE.rstrip(),
            ]
        )
        return {
            "status": "REPROMPT_REQUIRED",
            "alarm": SHORT_FORM_ALARM,
            "message": reprompt,
            "reprompt": reprompt,
            "block_count": block_count,
            "minimum_blocks": MIN_FORM_BLOCKS,
            "submitted_form": form_text,
            "attached_form": FORM_TEMPLATE,
            "trace": trace,
            "dependencies": [],
            "conflicts": [],
            "requirements": [],
            "conditionals": [],
            "derived": [],
            "root_blockers": [],
            "cycle": False,
            "cycle_nodes": [],
            "unlock_sequence": [],
            "parallel_work": [],
            "blocks": [],
            "derived_blocks": [],
            "n_blocks": block_count,
            "n_derived_blocks": 0,
            "params": self.params,
            "model": "pre-reasoning-1m",
            "version": self.VERSION,
            "mode": self.mode,
            "neural_verified": False,
            "neural_checks": [],
            "neural_operations": {},
            "neural_model_calls": 0,
            "neural_cached_operations": 0,
            "strict_checkpoint_load": True,
            "inference_ms": 0.0,
            "neural_enriched": False,
            "grounding_level": "reprompt_required",
            "has_cycle": False,
        }

    def _complete(self, capability: str, prompt: str) -> str:
        """Run one normalized learned operation through the checkpoint."""
        self._operation_counts[capability] += 1
        cache_key = (capability, prompt)
        cache_hit = cache_key in self._inference_cache
        if not cache_hit:
            self._inference_cache[cache_key] = generate_completion(self.model, prompt)
            self._execution_counts["model"] += 1
        else:
            self._execution_counts["cache"] += 1
        completion = self._inference_cache[cache_key]
        if not any(call["capability"] == capability for call in self._model_calls):
            self._model_calls.append(
                {
                    "capability": capability,
                    "prompt": prompt,
                    "completion": completion,
                    "source": "cache" if cache_hit else "model",
                }
            )
        return completion

    @staticmethod
    def _mapped_entity(value: str, bindings: dict[str, str]) -> str:
        try:
            return bindings[value]
        except KeyError as exc:
            raise NeuralContractError(
                f"Model emitted an unbound placeholder: {value!r}"
            ) from exc

    def _model_dependency(self, dependent: str, prerequisite: str) -> tuple[str, str]:
        prompt = f"{_ENTITY_A} needs {_ENTITY_B}. answer"
        completion = self._complete("dependency", prompt)
        match = re.fullmatch(
            rf"\s*cycle\s+(yes|no)\.\s+therefore\s+"
            rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)})\s+needs\s+"
            rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)})\.",
            completion,
        )
        if not match or match.group(1) != "no":
            raise NeuralContractError(f"Invalid dependency completion: {completion!r}")
        bindings = {_ENTITY_A: dependent, _ENTITY_B: prerequisite}
        return (
            self._mapped_entity(match.group(2), bindings),
            self._mapped_entity(match.group(3), bindings),
        )

    def _model_assumption(
        self, source: str, middle: str, target: str
    ) -> tuple[str, str] | None:
        prompt = (
            f"{_ENTITY_A} needs {_ENTITY_B}. and "
            f"{_ENTITY_B} needs {_ENTITY_C}. derive assumptions"
        )
        completion = self._complete("assumption", prompt)
        if re.fullmatch(r"\s*assumption\s+no\.\s+therefore\s+none\.", completion):
            return None
        match = re.fullmatch(
            rf"\s*assumption\s+yes\.\s+therefore\s+"
            rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)}|{re.escape(_ENTITY_C)})"
            rf"\s+needs\s+"
            rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)}|{re.escape(_ENTITY_C)})\.",
            completion,
        )
        if not match:
            raise NeuralContractError(f"Invalid assumption completion: {completion!r}")
        bindings = {_ENTITY_A: source, _ENTITY_B: middle, _ENTITY_C: target}
        return (
            self._mapped_entity(match.group(1), bindings),
            self._mapped_entity(match.group(2), bindings),
        )

    def _model_cycle(self, left: str, right: str) -> bool:
        prompt = (
            f"{_ENTITY_A} needs {_ENTITY_B}. and "
            f"{_ENTITY_B} needs {_ENTITY_A}. answer"
        )
        completion = self._complete("cycle", prompt)
        match = re.fullmatch(
            rf"\s*cycle\s+(yes|no)\.\s+therefore\s+"
            rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)})\s+needs\s+"
            rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)})\.",
            completion,
        )
        if not match:
            raise NeuralContractError(f"Invalid cycle completion: {completion!r}")
        bindings = {_ENTITY_A: left, _ENTITY_B: right}
        self._mapped_entity(match.group(2), bindings)
        self._mapped_entity(match.group(3), bindings)
        return match.group(1) == "yes"

    def _model_conflict(self, left: str, right: str) -> tuple[str, str] | None:
        prompt = f"{_ENTITY_A} conflicts with {_ENTITY_B}. answer"
        completion = self._complete("conflict", prompt)
        if re.fullmatch(r"\s*conflict\s+no\.\s+therefore\s+none\.", completion):
            return None
        match = re.fullmatch(
            rf"\s*conflict\s+yes\.\s+therefore\s+"
            rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)})\s+conflicts with\s+"
            rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)})\.",
            completion,
        )
        if not match:
            raise NeuralContractError(f"Invalid conflict completion: {completion!r}")
        bindings = {_ENTITY_A: left, _ENTITY_B: right}
        return (
            self._mapped_entity(match.group(1), bindings),
            self._mapped_entity(match.group(2), bindings),
        )

    def _model_requirement(
        self, entity: str, operator: str, original_value: str
    ) -> list[str] | None:
        phrases = {
            "GEQ": ("at least", "17", _ENTITY_A),
            "LEQ": ("at most", "23", _ENTITY_B),
            "EQ": ("exactly", "31", _ENTITY_C),
        }
        phrase, canonical_value, placeholder = phrases[operator]
        capability = f"requirement_{operator.lower()}"
        prompt = f"{placeholder} must be {phrase} {canonical_value}. answer"
        completion = self._complete(capability, prompt)
        if re.fullmatch(r"\s*requirement\s+no\.\s+therefore\s+none\.", completion):
            return None
        match = re.fullmatch(
            rf"\s*requirement\s+yes\.\s+therefore\s+"
            rf"({re.escape(placeholder)})\s+must be\s+"
            r"(at least|at most|exactly)\s+(-?\d+(?:\.\d+)?)\.",
            completion,
        )
        if not match:
            raise NeuralContractError(f"Invalid requirement completion: {completion!r}")
        decoded_operator = {
            "at least": "GEQ",
            "at most": "LEQ",
            "exactly": "EQ",
        }[match.group(2)]
        if decoded_operator != operator or match.group(3) != canonical_value:
            raise NeuralContractError(f"Incorrect requirement completion: {completion!r}")
        return [entity, decoded_operator, original_value]

    def _model_conditional(
        self, condition: str, consequence: str, otherwise: str
    ) -> list[str] | None:
        if otherwise:
            capability = "conditional_else"
            prompt = (
                f"if {_ENTITY_A} passes, then {_ENTITY_B} can proceed, "
                f"otherwise {_ENTITY_C} must proceed. answer"
            )
            pattern = (
                rf"\s*conditional\s+yes\.\s+therefore\s+if\s+"
                rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)}|{re.escape(_ENTITY_C)})"
                rf"\s+then\s+"
                rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)}|{re.escape(_ENTITY_C)})"
                rf"\s+else\s+"
                rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)}|{re.escape(_ENTITY_C)})\."
            )
        else:
            capability = "conditional"
            prompt = f"if {_ENTITY_A} passes, then {_ENTITY_B} can proceed. answer"
            pattern = (
                rf"\s*conditional\s+yes\.\s+therefore\s+if\s+"
                rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)})\s+then\s+"
                rf"({re.escape(_ENTITY_A)}|{re.escape(_ENTITY_B)})\."
            )
        completion = self._complete(capability, prompt)
        if re.fullmatch(r"\s*conditional\s+no\.\s+therefore\s+none\.", completion):
            return None
        match = re.fullmatch(pattern, completion)
        if not match:
            raise NeuralContractError(f"Invalid conditional completion: {completion!r}")
        bindings = {
            _ENTITY_A: condition,
            _ENTITY_B: consequence,
            _ENTITY_C: otherwise,
        }
        decoded = [
            self._mapped_entity(match.group(1), bindings),
            self._mapped_entity(match.group(2), bindings),
            "",
        ]
        if otherwise:
            decoded[2] = self._mapped_entity(match.group(3), bindings)
        return decoded

    def _model_dependency_graph(
        self, candidates: set[tuple[str, str]]
    ) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[str]]:
        direct = {
            self._model_dependency(dependent, prerequisite)
            for dependent, prerequisite in sorted(candidates)
        }
        known = set(direct)
        derived: set[tuple[str, str]] = set()
        cycle_nodes: set[str] = set()
        while True:
            additions: set[tuple[str, str]] = set()
            snapshot = sorted(known)
            for source, middle in snapshot:
                if source == middle:
                    continue
                for next_source, target in snapshot:
                    if next_source != middle or middle == target:
                        continue
                    if source == target:
                        if self._model_cycle(source, middle):
                            cycle_nodes.update((source, middle))
                        continue
                    if (source, target) in known:
                        continue
                    conclusion = self._model_assumption(source, middle, target)
                    if conclusion is not None:
                        additions.add(conclusion)
            additions -= known
            if not additions:
                break
            known.update(additions)
            derived.update(additions)
        return direct, derived, cycle_nodes

    def analyze_form(
        self, form_text: str, *, _enforce_minimum: bool = True
    ) -> dict:
        """Analyze the structured form written by the calling AI."""
        if not isinstance(form_text, str):
            raise TypeError("form_text must be a string")
        started = time.perf_counter()
        parsed = _parse_form(form_text)
        parsed_dependencies, parsed_conflicts, parsed_requirements, parsed_conditionals = parsed
        block_count = (
            len(parsed_dependencies)
            + len(parsed_conflicts)
            + len(parsed_requirements)
            + len(parsed_conditionals)
        )
        if _enforce_minimum and block_count < MIN_FORM_BLOCKS:
            alarm = self._minimum_form_alarm(form_text, block_count)
            alarm["inference_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return alarm
        self._model_calls = []
        self._operation_counts = Counter()
        self._execution_counts = Counter()

        dependencies, derived, cycle_nodes = self._model_dependency_graph(
            parsed_dependencies
        )
        conflicts = {
            decoded
            for left, right in sorted(parsed_conflicts)
            if (decoded := self._model_conflict(left, right)) is not None
        }
        requirements = [
            decoded
            for entity, operator, value in parsed_requirements
            if (decoded := self._model_requirement(entity, operator, value)) is not None
        ]
        conditionals = [
            decoded
            for condition, consequence, otherwise in parsed_conditionals
            if (
                decoded := self._model_conditional(condition, consequence, otherwise)
            )
            is not None
        ]
        unlock_sequence, parallel_work = _unlock_plan(dependencies)
        root_blockers = _root_blockers(dependencies)
        blocks = self._build_blocks(
            dependencies, conflicts, requirements, conditionals, derived
        )
        derived_blocks = [block for block in blocks if block.get("derived")]
        checks = [call["capability"] for call in self._model_calls]
        result = {
            "dependencies": sorted([list(pair) for pair in dependencies]),
            "conflicts": sorted([list(pair) for pair in conflicts]),
            "requirements": sorted(requirements),
            "conditionals": sorted(conditionals),
            "derived": sorted([list(pair) for pair in derived]),
            "root_blockers": root_blockers,
            "cycle": bool(cycle_nodes),
            "cycle_nodes": sorted(cycle_nodes),
            "unlock_sequence": unlock_sequence,
            "parallel_work": parallel_work,
            "blocks": blocks,
            "derived_blocks": derived_blocks,
            "n_blocks": len(blocks),
            "n_derived_blocks": len(derived_blocks),
            "derived_assumptions": [
                {"assuming": source, "premise": target}
                for source, target in sorted(derived)
            ],
            "params": self.params,
            "model": "pre-reasoning-1m",
            "version": self.VERSION,
            "mode": self.mode,
            "neural_verified": bool(self._model_calls),
            "neural_checks": checks,
            "neural_operations": dict(sorted(self._operation_counts.items())),
            "neural_model_calls": self._execution_counts["model"],
            "neural_cached_operations": self._execution_counts["cache"],
            "strict_checkpoint_load": True,
            "inference_ms": round((time.perf_counter() - started) * 1000, 1),
            "neural_enriched": True,
            "grounding_level": "grounding" if blocks else "empty_form",
            "has_cycle": bool(cycle_nodes),
            "derive_meta": {
                "strategy": "neural_compositional_windows",
                "n_edges": len(dependencies),
                "n_entities": len({entity for pair in dependencies for entity in pair}),
                "edge_source": "decoded_model_output",
                "form_source": "calling_ai",
            },
        }
        result["trace"] = self._render_trace(result)
        self._last_result = result
        return result

    @staticmethod
    def _build_blocks(
        dependencies, conflicts, requirements, conditionals, derived
    ) -> list[dict]:
        blocks: list[dict] = []

        def append(block: dict) -> None:
            item = dict(block)
            item["index"] = len(blocks) + 1
            item.setdefault("confidence", 1.0)
            blocks.append(item)

        for dependent, prerequisite in sorted(dependencies):
            append(
                {
                    "family": "dependency",
                    "entities": [dependent, prerequisite],
                    "roles": {"dependent": dependent, "prerequisite": prerequisite},
                    "source": f"{dependent} depends on {prerequisite}",
                }
            )
        for left, right in sorted(conflicts):
            append(
                {
                    "family": "conflict",
                    "entities": [left, right],
                    "roles": {"left": left, "right": right},
                    "source": f"{left} conflicts with {right}",
                }
            )
        for entity, operator, value in sorted(requirements):
            append(
                {
                    "family": "requirement",
                    "entities": [entity],
                    "roles": {"entity": entity, "operator": operator, "value": value},
                    "source": f"{entity} {operator} {value}",
                }
            )
        for condition, consequence, otherwise in sorted(conditionals):
            entities = [condition, consequence] + ([otherwise] if otherwise else [])
            append(
                {
                    "family": "conditional",
                    "entities": entities,
                    "roles": {
                        "condition": condition,
                        "consequence": consequence,
                        "otherwise": otherwise,
                    },
                    "source": f"if {condition} then {consequence}",
                }
            )
        for source, target in sorted(derived):
            append(
                {
                    "family": "dependency",
                    "entities": [source, target],
                    "roles": {"dependent": source, "prerequisite": target},
                    "source": f"{source} transitively depends on {target}",
                    "derived": True,
                    "derive_source": "neural_compositional_windows",
                }
            )
        return blocks

    @staticmethod
    def _render_trace(data: dict) -> str:
        lines = [
            "--- PRE-REASONING TRACE (V4) ---",
            "External structural map for the calling AI's next forward pass.",
            "Reconsider the original problem using this map before answering.",
            "",
        ]
        if data["root_blockers"]:
            lines.append("ROOT BLOCKERS:")
            lines.extend(f"  - {name}" for name in data["root_blockers"])
        else:
            lines.append("ROOT BLOCKERS: None")

        lines.extend(("", "UNLOCK SEQUENCE:"))
        if data["unlock_sequence"]:
            for item in data["unlock_sequence"]:
                lines.append(f"  Step {item['step']}: {item['entity']}")
        else:
            lines.append("  None")

        if data["parallel_work"]:
            lines.extend(("", "PARALLEL WINDOWS:"))
            for item in data["parallel_work"]:
                lines.append(
                    f"  Step {item['step']}: {', '.join(item['entities'])}"
                )
        if data["conflicts"]:
            lines.extend(("", "CONFLICTS:"))
            lines.extend(
                f"  - {left} conflicts with {right}"
                for left, right in data["conflicts"]
            )
        if data["requirements"]:
            lines.extend(("", "REQUIREMENTS:"))
            for entity, operator, value in data["requirements"]:
                symbol = {"GEQ": ">=", "LEQ": "<=", "EQ": "="}[operator]
                lines.append(f"  - {entity} {symbol} {value}")
        if data["conditionals"]:
            lines.extend(("", "CONDITIONALS:"))
            for condition, consequence, otherwise in data["conditionals"]:
                suffix = f" else {otherwise}" if otherwise else ""
                lines.append(f"  - if {condition}, then {consequence}{suffix}")

        lines.extend(("", "CYCLES:"))
        if data["cycle"]:
            lines.append(f"  - {', '.join(data['cycle_nodes'])}")
        else:
            lines.append("  None")

        if data["derived"]:
            lines.extend(("", "MODEL-DERIVED ASSUMPTIONS:"))
            lines.extend(
                f"  - {source} depends on {target}"
                for source, target in data["derived"][:12]
            )
            extra = len(data["derived"]) - 12
            if extra > 0:
                lines.append(f"  ... {extra} more")
        return "\n".join(lines)

    def coverage_check_result(self, analysis: dict, response: str) -> dict:
        """Run a lexical coverage check over an existing structural trace.

        This helper checks textual presence only. It does not establish that the
        response agrees with, resolves, or semantically addresses an obligation.
        """
        if analysis.get("status") == "REPROMPT_REQUIRED":
            return analysis
        if not isinstance(response, str):
            raise TypeError("response must be a string")
        response_lower = response.casefold()

        def mentions(value: str) -> bool:
            value = str(value).strip()
            if not value:
                return True
            if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
                return bool(
                    re.search(
                        rf"(?<![\d.]){re.escape(value)}(?!(?:\d|\.\d))",
                        response,
                    )
                )
            return value.casefold() in response_lower

        gaps: list[str] = []
        root_blockers = list(analysis.get("root_blockers", []))
        root_names = set(root_blockers)
        for blocker in root_blockers:
            if not mentions(blocker):
                gaps.append(blocker)

        for item in analysis.get("unlock_sequence", []):
            entity = str(item.get("entity") or item.get("name") or "")
            if entity and entity not in root_names and not mentions(entity):
                gaps.append(f"unlock: {entity}")

        for left, right in analysis.get("conflicts", []):
            if not (mentions(left) and mentions(right)):
                gaps.append(f"conflict: {left} <> {right}")

        for entity, operator, value in analysis.get("requirements", []):
            if not (mentions(entity) and mentions(value)):
                gaps.append(f"requirement: {entity} {operator} {value}")

        for condition, consequence, otherwise in analysis.get("conditionals", []):
            terms = [condition, consequence] + ([otherwise] if otherwise else [])
            if not all(mentions(term) for term in terms):
                suffix = f" else {otherwise}" if otherwise else ""
                gaps.append(
                    f"conditional: if {condition} then {consequence}{suffix}"
                )

        if analysis.get("cycle"):
            cycle_nodes = list(analysis.get("cycle_nodes", []))
            names_present = all(mentions(node) for node in cycle_nodes)
            cycle_named = "cycle" in response_lower or "circular" in response_lower
            if not (names_present and cycle_named):
                gaps.append(f"cycle: {', '.join(cycle_nodes)}")

        return {
            "status": "CONTINUE" if gaps else "COMPLETE",
            "gaps": gaps,
            "root_blockers": root_blockers,
            "check_type": "lexical_coverage",
            "semantic_verified": False,
        }

    def coverage_check(self, form_text: str, response: str, **_kwargs) -> dict:
        """Analyze a form and run the non-semantic lexical coverage check."""
        analysis = self.analyze_form(form_text)
        return self.coverage_check_result(analysis, response)

    def pulse_result(self, analysis: dict, response: str) -> dict:
        """Compatibility alias for :meth:`coverage_check_result`."""
        return self.coverage_check_result(analysis, response)

    def pulse(
        self, form_text: str, response: str | None = None, **_kwargs
    ) -> dict:
        """Run a fresh reasoning pass for a Focus Mode reflection checkpoint.

        Passing ``response`` preserves the pre-v4.1 lexical-check behavior for
        existing callers. New callers should use ``coverage_check`` explicitly.
        """
        if response is not None:
            return self.coverage_check(form_text, response)
        analysis = self.analyze_form(form_text)
        if analysis.get("status") == "REPROMPT_REQUIRED":
            return analysis
        result = dict(analysis)
        result["status"] = "FOCUS_PULSE_COMPLETE"
        result["focus_mode"] = {
            "mode": "focus",
            "event": "reasoning_pulse",
            "interval_minutes": FOCUS_INTERVAL_MINUTES,
        }
        result["reflection_prompt"] = FOCUS_REMINDER
        return result

    def analyze_blocks(self, blocks: list[dict], **_kwargs) -> dict:
        """Analyze explicit structured blocks without interpreting their source prose."""
        clauses: list[str] = []
        operator_phrase = {"GEQ": "at least", "LEQ": "at most", "EQ": "exactly"}
        for block in blocks:
            family = str(block.get("family", "")).lower()
            roles = block.get("roles", {}) or {}
            entities = [str(value) for value in block.get("entities", [])]
            if family in ("dependency", "prereq"):
                dependent = roles.get("dependent") or roles.get("blocked")
                prerequisite = roles.get("prerequisite") or roles.get("blocker")
                if not dependent and len(entities) >= 2:
                    dependent, prerequisite = entities[:2]
                if dependent and prerequisite:
                    clauses.append(f"{dependent} depends on {prerequisite}.")
            elif family == "conflict":
                left = roles.get("left") or roles.get("initiator")
                right = roles.get("right") or roles.get("opposing")
                if not left and len(entities) >= 2:
                    left, right = entities[:2]
                if left and right:
                    clauses.append(f"{left} conflicts with {right}.")
            elif family == "requirement":
                entity = roles.get("entity") or (entities[0] if entities else "")
                operator = str(roles.get("operator", "")).upper()
                value = roles.get("value")
                if entity and operator in operator_phrase and value is not None:
                    clauses.append(
                        f"{entity} must be {operator_phrase[operator]} {value}."
                    )
            elif family == "conditional":
                condition = roles.get("condition")
                consequence = roles.get("consequence")
                otherwise = roles.get("otherwise", "")
                if condition and consequence:
                    clause = f"If {condition} passes, then {consequence} can proceed"
                    if otherwise:
                        clause += f", otherwise {otherwise} must proceed"
                    clauses.append(clause + ".")
            else:
                raise FormError(f"Unsupported block family: {family!r}")
        return self.analyze_form("\n".join(clauses))

    def raw_generate(self, prompt: str, max_new_tokens: int = 192) -> str:
        """Expose raw greedy generation for checkpoint diagnostics."""
        return generate_completion(self.model, prompt, max_new_tokens)


class FocusMode:
    """Scheduler-backed reflection loop for an active work session.

    The host should consume ``scheduler_request()`` and create a recurring task
    in the current chat. ``check()`` remains a local fallback for hosts without
    scheduling support. No background thread or sleep is created by this package.
    """

    def __init__(
        self,
        *,
        interval_minutes: float = FOCUS_INTERVAL_MINUTES,
        engine: V4ReasoningEngine | None = None,
        checkpoint_path: str | None = None,
        device: str = "auto",
        clock: Callable[[], float] | None = None,
    ):
        interval_minutes = float(interval_minutes)
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")
        self.interval_minutes = interval_minutes
        self.interval_seconds = interval_minutes * 60.0
        self._engine = engine
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._clock = clock or time.monotonic
        self._last_pulse_at = float(self._clock())
        self._pulse_count = 0
        self._pulse_required = False

    def _get_engine(self) -> V4ReasoningEngine:
        if self._engine is None:
            self._engine = V4ReasoningEngine(
                self._checkpoint_path, device=self._device
            )
        return self._engine

    def scheduler_request(self) -> dict:
        """Return the model-facing request for a recurring in-chat pulse task."""
        return _focus_scheduler_request(self.interval_minutes)

    def check(self) -> dict:
        """Return the current Focus Mode state without blocking or loading weights."""
        elapsed = max(0.0, float(self._clock()) - self._last_pulse_at)
        due = self._pulse_required or elapsed >= self.interval_seconds
        remaining = 0.0 if due else max(0.0, self.interval_seconds - elapsed)
        state = {
            "status": "FOCUS_PULSE_DUE" if due else "FOCUS_MODE_ACTIVE",
            "mode": "focus",
            "event": "reasoning_pulse",
            "due": due,
            "interval_minutes": self.interval_minutes,
            "elapsed_seconds": round(elapsed, 3),
            "next_pulse_in_seconds": round(remaining, 3),
            "pulse_count": self._pulse_count,
            "reminder": FOCUS_REMINDER if due else None,
            "scheduler_request": self.scheduler_request(),
        }
        if due:
            state.update(
                {
                    "minimum_blocks": MIN_FORM_BLOCKS,
                    "attached_form": FORM_TEMPLATE,
                }
            )
        return state

    def pulse(self, form_text: str | None = None, *, force: bool = False) -> dict:
        """Run a due reflection pulse, or return the current timer state."""
        state = self.check()
        if form_text is None or (not state["due"] and not force):
            return state

        result = self._get_engine().pulse(form_text)
        if result.get("status") == "REPROMPT_REQUIRED":
            self._pulse_required = True
            failed = dict(result)
            failed["focus_mode"] = {
                "mode": "focus",
                "event": "reasoning_pulse",
                "due": True,
                "interval_minutes": self.interval_minutes,
                "pulse_count": self._pulse_count,
                "next_pulse_in_seconds": 0.0,
            }
            failed["reflection_prompt"] = FOCUS_REMINDER
            return failed

        self._last_pulse_at = float(self._clock())
        self._pulse_count += 1
        self._pulse_required = False
        completed = dict(result)
        completed["focus_mode"] = {
            "mode": "focus",
            "event": "reasoning_pulse",
            "due": False,
            "interval_minutes": self.interval_minutes,
            "pulse_count": self._pulse_count,
            "next_pulse_in_seconds": self.interval_seconds,
        }
        return completed


ReasoningEngine = V4ReasoningEngine


__all__ = [
    "FOCUS_INTERVAL_MINUTES",
    "FOCUS_REMINDER",
    "FOCUS_SCHEDULER_PROMPT",
    "FORM_TEMPLATE",
    "FocusMode",
    "FormError",
    "MIN_FORM_BLOCKS",
    "NeuralContractError",
    "ReasoningEngine",
    "SHORT_FORM_ALARM",
    "V4ReasoningEngine",
    "form_spec",
]
