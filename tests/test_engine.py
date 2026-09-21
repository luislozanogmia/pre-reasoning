import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file

import pre_reasoning.engine as engine_module
from pre_reasoning import (
    FOCUS_INTERVAL_MINUTES,
    FOCUS_SCHEDULER_PROMPT,
    FormError,
    FocusMode,
    ReasoningEngine,
    analyze_form,
    coverage_check,
    coverage_check_result,
    get_form,
    pulse,
    pulse_result,
    start_focus_mode,
)
from pre_reasoning.engine import NeuralContractError
from pre_reasoning.inference import DEFAULT_CHECKPOINT, MODEL_PARAMS, load_model

REPO_ROOT = Path(__file__).resolve().parents[1]


CHAIN_FORM = """DEPENDENCIES
Production Launch depends on Security Review.
Security Review depends on Architecture Approval.
"""

PULSE_FORM = """DEPENDENCIES
Production Launch depends on Security Review.
Security Review depends on Architecture Approval.

REQUIREMENTS
Test Coverage must be at least 95.
Rollback Time must be at most 15.
Critical Findings must be exactly 0.
"""


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


def test_form_is_the_outer_ai_contract():
    contract = get_form()
    assert contract["purpose"].startswith("The calling AI externalizes")
    assert "Interpret the user's language yourself." in contract["workflow"]
    assert "<dependent> depends on <prerequisite>." in contract["template"]
    assert contract["minimum_blocks"] == 5
    assert "INSUFFICIENT STRUCTURED BLOCKS" in contract["minimum_block_alarm"]


def test_short_form_returns_alarm_with_attached_template():
    result = analyze_form(
        "A depends on B.\nB depends on C.\nC conflicts with D.\n"
        "Coverage must be at least 95.",
        device="cpu",
    )
    assert result["status"] == "REPROMPT_REQUIRED"
    assert result["alarm"] == "PRE-REASONING ALARM: INSUFFICIENT STRUCTURED BLOCKS"
    assert result["block_count"] == 4
    assert result["minimum_blocks"] == 5
    assert result["neural_verified"] is False
    assert result["neural_model_calls"] == 0
    assert result["attached_form"] == get_form()["template"]
    assert "REPROMPT REQUIRED:" in result["trace"]


def test_five_blocks_clear_the_minimum_alarm():
    result = analyze_form(
        "A depends on B. B depends on C. C depends on D. D depends on E. "
        "E depends on F.",
        device="cpu",
    )
    assert "status" not in result
    assert result["neural_verified"] is True


def test_analyze_form_runs_model_and_preserves_entity_names():
    result = ReasoningEngine(device="cpu").analyze_form(
        CHAIN_FORM, _enforce_minimum=False
    )
    assert result["model"] == "pre-reasoning-1m"
    assert result["params"] == 1_019_580
    assert result["dependencies"] == [
        ["Production Launch", "Security Review"],
        ["Security Review", "Architecture Approval"],
    ]
    assert result["derived"] == [
        ["Production Launch", "Architecture Approval"]
    ]
    assert result["root_blockers"] == ["Architecture Approval"]
    assert result["neural_checks"] == ["dependency", "assumption"]
    assert result["neural_operations"] == {"assumption": 1, "dependency": 2}
    assert result["derive_meta"]["form_source"] == "calling_ai"
    assert "next forward pass" in result["trace"]


def test_raw_prose_is_rejected_instead_of_silently_interpreted():
    with pytest.raises(FormError, match="not a valid") as captured:
        analyze_form(
            "The launch cannot move until architecture signs off.", device="cpu"
        )
    error = captured.value
    assert error.form_text == "The launch cannot move until architecture signs off."
    assert error.form == error.form_text


def test_all_public_families_share_one_form():
    result = analyze_form(
        """DEPENDENCIES
Launch depends on Approval.
CONFLICTS
Speed Plan conflicts with Safety Plan.
REQUIREMENTS
Capacity must be at least 4.
Budget must be at most 9.
Retries must be exactly 3.
CONDITIONALS
If Security Review passes, then Launch can proceed, otherwise Remediation must proceed.
""",
        device="cpu",
    )
    assert result["dependencies"] == [["Launch", "Approval"]]
    assert result["conflicts"] == [["Speed Plan", "Safety Plan"]]
    assert result["requirements"] == [
        ["Budget", "LEQ", "9"],
        ["Capacity", "GEQ", "4"],
        ["Retries", "EQ", "3"],
    ]
    assert result["conditionals"] == [
        ["Security Review", "Launch", "Remediation"]
    ]
    assert set(result["neural_checks"]) == {
        "dependency",
        "conflict",
        "requirement_geq",
        "requirement_leq",
        "requirement_eq",
        "conditional_else",
    }


def test_cycle_is_model_detected_without_self_assumptions():
    result = ReasoningEngine(device="cpu").analyze_form(
        "A depends on B. B depends on C. C depends on A.",
        _enforce_minimum=False,
    )
    assert result["cycle"] is True
    assert result["cycle_nodes"] == ["A", "B", "C"]
    assert all(left != right for left, right in result["derived"])
    assert result["root_blockers"] == []
    assert result["unlock_sequence"] == []


def test_unlock_sequence_and_parallel_windows_are_exact():
    result = ReasoningEngine(device="cpu").analyze_form(
        "Frontend depends on API. Dashboard depends on API. API depends on Auth.",
        _enforce_minimum=False,
    )
    assert result["unlock_sequence"] == [
        {"step": 1, "entity": "Auth", "name": "Auth"},
        {"step": 2, "entity": "API", "name": "API"},
        {"step": 3, "entity": "Dashboard", "name": "Dashboard"},
        {"step": 3, "entity": "Frontend", "name": "Frontend"},
    ]
    assert result["parallel_work"] == [
        {"step": 3, "entities": ["Dashboard", "Frontend"]}
    ]


def test_large_form_has_no_historical_32_entity_ceiling():
    form = "\n".join(
        f"Stage {index:02d} depends on Stage {index - 1:02d}."
        for index in range(1, 38)
    )
    result = analyze_form(form, device="cpu")
    assert len(result["dependencies"]) == 37
    assert result["root_blockers"] == ["Stage 00"]
    assert result["unlock_sequence"][-1]["entity"] == "Stage 37"
    assert ["Stage 37", "Stage 00"] in result["derived"]


def test_analyze_blocks_uses_explicit_roles_for_every_family():
    engine = ReasoningEngine(device="cpu")
    result = engine.analyze_blocks(
        [
            {
                "family": "dependency",
                "roles": {"dependent": "Launch", "prerequisite": "Approval"},
            },
            {
                "family": "conflict",
                "roles": {"left": "Speed", "right": "Safety"},
            },
            {
                "family": "requirement",
                "roles": {"entity": "Coverage", "operator": "GEQ", "value": 95},
            },
            {
                "family": "conditional",
                "roles": {
                    "condition": "Audit",
                    "consequence": "Release",
                    "otherwise": "Remediation",
                },
            },
            {
                "family": "requirement",
                "roles": {"entity": "Retries", "operator": "EQ", "value": 3},
            },
        ]
    )
    assert result["dependencies"] == [["Launch", "Approval"]]
    assert result["conflicts"] == [["Speed", "Safety"]]
    assert result["requirements"] == [
        ["Coverage", "GEQ", "95"],
        ["Retries", "EQ", "3"],
    ]
    assert result["conditionals"] == [["Audit", "Release", "Remediation"]]


def test_coverage_check_checks_textual_presence_against_the_form_trace():
    first = coverage_check(
        PULSE_FORM, "We should schedule a meeting.", device="cpu"
    )
    assert first["status"] == "CONTINUE"
    assert first["check_type"] == "lexical_coverage"
    assert first["semantic_verified"] is False
    assert first["gaps"] == [
        "Architecture Approval",
        "unlock: Security Review",
        "unlock: Production Launch",
        "requirement: Critical Findings EQ 0",
        "requirement: Rollback Time LEQ 15",
        "requirement: Test Coverage GEQ 95",
    ]

    analysis = analyze_form(PULSE_FORM, device="cpu")
    second = coverage_check_result(
        analysis,
        "Resolve Architecture Approval before Security Review and Production Launch. "
        "Keep Test Coverage at 95 or higher, Rollback Time at 15 or lower, "
        "and Critical Findings at 0.",
        device="cpu",
    )
    assert second["status"] == "COMPLETE"
    assert second["gaps"] == []


def test_coverage_check_checks_all_explicit_structural_obligations():
    analysis = analyze_form(
        """DEPENDENCIES
Frontend depends on API.
API depends on Auth.
CONFLICTS
Fast Rollout conflicts with Safety Review.
REQUIREMENTS
Coverage must be at least 95.
CONDITIONALS
If Audit passes, then Release can proceed, otherwise Remediation must proceed.
""",
        device="cpu",
    )
    incomplete = coverage_check_result(
        analysis, "Resolve Auth first.", device="cpu"
    )
    assert incomplete["status"] == "CONTINUE"
    assert incomplete["gaps"] == [
        "unlock: API",
        "unlock: Frontend",
        "conflict: Fast Rollout <> Safety Review",
        "requirement: Coverage GEQ 95",
        "conditional: if Audit then Release else Remediation",
    ]

    complete = coverage_check_result(
        analysis,
        """Resolve Auth, then API, then Frontend.
Balance Fast Rollout against Safety Review.
Keep Coverage at 95 or higher.
If Audit passes, proceed with Release; otherwise use Remediation.""",
        device="cpu",
    )
    assert complete["status"] == "COMPLETE"
    assert complete["gaps"] == []


def test_coverage_check_accepts_numeric_requirement_before_punctuation():
    analysis = {"requirements": [["Coverage", "GEQ", "95"]]}

    complete = coverage_check_result(
        analysis, "Coverage is at least 95.", device="cpu"
    )
    assert complete["status"] == "COMPLETE"

    incomplete = coverage_check_result(
        analysis, "Coverage is at least 95.5.", device="cpu"
    )
    assert incomplete["status"] == "CONTINUE"
    assert incomplete["gaps"] == ["requirement: Coverage GEQ 95"]


def test_coverage_check_requires_cycle_acknowledgement():
    analysis = ReasoningEngine(device="cpu").analyze_form(
        "A depends on B. B depends on C. C depends on A.",
        _enforce_minimum=False,
    )
    assert coverage_check_result(analysis, "Review A, B, and C.")["gaps"] == [
        "cycle: A, B, C"
    ]
    assert coverage_check_result(
        analysis, "A, B, and C form a circular dependency."
    )["status"] == "COMPLETE"


def test_legacy_pulse_signatures_remain_coverage_aliases():
    response = (
        "Resolve Architecture Approval before Security Review and Production Launch. "
        "Keep Test Coverage at 95, Rollback Time at 15, and Critical Findings at 0."
    )
    public = pulse(PULSE_FORM, response, device="cpu")
    existing = analyze_form(PULSE_FORM, device="cpu")
    from_result = pulse_result(existing, response, device="cpu")
    assert public["status"] == "COMPLETE"
    assert from_result["status"] == "COMPLETE"
    assert public["check_type"] == "lexical_coverage"
    assert from_result["semantic_verified"] is False


def test_public_pulse_runs_a_fresh_checkpoint_analysis():
    result = pulse(PULSE_FORM, device="cpu")
    assert result["status"] == "FOCUS_PULSE_COMPLETE"
    assert result["neural_verified"] is True
    assert result["focus_mode"] == {
        "mode": "focus",
        "event": "reasoning_pulse",
        "interval_minutes": 10,
    }
    assert "fresh pre-reasoning pass" in result["reflection_prompt"]


def test_focus_mode_becomes_due_after_ten_minutes_without_loading_engine():
    clock = FakeClock()
    focus = FocusMode(clock=clock)
    assert focus.check()["status"] == "FOCUS_MODE_ACTIVE"
    assert focus._engine is None

    clock.advance(FOCUS_INTERVAL_MINUTES * 60)
    due = focus.check()
    assert due["status"] == "FOCUS_PULSE_DUE"
    assert due["due"] is True
    assert due["next_pulse_in_seconds"] == 0.0
    assert due["minimum_blocks"] == 5
    assert due["attached_form"] == get_form()["template"]
    assert focus._engine is None


def test_focus_mode_requests_one_recurring_task_in_the_current_chat():
    focus = FocusMode(interval_minutes=10)
    request = focus.scheduler_request()
    assert request["action"] == "create_or_update_recurring_task"
    assert request["destination"] == "current_chat"
    assert request["schedule"] == "every 10 minutes"
    assert request["interval_minutes"] == 10
    assert "Use the host's scheduling or automation capability now" in request[
        "instruction"
    ]
    assert "instead of creating a duplicate" in request["instruction"]
    assert "at least five valid Pre-Reasoning blocks" in request["task_prompt"]
    assert "Keep the pulse internal unless" in request["task_prompt"]
    assert focus.check()["scheduler_request"] == request
    assert FOCUS_SCHEDULER_PROMPT == request["instruction"]


def test_focus_mode_runs_reasoning_pulse_and_resets_interval():
    clock = FakeClock()
    focus = FocusMode(
        clock=clock,
        engine=ReasoningEngine(device="cpu"),
    )
    clock.advance(FOCUS_INTERVAL_MINUTES * 60)

    result = focus.pulse(PULSE_FORM)
    assert result["status"] == "FOCUS_PULSE_COMPLETE"
    assert result["neural_verified"] is True
    assert result["focus_mode"]["pulse_count"] == 1
    assert result["focus_mode"]["due"] is False
    assert result["focus_mode"]["next_pulse_in_seconds"] == 600.0
    assert focus.check()["status"] == "FOCUS_MODE_ACTIVE"


def test_focus_mode_short_form_stays_due_until_reprompted():
    clock = FakeClock()
    focus = FocusMode(
        clock=clock,
        engine=ReasoningEngine(device="cpu"),
    )
    clock.advance(FOCUS_INTERVAL_MINUTES * 60)

    result = focus.pulse(
        "A depends on B. B depends on C. C conflicts with D. "
        "Coverage must be at least 95."
    )
    assert result["status"] == "REPROMPT_REQUIRED"
    assert result["focus_mode"]["due"] is True
    assert result["focus_mode"]["pulse_count"] == 0
    assert result["attached_form"] == get_form()["template"]
    assert focus.check()["status"] == "FOCUS_PULSE_DUE"


def test_start_focus_mode_returns_non_blocking_default_session():
    focus = start_focus_mode(device="cpu")
    state = focus.check()
    assert isinstance(focus, FocusMode)
    assert state["status"] == "FOCUS_MODE_ACTIVE"
    assert state["interval_minutes"] == 10


def test_checkpoint_load_is_strict_and_uses_release_architecture():
    model, _, meta = load_model(device="cpu")
    assert model.count_parameters() == MODEL_PARAMS
    assert meta["strict_load"] is True
    assert meta["variant_id"] == "pre-reasoning-1m"


def test_raw_checkpoint_generation_is_not_adapter_output():
    engine = ReasoningEngine(device="cpu")
    output = engine.raw_generate("amber project one needs blue relay two. answer")
    assert output == " cycle no. therefore amber project one needs blue relay two."


def test_incompatible_checkpoint_fails_loudly(tmp_path: Path):
    weights = load_file(str(DEFAULT_CHECKPOINT), device="cpu")
    weights.pop("lm_head.weight")
    broken = tmp_path / "incompatible.safetensors"
    save_file(weights, str(broken))
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(broken, device="cpu")


def test_neural_output_controls_the_public_relation(monkeypatch):
    real_generate = engine_module.generate_completion

    def reversed_dependency(model, prompt, max_new_tokens=192):
        if prompt == "amber project one needs blue relay two. answer":
            return " cycle no. therefore blue relay two needs amber project one."
        return real_generate(model, prompt, max_new_tokens)

    monkeypatch.setattr(engine_module, "generate_completion", reversed_dependency)
    result = ReasoningEngine(device="cpu").analyze_form(
        "A depends on B.", _enforce_minimum=False
    )
    assert result["dependencies"] == [["B", "A"]]
    assert result["root_blockers"] == ["A"]


def test_disabled_neural_generation_has_no_adapter_fallback(monkeypatch):
    monkeypatch.setattr(
        engine_module,
        "generate_completion",
        lambda *_args, **_kwargs: "not a model result",
    )
    with pytest.raises(NeuralContractError, match="dependency completion"):
        ReasoningEngine(device="cpu").analyze_form(
            "A depends on B.", _enforce_minimum=False
        )


def test_requirement_value_corruption_fails_closed(monkeypatch):
    monkeypatch.setattr(
        engine_module,
        "generate_completion",
        lambda *_args, **_kwargs: (
            " requirement yes. therefore amber project one must be at least 99."
        ),
    )
    with pytest.raises(NeuralContractError, match="Incorrect requirement"):
        ReasoningEngine(device="cpu").analyze_form(
            "Capacity must be at least 4.", _enforce_minimum=False
        )


def test_model_and_cache_counters_report_actual_operations():
    engine = ReasoningEngine(device="cpu")
    first = engine.analyze_form(
        "Frontend depends on API. Dashboard depends on API.",
        _enforce_minimum=False,
    )
    assert first["neural_operations"] == {"dependency": 2}
    assert first["neural_model_calls"] == 1
    assert first["neural_cached_operations"] == 1

    second = engine.analyze_form(
        "Frontend depends on API.", _enforce_minimum=False
    )
    assert second["neural_operations"] == {"dependency": 1}
    assert second["neural_model_calls"] == 0
    assert second["neural_cached_operations"] == 1


def test_public_contract_board_3000_cases():
    engine = ReasoningEngine(device="cpu")

    for index in range(600):
        left = f"Workstream {index} Alpha"
        right = f"Gate {index} Beta"
        if index % 2:
            result = engine.analyze_form(
                f"{left} depends on {right}. {right} depends on {left}.",
                _enforce_minimum=False,
            )
            assert result["cycle"] is True
            assert result["cycle_nodes"] == sorted([left, right])
            assert set(map(tuple, result["dependencies"])) == {
                (left, right),
                (right, left),
            }
        else:
            result = engine.analyze_form(
                f"{left} depends on {right}.", _enforce_minimum=False
            )
            assert result["cycle"] is False
            assert result["dependencies"] == [[left, right]]
            assert result["root_blockers"] == [right]

    for index in range(600):
        left = f"Option {index} North"
        right = f"Option {index} South"
        result = engine.analyze_form(
            f"{left} conflicts with {right}.", _enforce_minimum=False
        )
        assert result["conflicts"] == [[left, right]]

    requirement_phrases = [
        ("at least", "GEQ"),
        ("at most", "LEQ"),
        ("exactly", "EQ"),
    ]
    for index in range(600):
        phrase, operator = requirement_phrases[index % 3]
        entity = f"Constraint {index} Capacity"
        value = str(index + 10)
        result = engine.analyze_form(
            f"{entity} must be {phrase} {value}.", _enforce_minimum=False
        )
        assert result["requirements"] == [[entity, operator, value]]

    for index in range(600):
        condition = f"Review {index} Pass"
        consequence = f"Release {index} Proceed"
        alternative = f"Remediation {index} Route"
        if index % 2:
            form = (
                f"If {condition} passes, then {consequence} can proceed, "
                f"otherwise {alternative} must proceed."
            )
            expected = [[condition, consequence, alternative]]
        else:
            form = f"If {condition} passes, then {consequence} can proceed."
            expected = [[condition, consequence, ""]]
        assert engine.analyze_form(
            form, _enforce_minimum=False
        )["conditionals"] == expected

    for index in range(600):
        source = f"Launch {index} Alpha"
        middle = f"Review {index} Beta"
        target = f"Approval {index} Gamma"
        result = engine.analyze_form(
            f"{source} depends on {middle}. {middle} depends on {target}.",
            _enforce_minimum=False,
        )
        assert set(map(tuple, result["dependencies"])) == {
            (source, middle),
            (middle, target),
        }
        assert result["derived"] == [[source, target]]
        assert result["root_blockers"] == [target]


def test_cli_form_contract_does_not_require_model_inference():
    completed = subprocess.run(
        [sys.executable, "-m", "pre_reasoning.cli", "--form"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(completed.stdout)
    assert contract["engine"] == "Pre-Reasoning V4"
    assert "<dependent> depends on <prerequisite>." in contract["template"]


def test_hooks_require_form_first_call_and_clear_the_turn_flag(tmp_path: Path):
    session_id = "pre-reasoning-hook-test"
    submit_payload = {
        "session_id": session_id,
        "prompt": "Please review this multi-stage project with several dependencies, constraints, conflicts, and conditional release decisions before proposing the implementation order.",
    }
    submitted = subprocess.run(
        [sys.executable, str(REPO_ROOT / "hooks" / "user_prompt_submit.py")],
        input=json.dumps(submit_payload),
        check=True,
        capture_output=True,
        text=True,
    )
    additional = json.loads(submitted.stdout)["hookSpecificOutput"][
        "additionalContext"
    ]
    assert "analyze_form" in additional
    assert "Do not pass the raw user prompt" in additional

    flag = Path(tempfile.gettempdir()) / (
        "pre_reasoning_"
        + hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:20]
        + ".required"
    )
    assert flag.exists()

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"role": "user"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "Bash",
                                    "input": {
                                        "command": "from pre_reasoning import analyze_form"
                                    },
                                }
                            ],
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    stopped = subprocess.run(
        [sys.executable, str(REPO_ROOT / "hooks" / "stop_enforcer.py")],
        input=json.dumps(
            {
                "session_id": session_id,
                "transcript_path": str(transcript),
            }
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert stopped.stdout == ""
    assert not flag.exists()


def test_stop_hook_blocks_when_form_engine_was_not_called(tmp_path: Path):
    session_id = "pre-reasoning-hook-negative-test"
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:20]
    flag = Path(tempfile.gettempdir()) / f"pre_reasoning_{digest}.required"
    flag.write_text("required\n", encoding="utf-8")
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"role": "user"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Done."}],
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    try:
        stopped = subprocess.run(
            [sys.executable, str(REPO_ROOT / "hooks" / "stop_enforcer.py")],
            input=json.dumps(
                {
                    "session_id": session_id,
                    "transcript_path": str(transcript),
                }
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        decision = json.loads(stopped.stdout)
        assert decision["decision"] == "block"
        assert "analyze_form" in decision["reason"]
    finally:
        flag.unlink(missing_ok=True)
