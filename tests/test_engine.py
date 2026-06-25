from pre_reasoning import ReasoningEngine, pulse


def test_analyze_happy_path():
    engine = ReasoningEngine()
    result = engine.analyze("A depends on B")

    assert result["mode"] == "full"
    assert result["neural_enriched"] is True
    assert result["root_blockers"]
    assert "ROOT BLOCKERS" in result["trace"]


def test_missing_checkpoint_raises(tmp_path):
    missing = tmp_path / "missing.safetensors"
    try:
        ReasoningEngine(checkpoint_path=missing)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_cycle_detection():
    engine = ReasoningEngine()
    result = engine.analyze("A depends on B. B depends on A.")

    assert result["has_cycle"] is True
    assert result["cycle_nodes"]


def test_conflict_detection():
    engine = ReasoningEngine()
    result = engine.analyze("CTO conflicts with senior dev.")

    assert result["conflicts"]
    assert result["conflicts"][0]["a_name"].lower() == "cto"
    assert result["conflicts"][0]["b_name"].lower() == "senior dev"


def test_pulse_hook_reports_gaps():
    result = pulse("A depends on B", "We should schedule a meeting.")

    assert result["status"] == "CONTINUE"
    assert result["gaps"]


def test_pulse_hook_accepts_addressed_entities():
    result = pulse("A depends on B", "Resolve B first, then A.")

    assert result["status"] == "COMPLETE"
    assert result["gaps"] == []
