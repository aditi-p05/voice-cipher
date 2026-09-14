import pytest

from backend.orchestration.audit import InvalidOperatorAction, record_operator_action


def action(**overrides):
    result = {
        "schema_version": "1.0.0",
        "case_id": "CASE-TEST",
        "operator_id": "OP-1",
        "action": "CONFIRM",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    result.update(overrides)
    return result


def test_valid_confirm_action_is_accepted():
    record_operator_action(action())  # must not raise


def test_valid_override_action_is_accepted():
    record_operator_action(action(action="OVERRIDE", reason="Operator judged HIGH not CRITICAL"))


def test_missing_required_field_is_rejected():
    incomplete = action()
    del incomplete["operator_id"]
    with pytest.raises(InvalidOperatorAction):
        record_operator_action(incomplete)


def test_invalid_action_value_is_rejected():
    with pytest.raises(InvalidOperatorAction):
        record_operator_action(action(action="DELETE_CASE"))
