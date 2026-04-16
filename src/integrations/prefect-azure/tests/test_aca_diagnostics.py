"""Unit tests for the ACA diagnostics module.

These tests mirror the style of ``tests/test_diagnostics.py`` in the
``prefect-kubernetes`` integration, verifying that each failure condition
produces the expected ``ACAInfrastructureDiagnosis`` and that healthy
or unknown states return ``None``.
"""

from unittest.mock import MagicMock

import pytest
from prefect_azure.container_apps_diagnostics import (
    ACAInfrastructureDiagnosis,
    DiagnosisLevel,
    diagnose_aca_job_execution,
)


def make_execution(status: str, name: str = "test-execution") -> MagicMock:
    """Build a mock ACA job execution object with the given status."""
    execution = MagicMock()
    execution.name = name
    execution.status = status
    return execution


class TestDiagnoseACAJobExecution:
    """Tests for the top-level ``diagnose_aca_job_execution`` dispatcher."""

    def test_returns_none_for_running(self):
        # A running execution is healthy — no diagnosis expected
        assert diagnose_aca_job_execution(make_execution("Running")) is None

    def test_returns_none_for_processing(self):
        # Processing is a transient state, not a failure
        assert diagnose_aca_job_execution(make_execution("Processing")) is None

    def test_returns_none_for_succeeded(self):
        # Successful executions should not produce any diagnosis
        assert diagnose_aca_job_execution(make_execution("Succeeded")) is None

    def test_returns_none_for_unknown_status(self):
        # Unrecognized statuses should not raise; they return None
        assert diagnose_aca_job_execution(make_execution("SomeFutureStatus")) is None

    def test_returns_none_when_no_status_attribute(self):
        # Objects without a 'status' attribute should be handled gracefully
        execution = MagicMock(spec=[])  # no attributes at all
        assert diagnose_aca_job_execution(execution) is None


class TestFailedExecution:
    """Tests for the Failed execution checker."""

    def test_failed_returns_error_diagnosis(self):
        execution = make_execution("Failed", name="my-execution")
        result = diagnose_aca_job_execution(execution)

        assert isinstance(result, ACAInfrastructureDiagnosis)
        assert result.level == DiagnosisLevel.ERROR
        assert "my-execution" in result.summary
        assert "failed" in result.summary.lower()

    def test_failed_diagnosis_contains_resolution(self):
        result = diagnose_aca_job_execution(make_execution("Failed"))

        # Resolution should give concrete next steps
        assert result is not None
        assert len(result.resolution) > 0
        assert "Azure portal" in result.resolution

    def test_failed_diagnosis_is_frozen(self):
        # ACAInfrastructureDiagnosis must be frozen for deduplication by value
        result = diagnose_aca_job_execution(make_execution("Failed"))
        assert result is not None
        with pytest.raises(Exception):
            result.level = DiagnosisLevel.INFO  # type: ignore[misc]


class TestDegradedExecution:
    """Tests for the Degraded execution checker."""

    def test_degraded_returns_warning_diagnosis(self):
        execution = make_execution("Degraded", name="degraded-exec")
        result = diagnose_aca_job_execution(execution)

        assert isinstance(result, ACAInfrastructureDiagnosis)
        assert result.level == DiagnosisLevel.WARNING
        assert "degraded-exec" in result.summary

    def test_degraded_resolution_mentions_replicas(self):
        result = diagnose_aca_job_execution(make_execution("Degraded"))
        assert result is not None
        assert "replica" in result.resolution.lower()


class TestStoppedExecution:
    """Tests for the Stopped execution checker."""

    def test_stopped_returns_warning_diagnosis(self):
        execution = make_execution("Stopped", name="stopped-exec")
        result = diagnose_aca_job_execution(execution)

        assert isinstance(result, ACAInfrastructureDiagnosis)
        assert result.level == DiagnosisLevel.WARNING
        assert "stopped-exec" in result.summary

    def test_stopped_resolution_mentions_timeout(self):
        result = diagnose_aca_job_execution(make_execution("Stopped"))
        assert result is not None
        assert "timeout" in result.resolution.lower()


class TestDiagnosisEquality:
    """Verify that diagnosis deduplication (by value) works correctly.

    The worker emits a log message only when the diagnosis changes. This relies
    on the ``ACAInfrastructureDiagnosis`` dataclass being frozen (hashable and
    comparable by value).
    """

    def test_same_status_produces_equal_diagnoses(self):
        d1 = diagnose_aca_job_execution(make_execution("Failed", name="exec-1"))
        d2 = diagnose_aca_job_execution(make_execution("Failed", name="exec-1"))
        assert d1 == d2

    def test_different_names_produce_different_diagnoses(self):
        # Different execution names → different summaries → different diagnoses
        d1 = diagnose_aca_job_execution(make_execution("Failed", name="exec-a"))
        d2 = diagnose_aca_job_execution(make_execution("Failed", name="exec-b"))
        assert d1 != d2

    def test_different_statuses_produce_different_diagnoses(self):
        d1 = diagnose_aca_job_execution(make_execution("Failed"))
        d2 = diagnose_aca_job_execution(make_execution("Degraded"))
        assert d1 != d2
