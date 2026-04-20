"""Azure Container Apps Job failure diagnostics.

Adapts the pattern from the Kubernetes worker's diagnostics module
(`prefect_kubernetes.diagnostics`) for Azure Container Apps Jobs.

Pattern-matches ACA job execution status into structured failure diagnoses
with actionable resolution hints, so the worker can surface meaningful
messages to operators without requiring them to navigate the Azure portal.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any


class DiagnosisLevel(str, enum.Enum):
    """Severity level for an ACA infrastructure diagnosis.

    Mirrors `prefect_kubernetes.diagnostics.DiagnosisLevel` so callers can
    use a consistent pattern across worker implementations.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclasses.dataclass(frozen=True)
class ACAInfrastructureDiagnosis:
    """A structured diagnosis of an Azure Container Apps Job execution failure.

    Frozen so that it can be compared by value when deduplicating repeated
    diagnostic events (the same failure should only be logged once per run).
    """

    level: DiagnosisLevel
    summary: str
    detail: str
    resolution: str


def diagnose_aca_job_execution(execution: Any) -> ACAInfrastructureDiagnosis | None:
    """Inspect a Container Apps Job execution object and return a diagnosis.

    Returns ``None`` when the execution is healthy or in an unrecognized state
    that doesn't require user intervention.

    Args:
        execution: A job execution object returned by the Azure SDK
            (e.g. the items yielded by ``client.jobs.list_executions(...)``).
            Must expose a ``status`` attribute with the current execution state.
    """
    # Walk through each specialized checker in priority order.
    # Return the first non-None diagnosis found.
    return (
        _check_failed(execution)
        or _check_degraded(execution)
        or _check_stopped(execution)
    )


# ── private checkers ──────────────────────────────────────────────────────────


def _check_failed(execution: Any) -> ACAInfrastructureDiagnosis | None:
    """Detect terminal failure on a Container Apps Job execution.

    Azure sets the execution status to 'Failed' when the container exits with
    a non-zero code or when the platform cannot provision the container.
    """
    if getattr(execution, "status", None) != "Failed":
        return None

    name = getattr(execution, "name", "<unknown>")
    return ACAInfrastructureDiagnosis(
        level=DiagnosisLevel.ERROR,
        summary=f"Container Apps Job execution '{name}' failed",
        detail=(
            "The job execution reached a Failed state. This may indicate an "
            "application error, a misconfigured container image, or an "
            "infrastructure provisioning problem."
        ),
        resolution=(
            "Check the Container Apps Job execution logs in the Azure portal "
            "for the root cause. Common causes: incorrect image name or tag, "
            "missing environment variables, insufficient CPU/memory allocation, "
            "or an unhealthy Container Apps Environment. Verify that the image "
            "exists in its registry and that required credentials are configured."
        ),
    )


def _check_degraded(execution: Any) -> ACAInfrastructureDiagnosis | None:
    """Detect a degraded execution (partial replica failure).

    Azure marks an execution as 'Degraded' when some replicas have failed but
    others are still running.  This is a warning because the execution may
    still complete if the remaining replicas succeed.
    """
    if getattr(execution, "status", None) != "Degraded":
        return None

    name = getattr(execution, "name", "<unknown>")
    return ACAInfrastructureDiagnosis(
        level=DiagnosisLevel.WARNING,
        summary=f"Container Apps Job execution '{name}' is degraded",
        detail=(
            "The execution is running in a degraded state: some replicas have "
            "failed while others continue to run. The overall execution may still "
            "succeed if the remaining replicas complete normally."
        ),
        resolution=(
            "Monitor the execution in the Azure portal. If the execution does not "
            "recover, consider stopping it and retrying the flow run. Check for "
            "resource pressure (CPU, memory, quota) in the Container Apps "
            "Environment and review replica logs for the specific failure cause."
        ),
    )


def _check_stopped(execution: Any) -> ACAInfrastructureDiagnosis | None:
    """Detect an externally stopped execution.

    Azure transitions an execution to 'Stopped' when it is manually cancelled,
    when the ``replica_timeout`` is exceeded, or when the platform evicts it.
    """
    if getattr(execution, "status", None) != "Stopped":
        return None

    name = getattr(execution, "name", "<unknown>")
    return ACAInfrastructureDiagnosis(
        level=DiagnosisLevel.WARNING,
        summary=f"Container Apps Job execution '{name}' was stopped",
        detail=(
            "The execution was stopped before it could complete. This can result "
            "from a manual cancellation request, the replica timeout being exceeded, "
            "or an Azure platform event (e.g. infrastructure maintenance)."
        ),
        resolution=(
            "If the flow run was not intentionally cancelled, check the Azure "
            "portal for events or alerts on the Container Apps Environment. "
            "Consider increasing 'replica_timeout_seconds' in the work pool "
            "configuration if the job is consistently timing out before finishing."
        ),
    )
