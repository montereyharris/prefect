"""Unit tests for the Azure Container Apps worker.

Tests mirror the style of ``tests/test_aci_worker.py``, mocking the Azure SDK
clients so no real Azure calls are made.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, Mock

import prefect_azure.credentials
import pytest
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import ClientSecretCredential
from prefect_azure import AzureContainerInstanceCredentials
from prefect_azure.workers.container_apps import (
    ACA_DEFAULT_CPU,
    ACA_DEFAULT_MEMORY,
    ENV_SECRETS,
    TERMINAL_EXECUTION_STATUSES,
    AzureContainerAppsJobConfiguration,
    AzureContainerAppsVariables,
    AzureContainerAppsWorker,
    AzureContainerAppsWorkerResult,
    ContainerAppsJobExecutionStatus,
)
from pydantic import SecretStr

from prefect.client.schemas import FlowRun
from prefect.exceptions import InfrastructureNotFound
from prefect.server.schemas.core import Flow

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def aca_credentials(monkeypatch):
    """AzureContainerInstanceCredentials backed by a mocked ClientSecretCredential."""
    mock_credential = Mock(wraps=ClientSecretCredential, return_value=Mock())
    monkeypatch.setattr(
        prefect_azure.credentials,
        "ClientSecretCredential",
        mock_credential,
    )
    return AzureContainerInstanceCredentials(
        client_id="test-client-id",
        client_secret="test-client-secret",
        tenant_id="test-tenant-id",
    )


@pytest.fixture
def worker_flow():
    return Flow(id=uuid.uuid4(), name="test-flow")


@pytest.fixture
def worker_flow_run(worker_flow):
    return FlowRun(id=uuid.uuid4(), flow_id=worker_flow.id, name="test-flow-run")


@pytest.fixture
def base_config_values(aca_credentials):
    """Minimal set of values needed to build an ACA job configuration."""
    return {
        "command": "prefect flow-run execute",
        "env": {},
        "aca_credentials": aca_credentials,
        "resource_group_name": "test-rg",
        "subscription_id": SecretStr("test-sub-id"),
        "location": "eastus",
        "container_app_environment_id": (
            "/subscriptions/test-sub/resourceGroups/test-rg"
            "/providers/Microsoft.App/managedEnvironments/test-env"
        ),
        "task_watch_poll_interval": 0.01,  # fast polling in tests
    }


async def build_job_configuration(
    base_values: dict, overrides: dict | None = None
) -> AzureContainerAppsJobConfiguration:
    """Test helper that builds an AzureContainerAppsJobConfiguration from values.

    Does NOT call ``prepare_for_flow_run``; call that separately when needed
    (e.g. to inject flow run env vars and labels before asserting on them).
    """
    values = {**base_values, **(overrides or {})}
    variables = AzureContainerAppsVariables(**values)

    json_config = {
        "job_configuration": AzureContainerAppsJobConfiguration.json_template(),
        "variables": variables.model_dump(),
    }
    return await AzureContainerAppsJobConfiguration.from_template_and_values(
        json_config, values
    )


@pytest.fixture
async def job_configuration(base_config_values, worker_flow_run):
    config = await build_job_configuration(base_config_values)
    config.prepare_for_flow_run(worker_flow_run)
    return config


@pytest.fixture
def mock_aca_client():
    """A MagicMock that mimics ContainerAppsAPIClient.jobs operations."""
    client = MagicMock(name="ContainerAppsAPIClient")

    # Mock the job creation poller
    create_poller = Mock()
    create_poller.result = Mock(return_value=None)
    client.jobs.begin_create_or_update = Mock(return_value=create_poller)

    # Mock the execution start poller — returns an object with a 'name' attribute
    execution_ref = Mock()
    execution_ref.name = "test-execution-001"
    start_poller = Mock()
    start_poller.result = Mock(return_value=execution_ref)
    client.jobs.begin_start = Mock(return_value=start_poller)

    # Mock list_executions to return a succeeded execution immediately
    succeeded_execution = Mock()
    succeeded_execution.name = "test-execution-001"
    succeeded_execution.status = ContainerAppsJobExecutionStatus.SUCCEEDED
    client.jobs_executions = Mock()
    client.jobs_executions.list = Mock(return_value=[succeeded_execution])

    # Mock the deletion poller
    delete_poller = Mock()
    delete_poller.done = Mock(return_value=True)
    delete_poller.result = Mock(return_value=None)
    client.jobs.begin_delete = Mock(return_value=delete_poller)

    # Mock stop_execution
    stop_poller = Mock()
    stop_poller.result = Mock(return_value=None)
    client.jobs.begin_stop_execution = Mock(return_value=stop_poller)

    return client


@pytest.fixture
def mock_prefect_client(monkeypatch, worker_flow):
    """Mock the Prefect API client so we don't need a live server."""
    mock_client = MagicMock()
    mock_client.read_flow = AsyncMock(return_value=worker_flow)
    # Support use as a context manager
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    monkeypatch.setattr(
        "prefect_azure.workers.container_apps.get_client",
        Mock(return_value=mock_client),
    )
    return mock_client


# ── configuration tests ───────────────────────────────────────────────────────


class TestAzureContainerAppsJobConfiguration:
    """Tests for AzureContainerAppsJobConfiguration helpers."""

    async def test_build_container_env_plain_vars(self, base_config_values, worker_flow_run):
        """Non-secret env vars should be plain EnvironmentVar objects."""
        config = await build_job_configuration(
            base_config_values, {"env": {"MY_VAR": "hello"}}
        )
        config.prepare_for_flow_run(worker_flow_run)

        env_vars, secrets = config._build_container_env()

        var_names = [v.name for v in env_vars]
        assert "MY_VAR" in var_names
        # Plain vars should have a 'value', not a 'secret_ref'
        my_var = next(v for v in env_vars if v.name == "MY_VAR")
        assert my_var.value == "hello"
        assert my_var.secret_ref is None

    async def test_build_container_env_secret_vars(self, base_config_values, worker_flow_run):
        """PREFECT_API_KEY should be stored as an ACA Secret and referenced."""
        config = await build_job_configuration(
            base_config_values, {"env": {"PREFECT_API_KEY": "s3cr3t"}}
        )
        config.prepare_for_flow_run(worker_flow_run)

        env_vars, secrets = config._build_container_env()

        # The secret should appear in the secrets list
        secret_names = [s.name for s in secrets]
        assert "prefect-api-key" in secret_names

        # The env var should reference the secret by name
        api_key_var = next(v for v in env_vars if v.name == "PREFECT_API_KEY")
        assert api_key_var.secret_ref == "prefect-api-key"
        assert api_key_var.value is None

    async def test_build_command_from_string(self, base_config_values, worker_flow_run):
        """A string command should be split into a list."""
        config = await build_job_configuration(
            base_config_values, {"command": "python -m mymodule --arg value"}
        )
        config.prepare_for_flow_run(worker_flow_run)

        cmd = config._build_command()
        assert cmd == ["python", "-m", "mymodule", "--arg", "value"]

    async def test_build_command_with_entrypoint(self, base_config_values, worker_flow_run):
        """Entrypoint should be prepended to the command list."""
        config = await build_job_configuration(
            base_config_values,
            {
                "command": "prefect flow-run execute",
                "entrypoint": "/opt/prefect/entrypoint.sh",
            },
        )
        config.prepare_for_flow_run(worker_flow_run)

        cmd = config._build_command()
        assert cmd is not None
        assert cmd[0] == "/opt/prefect/entrypoint.sh"

    async def test_build_command_none_when_no_command(self, base_config_values):
        """No command should result in None, letting the container use its CMD."""
        # Override the command to None and do NOT call prepare_for_flow_run,
        # because that method injects a default command from the flow run context.
        config = await build_job_configuration(
            base_config_values, {"command": None}
        )
        # Manually clear the command to simulate the no-command case
        config.command = None

        assert config._build_command() is None

    def test_default_cpu_and_memory(self):
        """Defaults should match the documented ACA SKU minimums."""
        assert ACA_DEFAULT_CPU == 0.5
        assert ACA_DEFAULT_MEMORY == "1Gi"

    def test_env_secrets_list(self):
        """Sensitive keys should be in ENV_SECRETS so they land in ACA Secrets."""
        assert "PREFECT_API_KEY" in ENV_SECRETS
        assert "PREFECT_API_AUTH_STRING" in ENV_SECRETS


class TestAzureContainerAppsWorkerBuildJobName:
    """Tests for the static ``_build_job_name`` helper."""

    def test_job_name_is_lowercase(self):
        name = AzureContainerAppsWorker._build_job_name("My Flow", uuid.uuid4())
        assert name == name.lower()

    def test_job_name_starts_with_pf_prefix(self):
        name = AzureContainerAppsWorker._build_job_name("my-flow", uuid.uuid4())
        assert name.startswith("pf-")

    def test_job_name_within_32_chars(self):
        # ACA imposes a 32-character limit on job names
        long_name = "A Very Long Flow Name That Exceeds Normal Limits"
        name = AzureContainerAppsWorker._build_job_name(long_name, uuid.uuid4())
        assert len(name) <= 32

    def test_job_name_is_unique_per_run(self):
        # Different run IDs should produce different job names
        n1 = AzureContainerAppsWorker._build_job_name("flow", uuid.uuid4())
        n2 = AzureContainerAppsWorker._build_job_name("flow", uuid.uuid4())
        assert n1 != n2

    def test_job_name_handles_special_chars(self):
        # Special characters should be stripped, not cause errors
        name = AzureContainerAppsWorker._build_job_name(
            "My Flow (v2) — special!", uuid.uuid4()
        )
        import re
        assert re.match(r"^[a-z0-9-]+$", name)


# ── worker run() tests ────────────────────────────────────────────────────────


class TestAzureContainerAppsWorkerRun:
    """Integration-style tests for the worker's ``run()`` method."""

    async def test_run_succeeds_with_mocked_client(
        self,
        job_configuration,
        worker_flow_run,
        mock_aca_client,
        mock_prefect_client,
        monkeypatch,
    ):
        """A fully mocked happy path should return status_code=0."""
        # Patch _get_aca_client to return our mock
        monkeypatch.setattr(
            AzureContainerAppsWorker,
            "_get_aca_client",
            Mock(return_value=mock_aca_client),
        )

        async with AzureContainerAppsWorker(work_pool_name="test-pool") as worker:
            result = await worker.run(worker_flow_run, job_configuration)

        assert isinstance(result, AzureContainerAppsWorkerResult)
        assert result.status_code == 0

    async def test_run_calls_create_and_start(
        self,
        job_configuration,
        worker_flow_run,
        mock_aca_client,
        mock_prefect_client,
        monkeypatch,
    ):
        """The worker must create the job definition and start an execution."""
        monkeypatch.setattr(
            AzureContainerAppsWorker,
            "_get_aca_client",
            Mock(return_value=mock_aca_client),
        )

        async with AzureContainerAppsWorker(work_pool_name="test-pool") as worker:
            await worker.run(worker_flow_run, job_configuration)

        mock_aca_client.jobs.begin_create_or_update.assert_called_once()
        mock_aca_client.jobs.begin_start.assert_called_once()

    async def test_run_deletes_job_by_default(
        self,
        job_configuration,
        worker_flow_run,
        mock_aca_client,
        mock_prefect_client,
        monkeypatch,
    ):
        """Job definition should be deleted after completion when keep_job=False."""
        job_configuration.keep_job = False
        monkeypatch.setattr(
            AzureContainerAppsWorker,
            "_get_aca_client",
            Mock(return_value=mock_aca_client),
        )

        async with AzureContainerAppsWorker(work_pool_name="test-pool") as worker:
            await worker.run(worker_flow_run, job_configuration)

        mock_aca_client.jobs.begin_delete.assert_called_once()

    async def test_run_keeps_job_when_requested(
        self,
        job_configuration,
        worker_flow_run,
        mock_aca_client,
        mock_prefect_client,
        monkeypatch,
    ):
        """Job definition should NOT be deleted when keep_job=True."""
        job_configuration.keep_job = True
        monkeypatch.setattr(
            AzureContainerAppsWorker,
            "_get_aca_client",
            Mock(return_value=mock_aca_client),
        )

        async with AzureContainerAppsWorker(work_pool_name="test-pool") as worker:
            await worker.run(worker_flow_run, job_configuration)

        mock_aca_client.jobs.begin_delete.assert_not_called()

    async def test_run_returns_status_code_1_on_failure(
        self,
        job_configuration,
        worker_flow_run,
        mock_aca_client,
        mock_prefect_client,
        monkeypatch,
    ):
        """A Failed execution should produce status_code=1."""
        # Override jobs_executions.list to return a failed execution
        failed_execution = Mock()
        failed_execution.name = "test-execution-001"
        failed_execution.status = ContainerAppsJobExecutionStatus.FAILED
        mock_aca_client.jobs_executions.list = Mock(return_value=[failed_execution])

        monkeypatch.setattr(
            AzureContainerAppsWorker,
            "_get_aca_client",
            Mock(return_value=mock_aca_client),
        )

        async with AzureContainerAppsWorker(work_pool_name="test-pool") as worker:
            result = await worker.run(worker_flow_run, job_configuration)

        assert result.status_code == 1

    async def test_run_signals_task_status(
        self,
        job_configuration,
        worker_flow_run,
        mock_aca_client,
        mock_prefect_client,
        monkeypatch,
    ):
        """The worker must call task_status.started() with the infrastructure PID."""
        task_status = Mock()
        task_status.started = Mock()

        monkeypatch.setattr(
            AzureContainerAppsWorker,
            "_get_aca_client",
            Mock(return_value=mock_aca_client),
        )

        async with AzureContainerAppsWorker(work_pool_name="test-pool") as worker:
            await worker.run(worker_flow_run, job_configuration, task_status=task_status)

        task_status.started.assert_called_once()
        # The PID should contain the flow run ID, job name, and execution name
        pid = task_status.started.call_args[1]["value"]
        assert str(worker_flow_run.id) in pid
        assert "test-execution-001" in pid


# ── kill_infrastructure tests ─────────────────────────────────────────────────


class TestKillInfrastructure:
    """Tests for the ``kill_infrastructure`` method."""

    async def test_kill_stops_named_execution(
        self,
        job_configuration,
        mock_aca_client,
        monkeypatch,
    ):
        """kill_infrastructure should stop the named execution."""
        monkeypatch.setattr(
            AzureContainerAppsWorker,
            "_get_aca_client",
            Mock(return_value=mock_aca_client),
        )

        pid = f"{uuid.uuid4()}:my-job-name:my-execution"

        async with AzureContainerAppsWorker(work_pool_name="test-pool") as worker:
            await worker.kill_infrastructure(pid, job_configuration)

        mock_aca_client.jobs.begin_stop_execution.assert_called_once_with(
            job_configuration.resource_group_name,
            "my-job-name",
            "my-execution",
        )

    async def test_kill_raises_infrastructure_not_found(
        self,
        job_configuration,
        mock_aca_client,
        monkeypatch,
    ):
        """kill_infrastructure should raise InfrastructureNotFound if gone."""
        mock_aca_client.jobs.begin_stop_execution.side_effect = ResourceNotFoundError(
            "not found"
        )
        monkeypatch.setattr(
            AzureContainerAppsWorker,
            "_get_aca_client",
            Mock(return_value=mock_aca_client),
        )

        pid = f"{uuid.uuid4()}:my-job:my-exec"

        async with AzureContainerAppsWorker(work_pool_name="test-pool") as worker:
            with pytest.raises(InfrastructureNotFound):
                await worker.kill_infrastructure(pid, job_configuration)

    async def test_kill_raises_on_invalid_pid_format(
        self,
        job_configuration,
        mock_aca_client,
        monkeypatch,
    ):
        """A malformed infrastructure PID should raise a ValueError."""
        monkeypatch.setattr(
            AzureContainerAppsWorker,
            "_get_aca_client",
            Mock(return_value=mock_aca_client),
        )

        async with AzureContainerAppsWorker(work_pool_name="test-pool") as worker:
            with pytest.raises(ValueError, match="Invalid infrastructure_pid"):
                await worker.kill_infrastructure("bad-pid", job_configuration)


# ── terminal status tests ─────────────────────────────────────────────────────


class TestTerminalStatuses:
    """Verify the terminal status set is complete."""

    def test_succeeded_is_terminal(self):
        assert ContainerAppsJobExecutionStatus.SUCCEEDED in TERMINAL_EXECUTION_STATUSES

    def test_failed_is_terminal(self):
        assert ContainerAppsJobExecutionStatus.FAILED in TERMINAL_EXECUTION_STATUSES

    def test_stopped_is_terminal(self):
        assert ContainerAppsJobExecutionStatus.STOPPED in TERMINAL_EXECUTION_STATUSES

    def test_degraded_is_terminal(self):
        assert ContainerAppsJobExecutionStatus.DEGRADED in TERMINAL_EXECUTION_STATUSES

    def test_running_is_not_terminal(self):
        assert ContainerAppsJobExecutionStatus.RUNNING not in TERMINAL_EXECUTION_STATUSES

    def test_processing_is_not_terminal(self):
        assert ContainerAppsJobExecutionStatus.PROCESSING not in TERMINAL_EXECUTION_STATUSES
