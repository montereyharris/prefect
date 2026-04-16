"""Azure Container Apps worker for executing Prefect flow runs as Container Apps Jobs.

Azure Container Apps Jobs are designed for short-lived, event-driven batch workloads.
They provide a serverless execution model that is billed only for actual compute time,
making them well-suited for on-demand Prefect flow runs.

Key differences from the ACI worker:
- Uses the Container Apps Jobs API rather than ARM template deployments.
- Requires a Container Apps *Environment* (a managed, shared networking layer).
- Job definitions are created per flow run and optionally deleted on completion.
- Execution status is polled via ``client.jobs_executions.list()``.

To start an ACA worker, run:

```bash
prefect worker start --pool 'my-work-pool' --type azure-container-apps
```

Replace ``my-work-pool`` with the name of your work pool.
"""

import time
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import anyio
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.mgmt.appcontainers import ContainerAppsAPIClient
from azure.mgmt.appcontainers.models import (
    Container,
    ContainerResources,
    EnvironmentVar,
    Job,
    JobConfiguration,
    JobConfigurationManualTriggerConfig,
    JobTemplate,
    Secret,
)
from pydantic import Field, SecretStr
from slugify import slugify

from prefect.client.orchestration import get_client
from prefect.exceptions import InfrastructureNotFound
from prefect.utilities.asyncutils import run_sync_in_worker_thread
from prefect.utilities.dockerutils import get_prefect_image_name
from prefect.utilities.processutils import command_from_string
from prefect.workers.base import (
    BaseJobConfiguration,
    BaseVariables,
    BaseWorker,
    BaseWorkerResult,
)
from prefect_azure.container_apps_diagnostics import (
    ACAInfrastructureDiagnosis,
    DiagnosisLevel,
    diagnose_aca_job_execution,
)
from prefect_azure.credentials import AzureContainerInstanceCredentials

if TYPE_CHECKING:
    from uuid import UUID

    from prefect.client.schemas.objects import Flow, FlowRun, WorkPool
    from prefect.client.schemas.responses import DeploymentResponse


# ── defaults ──────────────────────────────────────────────────────────────────

# Resource allocation defaults (these values run Prefect on the smallest ACA SKU)
ACA_DEFAULT_CPU = 0.5
ACA_DEFAULT_MEMORY = "1Gi"  # ACA requires Gi-suffix strings (not float GB like ACI)

# How long (seconds) Azure is allowed to run a single execution before stopping it
ACA_DEFAULT_REPLICA_TIMEOUT = 3600  # 1 hour

# Polling interval when waiting for a job to be deleted
JOB_DELETION_TIMEOUT_SECONDS = 60

# Environment variables that should be stored as ACA Secrets so they are
# redacted in Azure portal and execution logs.
ENV_SECRETS = ["PREFECT_API_KEY", "PREFECT_API_AUTH_STRING"]


# ── state enumerations ────────────────────────────────────────────────────────


class ContainerAppsJobProvisioningState(str, Enum):
    """Terminal provisioning states for a Container Apps Job definition."""

    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELED = "Canceled"
    DELETING = "Deleting"


class ContainerAppsJobExecutionStatus(str, Enum):
    """Possible runtime states for a Container Apps Job execution.

    Azure transitions an execution through these states; only the statuses
    listed in ``TERMINAL_EXECUTION_STATUSES`` mean the execution has finished.
    """

    RUNNING = "Running"
    PROCESSING = "Processing"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    STOPPED = "Stopped"
    DEGRADED = "Degraded"
    UNKNOWN = "Unknown"


# Execution states that mean the job will not make any further progress
TERMINAL_EXECUTION_STATUSES = {
    ContainerAppsJobExecutionStatus.SUCCEEDED,
    ContainerAppsJobExecutionStatus.FAILED,
    ContainerAppsJobExecutionStatus.STOPPED,
    ContainerAppsJobExecutionStatus.DEGRADED,
}


# ── configuration ─────────────────────────────────────────────────────────────


class AzureContainerAppsJobConfiguration(BaseJobConfiguration):
    """Configuration for running a Prefect flow as an Azure Container Apps Job.

    Maps directly to the fields needed to create a Container Apps Job and trigger
    a single execution.  Most fields mirror the ACI worker's configuration so
    that users migrating between the two workers have a familiar experience.
    """

    # Container image (defaults to the locally-installed Prefect image tag)
    image: str = Field(default_factory=get_prefect_image_name)

    # Azure resource targeting
    resource_group_name: str = Field(default=...)
    subscription_id: SecretStr = Field(default=...)

    # Azure region where the job and environment live (e.g. "eastus", "westus2")
    location: str = Field(default=...)

    # ARM resource ID of the Container Apps Environment that will host the job,
    # e.g. /subscriptions/{sub}/resourceGroups/{rg}/providers/
    #      Microsoft.App/managedEnvironments/{envName}
    container_app_environment_id: str = Field(default=...)

    # vCPU and memory allocation for each execution replica
    cpu: float = Field(default=ACA_DEFAULT_CPU)
    memory: str = Field(default=ACA_DEFAULT_MEMORY)

    # Optional list of user-assigned managed identity ARM resource IDs to attach
    identities: Optional[List[str]] = Field(default=None)

    # Credentials for the Azure management plane (re-uses the existing ACI block)
    aca_credentials: AzureContainerInstanceCredentials = Field(
        default_factory=AzureContainerInstanceCredentials,
    )

    # Maximum seconds a single execution may run before Azure stops it
    replica_timeout_seconds: int = Field(default=ACA_DEFAULT_REPLICA_TIMEOUT)

    # Seconds between Azure API calls while monitoring execution status
    task_watch_poll_interval: float = Field(default=10.0)

    # When True, the job definition is left in Azure after the flow run ends
    keep_job: bool = Field(default=False)

    # Optional container entrypoint to prepend to the flow run command.
    # Useful when the base image doesn't use the standard Prefect entrypoint.
    entrypoint: Optional[str] = Field(default=None)

    def prepare_for_flow_run(
        self,
        flow_run: "FlowRun",
        deployment: Optional["DeploymentResponse"] = None,
        flow: Optional["Flow"] = None,
        work_pool: Optional["WorkPool"] = None,
        worker_name: Optional[str] = None,
        worker_id: Optional["UUID"] = None,
    ):
        """Delegate to the base class (env vars and command substitution)."""
        super().prepare_for_flow_run(
            flow_run, deployment, flow, work_pool, worker_name, worker_id=worker_id
        )

    def _build_container_env(
        self,
    ) -> tuple[list[EnvironmentVar], list[Secret]]:
        """Split env vars into plain EnvironmentVar and ACA Secrets.

        Sensitive keys (``PREFECT_API_KEY``, etc.) are stored as job-level
        Secrets so they are redacted in the Azure portal and execution logs.
        Other vars are passed as plain environment variables.

        Returns:
            A ``(env_vars, secrets)`` tuple suitable for building a
            ``Container`` and ``JobConfiguration`` object.
        """
        merged_env = {**self._base_environment(), **self.env}

        env_vars: list[EnvironmentVar] = []
        secrets: list[Secret] = []

        for key, value in merged_env.items():
            if key in ENV_SECRETS:
                # Derive a valid ACA secret name: lowercase, hyphens only
                secret_name = key.lower().replace("_", "-")
                secrets.append(Secret(name=secret_name, value=value))
                # Reference the secret by name so ACA injects it at runtime
                env_vars.append(EnvironmentVar(name=key, secret_ref=secret_name))
            else:
                env_vars.append(EnvironmentVar(name=key, value=value))

        return env_vars, secrets

    def _build_command(self) -> Optional[list[str]]:
        """Build the flat command list for the container.

        ACA Jobs (like ACI) require a list of strings, not a shell string.
        We prepend the entrypoint if configured so that EXTRA_PIP_PACKAGES
        is still processed on startup when using the Prefect base image.
        """
        if not self.command:
            return None

        cmd = command_from_string(self.command)

        # Prepend entrypoint so Prefect base-image startup scripts still run
        if self.entrypoint:
            cmd.insert(0, self.entrypoint)

        return cmd


class AzureContainerAppsVariables(BaseVariables):
    """Variables for an Azure Container Apps Job flow run.

    These fields are exposed in the Prefect UI when creating or editing a work
    pool backed by the ``azure-container-apps`` worker type, and can be
    overridden per deployment.
    """

    image: str = Field(
        default_factory=get_prefect_image_name,
        description=(
            "The container image to run. Defaults to a Prefect base image "
            "matching your locally installed Prefect version."
        ),
    )
    resource_group_name: str = Field(
        default=...,
        title="Azure Resource Group Name",
        description=(
            "The Azure Resource Group that contains the Container Apps Environment."
        ),
    )
    subscription_id: SecretStr = Field(
        default=...,
        title="Azure Subscription ID",
        description="The Azure subscription ID to create Container Apps Jobs under.",
    )
    location: str = Field(
        default=...,
        title="Azure Region",
        description=(
            "Azure region for the Container Apps Job (e.g. 'eastus', 'westus2'). "
            "Must match the region of the Container Apps Environment."
        ),
    )
    container_app_environment_id: str = Field(
        default=...,
        title="Container Apps Environment ID",
        description=(
            "ARM resource ID of the Container Apps Environment to run jobs in. "
            "Format: /subscriptions/{sub}/resourceGroups/{rg}/providers/"
            "Microsoft.App/managedEnvironments/{envName}"
        ),
    )
    cpu: float = Field(
        default=ACA_DEFAULT_CPU,
        title="CPU",
        description=(
            f"Number of vCPUs to allocate to each execution replica. "
            f"Defaults to {ACA_DEFAULT_CPU}. "
            "Must be a value supported by the Container Apps SKU."
        ),
    )
    memory: str = Field(
        default=ACA_DEFAULT_MEMORY,
        title="Memory",
        description=(
            "Memory to allocate to each execution replica (e.g. '0.5Gi', '1Gi'). "
            f"Defaults to {ACA_DEFAULT_MEMORY}."
        ),
    )
    identities: Optional[List[str]] = Field(
        default=None,
        title="Managed Identities",
        description=(
            "User-assigned managed identity ARM resource IDs to attach to the job. "
            "Format: '/subscriptions/{sub}/resourceGroups/{rg}/providers/"
            "Microsoft.ManagedIdentity/userAssignedIdentities/{name}'"
        ),
    )
    aca_credentials: AzureContainerInstanceCredentials = Field(
        default_factory=AzureContainerInstanceCredentials,
        description=(
            "Azure credentials for authenticating with the Container Apps API. "
            "Defaults to DefaultAzureCredential (works with managed identity)."
        ),
    )
    replica_timeout_seconds: int = Field(
        default=ACA_DEFAULT_REPLICA_TIMEOUT,
        description=(
            "Maximum seconds Azure allows a single execution to run before stopping it. "
            "Increase this for long-running flows."
        ),
    )
    task_watch_poll_interval: float = Field(
        default=10.0,
        description="Seconds to wait between Azure API calls while monitoring status.",
    )
    keep_job: bool = Field(
        default=False,
        title="Keep Container Apps Job After Completion",
        description=(
            "When True, the Container Apps Job definition is left in Azure after "
            "the flow run completes. Useful for debugging execution logs. "
            "When False (default), the job definition is deleted on completion."
        ),
    )
    entrypoint: Optional[str] = Field(
        default=None,
        description=(
            "Optional container entrypoint to prepend to the flow run command. "
            "Only needed when using a custom image without the standard Prefect "
            "entrypoint script."
        ),
    )


# ── result ────────────────────────────────────────────────────────────────────


class AzureContainerAppsWorkerResult(BaseWorkerResult):
    """Holds the final outcome of a completed Container Apps Job execution."""


# ── worker ────────────────────────────────────────────────────────────────────


class AzureContainerAppsWorker(
    BaseWorker[
        AzureContainerAppsJobConfiguration,
        AzureContainerAppsVariables,
        AzureContainerAppsWorkerResult,
    ]
):
    """A Prefect worker that runs flows as Azure Container Apps Jobs.

    Each flow run gets its own Container Apps Job definition.  The worker
    triggers a single execution of that job, polls for completion, and
    (optionally) removes the job definition when done.

    This worker re-uses the existing ``AzureContainerInstanceCredentials``
    block for authentication so no additional credential blocks are needed.
    """

    type: str = "azure-container-apps"
    job_configuration = AzureContainerAppsJobConfiguration
    job_configuration_variables = AzureContainerAppsVariables
    _logo_url = "https://cdn.sanity.io/images/3ugk85nk/production/54e3fa7e00197a4fbd1d82ed62494cb58d08c96a-250x250.png"  # noqa
    _display_name = "Azure Container Apps"
    _description = (
        "Execute flow runs as Azure Container Apps Jobs. "
        "Requires an Azure account and a Container Apps Environment."
    )
    _documentation_url = "https://docs.prefect.io/integrations/prefect-azure"

    async def run(
        self,
        flow_run: "FlowRun",
        configuration: AzureContainerAppsJobConfiguration,
        task_status: Optional[anyio.abc.TaskStatus] = None,
    ) -> AzureContainerAppsWorkerResult:
        """Execute a flow run as an Azure Container Apps Job.

        High-level flow:
        1. Derive a unique, URL-safe job name from the flow run.
        2. Create (or update) a Container Apps Job definition in Azure.
        3. Start a single execution of that job.
        4. Poll the execution status until it reaches a terminal state.
        5. Delete the job definition (unless ``keep_job=True``).

        Args:
            flow_run: Prefect flow run to execute.
            configuration: Worker configuration for this specific run.
            task_status: Signal object used to tell Prefect that the
                infrastructure is ready (passes back the infrastructure PID).

        Returns:
            ``AzureContainerAppsWorkerResult`` with a POSIX-style exit code.
        """
        # Build an authenticated Azure management client
        aca_client = self._get_aca_client(configuration)

        # Derive a stable, human-readable job name from the flow name + run ID
        prefect_client = get_client()
        flow = await prefect_client.read_flow(flow_run.flow_id)
        job_name = self._build_job_name(flow.name, flow_run.id)

        self._logger.info(
            f"{self._log_prefix}: Creating Container Apps Job '{job_name}' "
            f"with image '{configuration.image}'..."
        )

        execution_name: Optional[str] = None
        status_code = -1

        try:
            # Step 1: Provision the Container Apps Job definition in Azure
            await self._create_job(aca_client, configuration, job_name)

            # Step 2: Trigger a single execution (manual trigger type)
            execution_name = await self._start_execution(
                aca_client, configuration, job_name
            )

            # Advertise the infrastructure identifier back to the Prefect engine
            # so the run can be cancelled if needed (see kill_infrastructure).
            identifier = f"{flow_run.id}:{job_name}:{execution_name}"
            if task_status is not None:
                task_status.started(value=identifier)

            self._logger.info(
                f"{self._log_prefix}: Execution '{execution_name}' started. "
                "Monitoring for completion..."
            )

            # Step 3: Watch execution status, surfacing diagnostics along the way
            status_code = await run_sync_in_worker_thread(
                self._watch_execution,
                aca_client,
                configuration,
                job_name,
                execution_name,
            )

            self._logger.info(
                f"{self._log_prefix}: Execution '{execution_name}' finished "
                f"with status code {status_code}."
            )

        finally:
            # Clean up the job definition unless the caller asked us to keep it
            if not configuration.keep_job and job_name:
                await self._delete_job(aca_client, configuration, job_name)

        return AzureContainerAppsWorkerResult(
            identifier=execution_name or job_name,
            status_code=status_code,
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_aca_client(
        self, configuration: AzureContainerAppsJobConfiguration
    ) -> ContainerAppsAPIClient:
        """Return an authenticated Azure Container Apps management client."""
        credential = configuration.aca_credentials._create_credential()
        return ContainerAppsAPIClient(
            credential=credential,
            subscription_id=configuration.subscription_id.get_secret_value(),
        )

    @staticmethod
    def _build_job_name(flow_name: str, flow_run_id: Any) -> str:
        """Produce a unique, URL-safe Container Apps Job name.

        ACA job names must be lowercase, start with a letter, contain only
        alphanumerics and hyphens, and be at most 32 characters long.
        We use a slug of the flow name plus the first 8 hex chars of the run ID.
        """
        slug = slugify(
            flow_name,
            max_length=20,  # leave room for the 'pf-' prefix and ID suffix
            regex_pattern=r"[^a-z0-9-]+",
            lowercase=True,
        )
        # Take the first 8 hex digits of the run UUID for short but unique suffix
        short_id = str(flow_run_id).replace("-", "")[:8]
        return f"pf-{slug}-{short_id}"

    async def _create_job(
        self,
        client: ContainerAppsAPIClient,
        configuration: AzureContainerAppsJobConfiguration,
        job_name: str,
    ) -> None:
        """Create (or update) the Container Apps Job definition in Azure.

        The job is configured for on-demand (manual) triggering with a single
        replica and no automatic retries — Prefect handles retry logic itself.
        Sensitive environment variables are stored as ACA Secrets.
        """
        # Split env vars into plain values and secrets
        env_vars, secrets = configuration._build_container_env()
        command = configuration._build_command()

        # Build the container specification with image, command, and resource requests
        container = Container(
            name="prefect-container",
            image=configuration.image,
            command=command,  # ACA 'command' overrides the container's ENTRYPOINT+CMD
            env=env_vars,
            resources=ContainerResources(
                cpu=configuration.cpu,
                memory=configuration.memory,
            ),
        )

        # Assemble the full job object with a manual trigger type
        job = Job(
            location=configuration.location,
            environment_id=configuration.container_app_environment_id,
            configuration=JobConfiguration(
                trigger_type="Manual",
                # Maximum time one execution replica may run
                replica_timeout=configuration.replica_timeout_seconds,
                # Disable automatic retries — Prefect controls retries
                replica_retry_limit=0,
                # One execution at a time, one replica per execution
                manual_trigger_config=JobConfigurationManualTriggerConfig(
                    parallelism=1,
                    replica_completion_count=1,
                ),
                # Secrets are set at the job level and referenced from env vars
                secrets=secrets if secrets else None,
            ),
            template=JobTemplate(
                containers=[container],
            ),
        )

        # Attach user-assigned managed identities when specified
        if configuration.identities:
            job.identity = {
                "type": "UserAssigned",
                "userAssignedIdentities": {
                    identity: {} for identity in configuration.identities
                },
            }

        self._logger.debug(
            f"{self._log_prefix}: Submitting job definition '{job_name}' to Azure..."
        )

        # begin_create_or_update is a long-running operation; wait for it
        poller = await run_sync_in_worker_thread(
            client.jobs.begin_create_or_update,
            configuration.resource_group_name,
            job_name,
            job,
        )
        await run_sync_in_worker_thread(poller.result)

        self._logger.info(
            f"{self._log_prefix}: Job definition '{job_name}' provisioned."
        )

    async def _start_execution(
        self,
        client: ContainerAppsAPIClient,
        configuration: AzureContainerAppsJobConfiguration,
        job_name: str,
    ) -> str:
        """Trigger one execution of the Container Apps Job.

        Returns the execution name assigned by Azure, which is needed to poll
        for completion status and to cancel the run if requested.
        """
        # begin_start submits the execution and returns a poller
        poller = await run_sync_in_worker_thread(
            client.jobs.begin_start,
            configuration.resource_group_name,
            job_name,
        )
        # Wait for Azure to accept the start request and return the execution ref
        execution = await run_sync_in_worker_thread(poller.result)

        # execution.name is the stable identifier for this specific run
        return execution.name

    def _watch_execution(
        self,
        client: ContainerAppsAPIClient,
        configuration: AzureContainerAppsJobConfiguration,
        job_name: str,
        execution_name: str,
    ) -> int:
        """Poll the execution status until a terminal state is reached.

        This method mirrors the Kubernetes worker's 'awareness layer' pattern:
        it uses the diagnostics module to classify failures and emit structured
        log messages so operators can act without consulting the Azure portal.

        This runs in a sync worker thread (called via ``run_sync_in_worker_thread``)
        so it uses ``time.sleep`` rather than ``anyio.sleep``.

        Returns:
            0 if the execution succeeded; 1 for any other terminal state.
        """
        # Track the last diagnosis emitted to avoid repeating the same message
        last_diagnosis: Optional[ACAInfrastructureDiagnosis] = None

        # Keep the final status around so we can map it to an exit code after
        # the loop exits
        status: str = ContainerAppsJobExecutionStatus.UNKNOWN

        while True:
            # Fetch the current execution object from Azure
            execution = self._get_execution(
                client, configuration, job_name, execution_name
            )

            status = (
                getattr(execution, "status", ContainerAppsJobExecutionStatus.UNKNOWN)
                if execution
                else ContainerAppsJobExecutionStatus.UNKNOWN
            )

            self._logger.debug(
                f"{self._log_prefix}: Execution '{execution_name}' status: {status}"
            )

            # Run the structured diagnostics check and log any new findings
            if execution:
                diagnosis = diagnose_aca_job_execution(execution)
                if diagnosis and diagnosis != last_diagnosis:
                    # Choose log severity based on the diagnosis level
                    log_fn = (
                        self._logger.error
                        if diagnosis.level == DiagnosisLevel.ERROR
                        else self._logger.warning
                    )
                    log_fn(
                        f"{self._log_prefix}: {diagnosis.summary}\n"
                        f"Detail: {diagnosis.detail}\n"
                        f"Resolution: {diagnosis.resolution}"
                    )
                    last_diagnosis = diagnosis

            # Break out of the polling loop once we reach a terminal state
            if status in TERMINAL_EXECUTION_STATUSES:
                break

            time.sleep(configuration.task_watch_poll_interval)

        # Map the terminal status to a POSIX-style exit code
        return 0 if status == ContainerAppsJobExecutionStatus.SUCCEEDED else 1

    @staticmethod
    def _get_execution(
        client: ContainerAppsAPIClient,
        configuration: AzureContainerAppsJobConfiguration,
        job_name: str,
        execution_name: str,
    ) -> Any:
        """Fetch a specific execution object by name from the job's execution list.

        The ACA management API exposes executions via ``client.jobs_executions.list()``.
        We iterate and return the first item whose ``name`` matches.

        Returns ``None`` on transient API errors so the polling loop can retry.
        """
        try:
            executions = client.jobs_executions.list(
                resource_group_name=configuration.resource_group_name,
                job_name=job_name,
            )
            for execution in executions:
                if execution.name == execution_name:
                    return execution
        except (HttpResponseError, ResourceNotFoundError):
            # Swallow transient errors — the polling loop will retry on the
            # next interval rather than crashing the worker.
            pass

        return None

    async def _delete_job(
        self,
        client: ContainerAppsAPIClient,
        configuration: AzureContainerAppsJobConfiguration,
        job_name: str,
    ) -> None:
        """Delete the Container Apps Job definition (and its execution history).

        Called in the ``finally`` block of ``run()`` unless ``keep_job=True``.
        Logs a warning and returns early if deletion does not complete within
        ``JOB_DELETION_TIMEOUT_SECONDS``.
        """
        self._logger.info(
            f"{self._log_prefix}: Deleting Container Apps Job '{job_name}'..."
        )
        try:
            poller = await run_sync_in_worker_thread(
                client.jobs.begin_delete,
                configuration.resource_group_name,
                job_name,
            )

            # Poll with a timeout so we don't block the worker indefinitely
            t0 = time.time()
            while not poller.done():
                if time.time() - t0 > JOB_DELETION_TIMEOUT_SECONDS:
                    self._logger.warning(
                        f"{self._log_prefix}: Timed out waiting for deletion of "
                        f"job '{job_name}'. The job may still exist in Azure."
                    )
                    return
                await anyio.sleep(configuration.task_watch_poll_interval)

        except ResourceNotFoundError:
            # Job was already gone — nothing to clean up
            pass

        self._logger.info(f"{self._log_prefix}: Job '{job_name}' deleted.")

    async def kill_infrastructure(
        self,
        infrastructure_pid: str,
        configuration: AzureContainerAppsJobConfiguration,
        grace_seconds: int = 30,
    ) -> None:
        """Cancel a running Container Apps Job execution.

        Args:
            infrastructure_pid: The identifier returned by ``run()``, in the
                format ``"flow_run_id:job_name:execution_name"``.
            configuration: The job configuration used to connect to Azure.
            grace_seconds: Unused (Azure handles graceful shutdown internally).

        Raises:
            InfrastructureNotFound: If the job or execution no longer exists.
        """
        parts = infrastructure_pid.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid infrastructure_pid: {infrastructure_pid!r}. "
                "Expected format: 'flow_run_id:job_name:execution_name'."
            )
        _, job_name, execution_name = parts

        aca_client = self._get_aca_client(configuration)

        try:
            # Stop this specific execution (not the whole job definition)
            poller = await run_sync_in_worker_thread(
                aca_client.jobs.begin_stop_execution,
                configuration.resource_group_name,
                job_name,
                execution_name,
            )
            await run_sync_in_worker_thread(poller.result)
        except ResourceNotFoundError:
            raise InfrastructureNotFound(
                f"Container Apps Job '{job_name}' execution "
                f"'{execution_name}' not found."
            )

    @property
    def _log_prefix(self) -> str:
        """Consistent log prefix that includes the worker name when available."""
        if self.name is not None:
            return f"AzureContainerAppsJob {self.name!r}"
        return "AzureContainerAppsJob"
