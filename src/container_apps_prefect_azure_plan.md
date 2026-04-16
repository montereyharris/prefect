Plan: Azure Container Apps Worker for Prefect
Draft for review: add a new Azure Container Apps worker path in the Azure integration package, reusing Azure Container Instance patterns for credentials/job submission while borrowing Kubernetes-style runtime awareness patterns (status watching, event surfacing, diagnostics, and failure classification). This keeps behavior familiar for existing Azure users and aligns worker observability with Kubernetes worker expectations.

Steps
Map current ACI worker flow in container_instance.py, container_instance.py, and credentials.py to define ACA equivalents for job_configuration, worker, and auth blocks.
Add ACA worker implementation and typed config models under workers, then expose imports/exports in __init__.py and package metadata in pyproject.toml.
Introduce Kubernetes-like awareness layer for ACA by adapting concepts from observer.py, diagnostics.py, and _logging.py into ACA-focused status/event/diagnostic helpers.
Add ACA-facing docs and usage examples in README.md, including parity/contrast with ACI behavior and clear worker type configuration fields.
Add tests mirroring ACI and Kubernetes confidence areas in tests (new ACA worker unit tests + credentials/job translation tests) and ensure compatibility with existing worker/block standards patterns.
Further Considerations
Should ACA runtime target Container Apps Jobs only, Apps only, or both? Option A: Jobs-first (simpler parity), Option B: dual-mode, Option C: Apps-first long-running model.
How much “awareness” is required in v1? Option A: terminal state + logs, Option B: add event stream + failure taxonomy, Option C: full watcher/diagnostics parity with Kubernetes.
Should ACA reuse existing Azure credentials block as-is or introduce an ACA-specific block surface for environment/resource targeting?
