"""
Worker classes for Azure.

Provides two worker types:
- ``AzureContainerWorker``: Runs flow runs as Azure Container Instance groups
  (ARM-template based deployment).
- ``AzureContainerAppsWorker``: Runs flow runs as Azure Container Apps Jobs
  (serverless, billed per execution second).
"""
