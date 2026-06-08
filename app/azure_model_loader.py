"""Download model artifacts from Azure Blob Storage using managed identity."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_model_path(default_path: Path) -> Path:
    """
    Resolve model path for local or Azure Blob Storage deployment.

    When AZURE_STORAGE_ACCOUNT_NAME is set, downloads the model blob to
    MODEL_PATH (default /tmp/delay_model.pkl) using DefaultAzureCredential.
    Falls back to the local default path when Azure settings are absent.
    """
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "").strip()
    container_name = os.getenv("MODEL_BLOB_CONTAINER", "models").strip()
    blob_name = os.getenv("MODEL_BLOB_NAME", "delay_model.pkl").strip()
    local_path = Path(os.getenv("MODEL_PATH", str(default_path)))

    if not account_name:
        logger.info("Azure Blob not configured; using local model path: %s", default_path)
        return default_path

    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:
        raise RuntimeError(
            "Azure Storage SDK required for blob model loading. "
            "Install requirements-azure.txt"
        ) from exc

    account_url = os.getenv(
        "AZURE_STORAGE_ACCOUNT_URL",
        f"https://{account_name}.blob.core.windows.net",
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        logger.info("Using cached model at %s", local_path)
        return local_path

    logger.info(
        "Downloading model blob %s/%s from %s",
        container_name,
        blob_name,
        account_url,
    )
    credential = DefaultAzureCredential()
    blob_service = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)

    with open(local_path, "wb") as model_file:
        model_file.write(blob_client.download_blob().readall())

    logger.info("Model downloaded to %s", local_path)
    return local_path
