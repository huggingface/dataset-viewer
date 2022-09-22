from typing import Optional

from huggingface_hub.hf_api import HfApi  # type: ignore
from huggingface_hub.utils import RepositoryNotFoundError  # type: ignore


def is_dataset_supported(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> bool:
    """
    Check if the dataset exists on the Hub and is supported by the datasets-server.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
    Returns:
        [`bool`]: True if the dataset is supported by the datasets-server.
    """
    try:
        # note that token is required to access gated dataset info
        info = HfApi(endpoint=hf_endpoint).dataset_info(dataset, token=hf_token)
    except RepositoryNotFoundError:
        return False
    return info.private is False
