import os
import functools

import anyio
from huggingface_hub import hf_hub_download, list_repo_files

from ._internal import PyDataset, PyDatasetError as DatasetError  # noqa: F401


class Dataset(PyDataset):
    """Abstraction to read a set of parquet files with page pruning.

    A dataset representation that can be created from Hugging Face Hub or local cache.
    This class extends PyDataset to provide convenient methods for loading datasets
    from different sources, particularly Hugging Face repositories. PyDataset is
    implemented in a Rust extension module using parquet-rs to enable page pruning.

    See PyDataset for low-level details in libviewer/src/lib.rs.

    Parameters
    ----------
    name : str
        The name/identifier of the dataset, typically a Hugging Face repository name.
    files : list of dict
        A list of file dictionaries, each containing:
        - "data_path" or "path": Path to the data file
        - "data_size" or "size": Size of the data file in bytes
        - "metadata_path": Path to the metadata file
        - "num_rows": Number of rows in the file (optional)
    metadata_store : str
        URI pointing to the metadata storage location.
    data_store : str, optional
        URI pointing to the data storage location. Defaults to None, which implies
        using Hugging Face Hub. If specified, it can be a local file URI
        (e.g., "file://...") or other supported storage backends supported by
        the object_store crate. Mutually exclusive with `hf_token`, `revision`
        parameters which are used for Hugging Face Hub access when `data_store` is
        None.
    hf_token : str, optional
        Hugging Face authentication token, optional if data_store is specified.
    revision : str, optional
        The specific revision/branch/tag of the repository to use. Optional if
        data_store is specified.
    """

    # it can also be constructed from an explicit list of [
    #     {"data_path": ..., "data_size": ..., "metadata_path": ...},
    #     ...
    # ]

    def from_hub(repo, metadata_store, revision, hf_token):
        """Create a Dataset from Hugging Face Hub."""
        repo_files = list_repo_files(repo, repo_type="dataset", revision=revision)

        parquet_files = []
        for filename in repo_files:
            if filename.endswith(".parquet"):
                parquet_files.append(
                    {
                        "path": filename,
                        "size": None,
                        "num_rows": None,
                        "metadata_path": filename,
                    }
                )
        if not parquet_files:
            raise ValueError(f"No parquet files found in the dataset '{repo}'.")

        return Dataset(
            repo,
            parquet_files,
            revision=revision,
            hf_token=hf_token,
            metadata_store=metadata_store,
        )

    def from_cache(repo, metadata_store, revision=None, download=False):
        """Create a Dataset from HF local cache."""
        repo_files = list_repo_files(repo, repo_type="dataset", revision=revision)

        parquet_files = []
        for filename in repo_files:
            if filename.endswith(".parquet"):
                data_path = hf_hub_download(
                    repo, filename, repo_type="dataset", local_files_only=not download
                )
                data_size = os.path.getsize(data_path)
                parquet_files.append(
                    {
                        "path": data_path,
                        "size": data_size,
                        "num_rows": None,
                        "metadata_path": filename,
                    }
                )
        if not parquet_files:
            raise ValueError(f"No parquet files found in the dataset '{repo}'.")

        return Dataset(
            repo,
            parquet_files,
            revision=revision,
            data_store="file://",
            metadata_store=metadata_store,
        )

    def sync_scan(
        self, limit=None, offset=None, scan_size_limit=1 * 1024 * 1024 * 1024
    ):
        fn = functools.partial(
            self.scan, limit=limit, offset=offset, scan_size_limit=scan_size_limit
        )
        return anyio.run(fn)

    def sync_index(self, files=None):
        fn = functools.partial(self.index, files=files)
        return anyio.run(fn)
