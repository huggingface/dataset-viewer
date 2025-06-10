import os
from ._internal import PyDataset
from huggingface_hub import hf_hub_download, list_repo_files

__all__ = ["Dataset"]


# TODO(kszucs): add config/split arguments to restrict the dataset files
class Dataset(PyDataset):
    # it can also be constructed from an explicit list of [
    #     {"data_path": ..., "data_size": ..., "metadata_path": ...},
    #     ...
    # ]

    def from_hub(repo, metadata_store):
        """Create a Dataset from Hugging Face Hub."""
        repo_files = list_repo_files(repo, repo_type="dataset")

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
            repo, parquet_files, data_store="hf://", metadata_store=metadata_store
        )

    def from_cache(repo, metadata_store, download=False):
        """Create a Dataset from HF local cache."""
        repo_files = list_repo_files(repo, repo_type="dataset")

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
            repo, parquet_files, data_store="file://", metadata_store=metadata_store
        )
