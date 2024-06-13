# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from collections import Counter
from typing import Optional

from datasets import DownloadConfig, StreamingDownloadManager
from datasets.utils.file_utils import xbasename
from huggingface_hub.hf_api import HfApi, RepoSibling
from huggingface_hub.utils import RepositoryNotFoundError
from libcommon.exceptions import DatasetNotFoundError

from worker.dtos import CompleteJobResult, DatasetFiletypesResponse, Filetype
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunnerWithDatasetsCache
from worker.utils import FileExtensionTuple, get_file_extension


def extension_to_filetypes(file_extension_tuple: FileExtensionTuple, count: int) -> list[Filetype]:
    if file_extension_tuple[1]:
        return Filetype(extension=file_extension_tuple[0], count=count, archived_in=file_extension_tuple[1])
    return Filetype(extension=file_extension_tuple[0], count=count)


def get_filetypes(siblings: list[RepoSibling]) -> list[Filetype]:
    # count by extension

    counter = Counter[FileExtensionTuple](
        t for sibling in siblings for t in get_file_extension(sibling.rfilename).get_tuples()
    )

    return [extension_to_filetypes(k, v) for k, v in counter.items()]


def get_counter_from_archive(
    dataset: str,
    archive_filename: str,
    hf_token: Optional[str] = None,
) -> Counter[FileExtensionTuple]:
    dl_manager = StreamingDownloadManager(download_config=DownloadConfig(token=hf_token))
    base_url = f"hf://datasets/{dataset}/"
    archived_in = get_file_extension(archive_filename, recursive=False, clean=False).extension
    return Counter[FileExtensionTuple](
        (get_file_extension(xbasename(filename), recursive=False, clean=False).extension, archived_in)
        for filename in dl_manager.iter_files(dl_manager.extract(base_url + archive_filename))
    )


def get_filetypes_from_archives(
    dataset: str,
    archive_filenames: list[str],
    hf_token: Optional[str] = None,
) -> list[Filetype]:
    counter = Counter[FileExtensionTuple]()
    for archive_filename in archive_filenames:
        counter.update(get_counter_from_archive(dataset=dataset, archive_filename=archive_filename, hf_token=hf_token))
    return [
        Filetype(extension=extension, count=v, archived_in=archived_in)
        for (extension, archived_in), v in counter.items()
    ]


def compute_filetypes_response(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> DatasetFiletypesResponse:
    """
    Get the response of 'dataset-filetypes' for one specific dataset on huggingface.co.
    It is assumed that the dataset exists and can be accessed using the token.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)

    Raises:
        [~`huggingface_hub.utils.RepositoryNotFoundError`]:
            If repository is not found (error 404): wrong repo_id/repo_type, private
            but not authenticated or repo does not exist.

    Returns:
        `DatasetFiletypesResponse`: An object with the files count for each extension.
    """
    logging.info(f"compute 'dataset-filetypes' for {dataset=}")

    # get the list of files
    try:
        info = HfApi(endpoint=hf_endpoint, token=hf_token).dataset_info(dataset)
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError(f"Cannot get the dataset info for {dataset=}") from err

    # get file types count
    filetypes = get_filetypes(info.siblings)

    # look into the zip archives to get the file types
    SUPPORTED_ARCHIVE_EXTENSIONS = [".zip"]
    archive_filenames = [
        sibling.rfilename
        for sibling in info.siblings
        if get_file_extension(sibling.rfilename, recursive=False, clean=False).extension
        in SUPPORTED_ARCHIVE_EXTENSIONS
    ]
    filetypes_from_archives = get_filetypes_from_archives(
        dataset=dataset, archive_filenames=archive_filenames, hf_token=hf_token
    )

    return DatasetFiletypesResponse(filetypes=filetypes + filetypes_from_archives)


class DatasetFiletypesJobRunner(DatasetJobRunnerWithDatasetsCache):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-filetypes"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_filetypes_response(
                dataset=self.dataset,
                hf_endpoint=self.app_config.common.hf_endpoint,
                hf_token=self.app_config.common.hf_token,
            )
        )
