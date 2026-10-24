# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import importlib
from typing import Any, NoReturn

import pyarrow.dataset as ds
from datasets.utils.patching import patch_submodule

from libcommon.parquet_utils import raise_if_brotli_compressed

# The Lance decoder reads variable-width values without validating the offsets stored in the file,
# so a crafted .lance file makes the worker return adjacent heap memory or segfault. The dataset
# entry point can be hardened with read_params={"validate_on_decode": True}, but LanceFileReader,
# which datasets uses for standalone .lance files, exposes no such option. Refuse the format until
# the offsets are validated upstream.
LANCE_DISABLED_MESSAGE = "The Lance format is not supported."

lance_module = importlib.import_module("datasets.packaged_modules.lance.lance")


def _refuse_lance(*args: Any, **kwargs: Any) -> NoReturn:
    raise NotImplementedError(LANCE_DISABLED_MESSAGE)


# Refuse in the builder rather than unregistering the format: .lance files keep resolving to this
# module, so they fail explicitly instead of falling back to whichever module matches the Lance
# metadata files.
setattr(lance_module.Lance, "_split_generators", _refuse_lance)


class _BrotliRefusingParquetFileFormat(ds.ParquetFileFormat):  # type: ignore
    def make_fragment(self, *args: Any, **kwargs: Any) -> Any:
        fragment = super().make_fragment(*args, **kwargs)
        raise_if_brotli_compressed(fragment.metadata)
        return fragment


# The Parquet builder and the folder-based builders (for metadata.parquet files) decode the pages with
# ParquetFileFormat().make_fragment(f).to_batches(), so check the codecs in the footer of each file first.
for module_name in [
    "datasets.packaged_modules.parquet.parquet",
    "datasets.packaged_modules.folder_based_builder.folder_based_builder",
]:
    patch_submodule(
        importlib.import_module(module_name), "pyarrow.dataset.ParquetFileFormat", _BrotliRefusingParquetFileFormat
    ).start()
