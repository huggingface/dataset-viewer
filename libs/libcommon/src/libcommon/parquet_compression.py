# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

# Kept apart from parquet_utils, which imports libviewer: libcommon imports this module at package import, and the
# api, admin, sse-api and webhook images ship without libviewer.

import pyarrow.parquet as pq

from libcommon.exceptions import ParquetBrotliCompressionNotSupportedError


def raise_if_brotli_compressed(parquet_metadata: pq.FileMetaData) -> None:
    """
    Refuse Parquet files that have a column chunk compressed with the BROTLI codec.

    The Brotli decoder bundled in pyarrow and DuckDB has a known memory-safety bug (CVE-2026-111497) and no fixed
    release exists yet. The footer is never compressed, so its metadata gives the codec before any page is decoded.

    Args:
        parquet_metadata (`pq.FileMetaData`): The metadata read from the footer of the Parquet file.

    Raises:
        [~`libcommon.exceptions.ParquetBrotliCompressionNotSupportedError`]:
            If a column chunk uses the BROTLI compression codec.
    """
    for row_group_id in range(parquet_metadata.num_row_groups):
        row_group = parquet_metadata.row_group(row_group_id)
        for column_id in range(row_group.num_columns):
            if row_group.column(column_id).compression == "BROTLI":
                raise ParquetBrotliCompressionNotSupportedError(
                    "The Parquet file uses the BROTLI compression codec, which is not supported. "
                    "Please use another codec, for example ZSTD or SNAPPY."
                )
