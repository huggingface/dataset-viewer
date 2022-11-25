# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.processing_steps import FIRST_ROWS_STEP, PARQUET_STEP, SPLITS_STEP


def test_next_steps():
    assert SPLITS_STEP.next_steps == [FIRST_ROWS_STEP]
    assert PARQUET_STEP.next_steps == []
    assert FIRST_ROWS_STEP.next_steps == []
