# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

import pytest

from mongodb_migration.migration import Migration


class MigrationOK(Migration):
    def up(self) -> None:
        pass

    def down(self) -> None:
        pass

    def validate(self) -> None:
        pass


version_ok = "20221110230400"
description = "description a"
version_date_error = "20225510230400"
version_format_error = "wrong format"
version_too_short = "20221110"


@pytest.mark.parametrize(
    "version,description,exception",
    [
        (version_ok, None, ValueError),
        (None, description, ValueError),
        (version_date_error, description, ValueError),
        (version_format_error, description, ValueError),
        (version_too_short, description, ValueError),
        (version_ok, description, None),
    ],
)
def test_migration(version: str, description: str, exception: Optional[type[Exception]]) -> None:
    if exception is None:
        MigrationOK(version=version, description=description)
    else:
        with pytest.raises(exception):
            MigrationOK(version=version, description=description)
