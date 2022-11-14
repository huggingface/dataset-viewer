# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List

from migration.migration import Migration
from migration.migrations._20221110230400_example import MigrationExample


# TODO: add a way to automatically collect migrations from the migrations/ folder
class MigrationsCollector:
    def get_migrations(self) -> List[Migration]:
        return [
            MigrationExample(version="20221110230400", description="example"),
        ]
