# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from mongodb_migration.migration import Migration


class MigrationExample(Migration):
    def up(self) -> None:
        logging.info("Example migration, upgrade step")

    def down(self) -> None:
        logging.info("Example migration, downgrade step")

    def validate(self) -> None:
        logging.info("Example migration, validation is OK")
