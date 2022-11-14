# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from migration.collector import MigrationsCollector


def test_collector():
    collector = MigrationsCollector()
    migrations = collector.get_migrations()
    assert len(migrations) == 1
    assert migrations[0].version == "20221110230400"
    assert migrations[0].description == "example"
