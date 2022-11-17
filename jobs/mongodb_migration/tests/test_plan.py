# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List, Optional, Type

import pytest
from mongodb_migration.database_migrations import (
    DatabaseMigration,
    _clean_maintenance_database,
)
from mongodb_migration.migration import IrreversibleMigration, Migration
from mongodb_migration.plan import Plan, SavedMigrationsError


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_maintenance_database()


class MigrationOK(Migration):
    def up(self) -> None:
        pass

    def down(self) -> None:
        pass

    def validate(self) -> None:
        pass


class MigrationErrorInUp(Migration):
    def up(self) -> None:
        raise RuntimeError("Error in up")

    def down(self) -> None:
        pass

    def validate(self) -> None:
        pass


class MigrationErrorInValidate(Migration):
    def up(self) -> None:
        pass

    def down(self) -> None:
        pass

    def validate(self) -> None:
        raise RuntimeError("Error in validation")


class MigrationErrorInUpAndDown(Migration):
    def up(self) -> None:
        raise RuntimeError("Error in up")

    def down(self) -> None:
        raise RuntimeError("Error in down")

    def validate(self) -> None:
        pass


class MigrationErrorIrreversible(Migration):
    def up(self) -> None:
        raise RuntimeError("Error in up")

    def down(self) -> None:
        raise IrreversibleMigration("Error in down")

    def validate(self) -> None:
        pass


def test_empty_plan():
    plan = Plan(collected_migrations=[])

    assert plan.collected_migrations == []
    plan.execute()
    assert plan.executed_migrations == []


migration_ok_a = MigrationOK(version="20221110230400", description="ok a")
migration_ok_b = MigrationOK(version="20221110230401", description="ok b")
migration_error_in_up = MigrationErrorInUp(version="20221110230402", description="error in up")
migration_error_in_validate = MigrationErrorInValidate(version="20221110230403", description="error in validate")
migration_error_in_up_and_down = MigrationErrorInUpAndDown(
    version="20221110230404", description="error in up and down"
)
migration_error_irreversible = MigrationErrorIrreversible(
    version="20221110230405", description="error because migration is irreversible"
)


@pytest.mark.parametrize(
    "collected_migrations",
    (
        [migration_ok_a, migration_ok_b],
        [migration_ok_b, migration_ok_a],
    ),
)
def test_collected_migrations_order_dont_matter(collected_migrations: List[Migration]):
    assert DatabaseMigration.objects.distinct("version") == []
    plan = Plan(collected_migrations=collected_migrations)
    assert plan.executed_migrations == []
    plan.execute()
    sorted_migrations = sorted(collected_migrations, key=lambda migration: migration.version)
    assert plan.executed_migrations == sorted_migrations
    assert DatabaseMigration.objects.distinct("version") == [migration.version for migration in sorted_migrations]


@pytest.mark.parametrize(
    "collected_migrations,executed_migrations,exception",
    [
        ([migration_error_in_up], [], None),
        ([migration_error_in_validate], [], None),
        ([migration_error_in_up_and_down], [migration_error_in_up_and_down], RuntimeError),
        ([migration_error_irreversible], [migration_error_irreversible], IrreversibleMigration),
        ([migration_ok_a, migration_error_in_up], [], None),
        (
            [migration_ok_a, migration_error_in_up_and_down],
            [migration_ok_a, migration_error_in_up_and_down],
            RuntimeError,
        ),
    ],
)
def test_errors_in_migration_steps(
    collected_migrations: List[Migration], executed_migrations: List[Migration], exception: Optional[Type[Exception]]
):
    assert DatabaseMigration.objects.distinct("version") == []
    plan = Plan(collected_migrations=collected_migrations)
    assert plan.executed_migrations == []
    if exception is None:
        # rollback worked
        plan.execute()
    else:
        # rollback failed
        with pytest.raises(exception):
            plan.execute()
    assert plan.executed_migrations == executed_migrations
    assert DatabaseMigration.objects.distinct("version") == [migration.version for migration in executed_migrations]


@pytest.mark.parametrize(
    "previous_migrations,collected_migrations,executed_migrations,exception",
    [
        ([], [], [], None),
        ([], [migration_ok_a], [migration_ok_a], None),
        ([migration_ok_a], [migration_ok_a, migration_ok_b], [migration_ok_b], None),
        # the previous migrations must be in the collected migrations
        ([migration_ok_a], [], [], SavedMigrationsError),
        ([migration_ok_a], [migration_ok_b], [], SavedMigrationsError),
        # error with the versions order
        ([migration_ok_b], [migration_ok_a, migration_ok_b], [], SavedMigrationsError),
    ],
)
def test_get_planned_migrations(
    previous_migrations: List[Migration],
    collected_migrations: List[Migration],
    executed_migrations: List[Migration],
    exception: Optional[Type[Exception]],
):
    for migration in previous_migrations:
        DatabaseMigration(version=migration.version, description=migration.description).save()
    assert DatabaseMigration.objects.distinct("version") == [migration.version for migration in previous_migrations]
    plan = Plan(collected_migrations=collected_migrations)
    assert plan.executed_migrations == []
    if exception is None:
        # up worked
        plan.apply()
    else:
        # up failed
        with pytest.raises(exception):
            plan.apply()
    assert plan.executed_migrations == executed_migrations
    assert DatabaseMigration.objects.distinct("version") == [
        migration.version for migration in (previous_migrations + executed_migrations)
    ]


def test_internal_operations_are_idempotent():
    plan = Plan(collected_migrations=[migration_ok_a, migration_ok_b])
    plan.rollback()
    plan.rollback()
    plan.rollback()
    plan.apply()
    plan.apply()
    plan.apply()
    plan.apply()
    plan.rollback()
    plan.apply()
    plan.rollback()


def test_execute_is_idempotent():
    plan = Plan(collected_migrations=[migration_ok_a, migration_ok_b])
    plan.execute()
    plan.execute()
    Plan(collected_migrations=[migration_ok_a, migration_ok_b]).execute()
