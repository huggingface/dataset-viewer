# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import List

from mongodb_migration.database_migrations import DatabaseMigration
from mongodb_migration.migration import Migration


class SavedMigrationsError(Exception):
    pass


class Plan:
    collected_migrations: List[Migration]
    executed_migrations: List[Migration]

    def __init__(self, collected_migrations: List[Migration]):
        self.collected_migrations = collected_migrations
        self.executed_migrations = []

    def get_saved_migrations_versions(self) -> List[str]:
        return DatabaseMigration.objects().distinct("version")

    def get_planned_migrations(self) -> List[Migration]:
        saved_migrations_versions = sorted(self.get_saved_migrations_versions())
        collected_migrations = sorted(self.collected_migrations, key=lambda m: m.version)
        first_collected_migrations_versions = [
            migration.version for migration in collected_migrations[: len(saved_migrations_versions)]
        ]
        if saved_migrations_versions != first_collected_migrations_versions:
            logging.error(
                "Database migrations are not in sync with collected migrations. Database:"
                f" {saved_migrations_versions}, Collected: {first_collected_migrations_versions}"
            )
            raise SavedMigrationsError(
                "The saved migrations in the database should be the first collected migrations."
            )
        num_saved_migrations = len(saved_migrations_versions)
        num_collected_migrations = len(collected_migrations)
        if not num_collected_migrations:
            logging.error("No collected migrations")
        if num_saved_migrations:
            logging.info(f"{num_saved_migrations} migrations have already been applied. They will be skipped.")
        if num_saved_migrations == len(collected_migrations):
            logging.info("All migrations have already been applied.")
        return collected_migrations[num_saved_migrations:]

    def execute(self) -> None:
        try:
            self.apply()
        except Exception as e:
            logging.error(f"Migration failed: {e}")
            self.rollback()
            raise e
            # ^ the script must stop with an error code

    def apply(self) -> None:
        logging.info("Start migrations")
        self.executed_migrations = []
        for migration in self.get_planned_migrations():
            self.executed_migrations.append(migration)
            logging.info(f"Migrate {migration.version}: add to the migrations collection")
            self.save(migration)
            logging.info(f"Migrate {migration.version}: apply")
            migration.up()
            logging.info(f"Migrate {migration.version}: validate")
            migration.validate()
            logging.info(f"Migrate {migration.version}: done")
        logging.info("All migrations have been applied")

    def rollback(self) -> None:
        logging.info("Start rollback")
        try:
            while self.executed_migrations:
                migration = self.executed_migrations[-1]
                logging.info(f"Rollback {migration.version}: roll back")
                migration.down()
                logging.info(f"Rollback {migration.version}: removed from the migrations collection")
                self.remove(migration)
                logging.info(f"Rollback {migration.version}: done")
                self.executed_migrations.pop()
            logging.info("All executed migrations have been rolled back")
        except Exception as e:
            logging.error(
                f"Rollback failed: {e}. The database is in an inconsistent state. Try to restore the backup manually."
            )
            raise e

    def save(self, migration: Migration) -> None:
        DatabaseMigration(version=migration.version, description=migration.description).save()

    def remove(self, migration: Migration) -> None:
        DatabaseMigration.objects(version=migration.version).delete()
