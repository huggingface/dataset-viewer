# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
import datetime
from abc import ABC, abstractmethod


class IrreversibleMigrationError(Exception):
    pass


class Migration(ABC):
    def __init__(self, version: str, description: str):
        if version is None or description is None:
            raise ValueError("The version and the description are required.")
        try:
            datetime.datetime.strptime(version, "%Y%m%d%H%M%S")
        except Exception as e:
            raise ValueError("The version should be a string representing a date in the format YYYYMMDDHHMMSS") from e
        self.version = version
        self.description = description

    @abstractmethod
    def up(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def validate(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def down(self) -> None:
        raise IrreversibleMigrationError()
