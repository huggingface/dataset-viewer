from typing import List, Optional, TypedDict

from datasets_preview_backend.types import (
    ConfigsDict,
    InfoDict,
    RowsDict,
    SplitsDict,
    StatusErrorDict,
)


class ReportDict(TypedDict):
    success: bool
    error: Optional[StatusErrorDict]
    elapsed_seconds: float


InfoArgs = TypedDict("InfoArgs", {"dataset": str})
InfoResult = TypedDict("InfoResult", {"info_num_keys": int})


class InfoReportDict(ReportDict):
    args: InfoArgs
    result: Optional[InfoResult]


ConfigsArgs = TypedDict("ConfigsArgs", {"dataset": str})
ConfigsResult = TypedDict("ConfigsResult", {"configs": List[str]})


class ConfigsReportDict(ReportDict):
    args: ConfigsArgs
    result: Optional[ConfigsResult]


SplitsArgs = TypedDict("SplitsArgs", {"dataset": str, "config": str})
SplitsResult = TypedDict("SplitsResult", {"splits": List[str]})


class SplitsReportDict(ReportDict):
    args: SplitsArgs
    result: Optional[SplitsResult]


RowsArgs = TypedDict("RowsArgs", {"dataset": str, "config": str, "split": str, "num_rows": int})
RowsResult = TypedDict("RowsResult", {"rows_length": int})


class RowsReportDict(ReportDict):
    args: RowsArgs
    result: Optional[RowsResult]


class Report:
    def __init__(
        self,
        elapsed_seconds: float,
    ):
        self.elapsed_seconds = elapsed_seconds


class InfoReport(Report):
    def __init__(
        self,
        args: InfoArgs,
        response: Optional[InfoDict],
        error: Optional[StatusErrorDict],
        elapsed_seconds: float,
    ):
        super().__init__(elapsed_seconds=elapsed_seconds)
        self.args = args
        self.error = error
        self.success = self.error is None and response is not None
        self.result: Optional[InfoResult] = None
        if response is not None:
            self.result = {"info_num_keys": len(response["info"])}

    def to_dict(self) -> InfoReportDict:
        return {
            "args": self.args,
            "error": self.error,
            "success": self.error is None,
            "result": self.result,
            "elapsed_seconds": self.elapsed_seconds,
        }


class ConfigsReport(Report):
    def __init__(
        self,
        args: ConfigsArgs,
        response: Optional[ConfigsDict],
        error: Optional[StatusErrorDict],
        elapsed_seconds: float,
    ):
        super().__init__(elapsed_seconds=elapsed_seconds)
        self.args = args
        self.error = error
        self.success = self.error is None and response is not None
        self.result: Optional[ConfigsResult] = None
        if response is not None:
            self.result = {"configs": response["configs"]}

    def to_dict(self) -> ConfigsReportDict:
        return {
            "args": self.args,
            "error": self.error,
            "success": self.error is None,
            "result": self.result,
            "elapsed_seconds": self.elapsed_seconds,
        }


class SplitsReport(Report):
    def __init__(
        self,
        args: SplitsArgs,
        response: Optional[SplitsDict],
        error: Optional[StatusErrorDict],
        elapsed_seconds: float,
    ):
        super().__init__(elapsed_seconds=elapsed_seconds)
        self.args = args
        self.error = error
        self.success = self.error is None and response is not None
        self.result: Optional[SplitsResult] = None
        if response is not None:
            self.result = {"splits": response["splits"]}

    def to_dict(self) -> SplitsReportDict:
        return {
            "args": self.args,
            "error": self.error,
            "success": self.error is None,
            "result": self.result,
            "elapsed_seconds": self.elapsed_seconds,
        }


class RowsReport(Report):
    def __init__(
        self,
        args: RowsArgs,
        response: Optional[RowsDict],
        error: Optional[StatusErrorDict],
        elapsed_seconds: float,
    ):
        super().__init__(elapsed_seconds=elapsed_seconds)
        self.args = args
        self.error = error
        self.success = self.error is None and response is not None
        self.result: Optional[RowsResult] = None
        if response is not None:
            self.result = {"rows_length": len(response["rows"])}

    def to_dict(self) -> RowsReportDict:
        return {
            "args": self.args,
            "error": self.error,
            "success": self.error is None,
            "result": self.result,
            "elapsed_seconds": self.elapsed_seconds,
        }
