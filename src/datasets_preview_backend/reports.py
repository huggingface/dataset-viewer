import time
from typing import Any, Dict, Optional, TypedDict

import requests

from datasets_preview_backend.types import StatusErrorDict

# class ReportDict(TypedDict):
#     success: bool
#     error: Optional[StatusErrorDict]
#     elapsed_seconds: float


InfoArgs = TypedDict("InfoArgs", {"dataset": str})
# InfoResult = TypedDict("InfoResult", {"info_num_keys": int})


# class InfoReportDict(ReportDict):
#     args: InfoArgs
#     result: Optional[InfoResult]


ConfigsArgs = TypedDict("ConfigsArgs", {"dataset": str})
# ConfigsResult = TypedDict("ConfigsResult", {"configs": List[str]})


# class ConfigsReportDict(ReportDict):
#     args: ConfigsArgs
#     result: Optional[ConfigsResult]


SplitsArgs = TypedDict("SplitsArgs", {"dataset": str, "config": str})
# SplitsResult = TypedDict("SplitsResult", {"splits": List[str]})


# class SplitsReportDict(ReportDict):
#     args: SplitsArgs
#     result: Optional[SplitsResult]


RowsArgs = TypedDict("RowsArgs", {"dataset": str, "config": str, "split": str, "num_rows": int})
# RowsResult = TypedDict("RowsResult", {"rows_length": int})


# class RowsReportDict(ReportDict):
#     args: RowsArgs
#     result: Optional[RowsResult]


# class Report:
#     def __init__(
#         self,
#         elapsed_seconds: float,
#     ):
#         self.elapsed_seconds = elapsed_seconds


# class InfoReport(Report):
#     def __init__(
#         self,
#         args: InfoArgs,
#         response: Optional[InfoDict],
#         error: Optional[StatusErrorDict],
#         elapsed_seconds: float,
#     ):
#         super().__init__(elapsed_seconds=elapsed_seconds)
#         self.args = args
#         self.error = error
#         self.success = self.error is None and response is not None
#         self.result: Optional[InfoResult] = None
#         if response is not None:
#             self.result = {"info_num_keys": len(response["info"])}

#     def to_dict(self) -> InfoReportDict:
#         return {
#             "args": self.args,
#             "error": self.error,
#             "success": self.error is None,
#             "result": self.result,
#             "elapsed_seconds": self.elapsed_seconds,
#         }


# class ConfigsReport(Report):
#     def __init__(
#         self,
#         args: ConfigsArgs,
#         response: Optional[ConfigsDict],
#         error: Optional[StatusErrorDict],
#         elapsed_seconds: float,
#     ):
#         super().__init__(elapsed_seconds=elapsed_seconds)
#         self.args = args
#         self.error = error
#         self.success = self.error is None and response is not None
#         self.result: Optional[ConfigsResult] = None
#         if response is not None:
#             self.result = {"configs": response["configs"]}

#     def to_dict(self) -> ConfigsReportDict:
#         return {
#             "args": self.args,
#             "error": self.error,
#             "success": self.error is None,
#             "result": self.result,
#             "elapsed_seconds": self.elapsed_seconds,
#         }


# class SplitsReport(Report):
#     def __init__(
#         self,
#         args: SplitsArgs,
#         response: Optional[SplitsDict],
#         error: Optional[StatusErrorDict],
#         elapsed_seconds: float,
#     ):
#         super().__init__(elapsed_seconds=elapsed_seconds)
#         self.args = args
#         self.error = error
#         self.success = self.error is None and response is not None
#         self.result: Optional[SplitsResult] = None
#         if response is not None:
#             self.result = {"splits": response["splits"]}

#     def to_dict(self) -> SplitsReportDict:
#         return {
#             "args": self.args,
#             "error": self.error,
#             "success": self.error is None,
#             "result": self.result,
#             "elapsed_seconds": self.elapsed_seconds,
#         }


# class RowsReport(Report):
#     def __init__(
#         self,
#         args: RowsArgs,
#         response: Optional[RowsDict],
#         error: Optional[StatusErrorDict],
#         elapsed_seconds: float,
#     ):
#         super().__init__(elapsed_seconds=elapsed_seconds)
#         self.args = args
#         self.error = error
#         self.success = self.error is None and response is not None
#         self.result: Optional[RowsResult] = None
#         if response is not None:
#             self.result = {"rows_length": len(response["rows"])}

#     def to_dict(self) -> RowsReportDict:
#         return {
#             "args": self.args,
#             "error": self.error,
#             "success": self.error is None,
#             "result": self.result,
#             "elapsed_seconds": self.elapsed_seconds,
#         }


class CommonReportDict(TypedDict):
    url: str
    params: Optional[Dict[str, str]]
    success: bool
    result: Any
    error: Optional[StatusErrorDict]
    elapsed_seconds: float


class CommonReport:
    def __init__(
        self,
        url: str,
        params: Optional[Dict[str, str]],
        response: Any,
        error: Optional[StatusErrorDict],
        elapsed_seconds: float,
    ):
        self.url = url
        self.params = params
        self.error = error
        self.success = self.error is None and response is not None
        self.elapsed_seconds = elapsed_seconds
        self.result = None
        if response is not None:
            # response might be too heavy (we don't want to replicate the cache)
            # we get the essence of the response, depending on the case
            if "info" in response:
                self.result = {"info_num_keys": len(response["info"])}
            elif "configs" in response:
                self.result = {"configs": response["configs"]}
            elif "splits" in response:
                self.result = {"splits": response["splits"]}
            elif "rows" in response:
                self.result = {"rows_length": len(response["rows"])}
            else:
                self.result = {}

    def to_dict(self) -> CommonReportDict:
        return {
            "url": self.url,
            "params": self.params,
            "success": self.error is None,
            "result": self.result,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
        }


def get_report_dict(url: str, endpoint: str, params: Dict[str, str]) -> CommonReportDict:
    t = time.process_time()
    r = requests.get(f"{url}/{endpoint}", params=params)
    try:
        r.raise_for_status()
        response = r.json()
        error = None
    except Exception as err:
        response = None
        if r.status_code in [400, 404]:
            # these error code are managed and return a json we can parse
            error = r.json()
        else:
            error = {
                "exception": type(err).__name__,
                "message": str(err),
                "cause": type(err.__cause__).__name__,
                "cause_message": str(err.__cause__),
                "status_code": r.status_code,
            }
    elapsed_seconds = time.process_time() - t
    return CommonReport(
        url=url, params=params, response=response, error=error, elapsed_seconds=elapsed_seconds
    ).to_dict()
