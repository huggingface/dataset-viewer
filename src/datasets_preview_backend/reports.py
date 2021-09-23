import time
from typing import Any, Dict, Optional, TypedDict

import requests

from datasets_preview_backend.types import StatusErrorDict


class RequestReportDict(TypedDict):
    url: str
    params: Optional[Dict[str, str]]
    success: bool
    result: Any
    error: Optional[StatusErrorDict]
    elapsed_seconds: float


class RequestReport:
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

    def to_dict(self) -> RequestReportDict:
        return {
            "url": self.url,
            "params": self.params,
            "success": self.error is None,
            "result": self.result,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
        }


def get_request_report(url: str, endpoint: str, params: Dict[str, str]) -> RequestReportDict:
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
    return RequestReport(
        url=url, params=params, response=response, error=error, elapsed_seconds=elapsed_seconds
    ).to_dict()
