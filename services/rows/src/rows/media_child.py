# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import logging
import multiprocessing
import os
import signal
from dataclasses import dataclass
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnProcess
from typing import Any, Optional, cast

import anyio
import pyarrow as pa
from libapi.exceptions import ApiError, TooBigContentError, UnexpectedApiError
from libapi.response import create_response
from libcommon.cloudfront import get_cloudfront_signer
from libcommon.dtos import PaginatedResponse
from libcommon.log import init_logging
from libcommon.prometheus import remove_process_metric_files
from libcommon.storage_client import StorageClient
from libcommon.url_preparator import URLPreparator
from libcommon.utils import get_private_memory_bytes
from prometheus_client import Counter

from rows.config import AppConfig
from rows.memory import CGROUP_MEMORY_MAX, _read_int

# The worker kills the child when it holds this share of the pod memory limit, before the kernel kills the whole pod.
# A pod runs two workers, so two children of this size still leave a third of the pod to the web processes.
MEDIA_CHILD_KILL_MEMORY_SHARE = 1 / 3
# glibc keeps the memory freed by a response: the worker replaces the child when it holds more than this after a
# response, and its memory goes back to the OS. A new child holds about 300 MiB.
MEDIA_CHILD_MAX_IDLE_MEMORY_BYTES = 2**30
MEDIA_CHILD_MEMORY_CHECK_INTERVAL_SECONDS = 0.25
# a child that closed its connection is exiting: wait for its own exit code before killing it
MEDIA_CHILD_EXIT_WAIT_SECONDS = 1.0

ROWS_MEDIA_CHILD_STOPS_TOTAL = Counter(
    "rows_media_child_stops_total",
    "Number of times a rows worker stopped the child process that builds the media of the responses, by reason",
    ["reason"],
)

# The web worker runs threads (uvicorn, anyio, libviewer, Arrow): a forked child could inherit a lock held by one of
# them, and deadlock. A spawned child starts from a fresh interpreter instead.
_SPAWN = multiprocessing.get_context("spawn")


def create_cached_assets_storage_client(app_config: AppConfig) -> StorageClient:
    url_preparator = URLPreparator(
        url_signer=get_cloudfront_signer(cloudfront_config=app_config.cloudfront),
        hf_endpoint=app_config.common.hf_endpoint,
        assets_base_url=app_config.cached_assets.base_url,
    )
    return StorageClient(
        protocol=app_config.cached_assets.storage_protocol,
        storage_root=app_config.cached_assets.storage_root,
        base_url=app_config.cached_assets.base_url,
        s3_config=app_config.s3,
        url_preparator=url_preparator,
    )


def _serve(connection: Connection, app_config: AppConfig) -> None:
    """Build the responses that the worker sends, until it closes the connection."""
    # the child doesn't read the environment or the secrets files: the worker sent its config
    init_logging(level=app_config.log.level)
    storage_client = create_cached_assets_storage_client(app_config)
    while True:
        try:
            params = connection.recv()
            pa_table = pa.ipc.open_stream(connection.recv_bytes()).read_all()
        except EOFError:
            return
        result: Any
        try:
            result = anyio.run(
                partial(
                    create_response,
                    storage_client=storage_client,
                    hf_endpoint=app_config.common.hf_endpoint,
                    hf_token=app_config.common.hf_token,
                    pa_table=pa_table,
                    **params,
                )
            )
        except ApiError as err:
            result = err
        except Exception as err:
            result = UnexpectedApiError("Unexpected error.", err)
        del pa_table
        connection.send(result)
        # the memory check after the response must not count it
        del result


@dataclass
class _Child:
    process: SpawnProcess
    connection: Connection

    @property
    def pid(self) -> int:
        # set once the process is started, which _start does before it returns the child
        return cast(int, self.process.pid)


class MediaChildProcess:
    """A child process of the uvicorn worker, that builds the responses whose features have media, one at a time.

    Building the media of a response (images re-encoded, audio converted, PDF and 3D mesh thumbnails) can take
    gigabytes, and glibc keeps them after the response. In a child, the worker can kill the request that takes too much
    memory, and give the memory back to the OS by replacing the child, without killing the pod.

    It builds one response at a time: the caller must not call `create_response` again before it returns.

    Args:
        app_config (`AppConfig`): the config of the worker, sent to the child.
        kill_memory_bytes (`int`, *optional*): kill the child when its private memory reaches this size. Defaults to a
          third of the pod memory limit, or never if the pod has no limit.
        max_idle_memory_bytes (`int`): replace the child when its private memory reaches this size after a response.
    """

    def __init__(
        self,
        app_config: AppConfig,
        kill_memory_bytes: Optional[int] = None,
        max_idle_memory_bytes: int = MEDIA_CHILD_MAX_IDLE_MEMORY_BYTES,
    ) -> None:
        self.app_config = app_config
        if kill_memory_bytes is None:
            pod_limit = _read_int(CGROUP_MEMORY_MAX)
            kill_memory_bytes = int(pod_limit * MEDIA_CHILD_KILL_MEMORY_SHARE) if pod_limit else None
        self.kill_memory_bytes = kill_memory_bytes
        self.max_idle_memory_bytes = max_idle_memory_bytes
        self._child: Optional[_Child] = None

    async def create_response(self, description: str, pa_table: pa.Table, **params: Any) -> PaginatedResponse:
        """Build the response of `libapi.response.create_response` in the child, with the worker's storage client."""
        return await anyio.to_thread.run_sync(self._create_response, description, pa_table, params)

    def _create_response(self, description: str, pa_table: pa.Table, params: dict[str, Any]) -> PaginatedResponse:
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, pa_table.schema) as writer:
            writer.write_table(pa_table)
        table_bytes = sink.getvalue()
        child, self._child = self._child, None
        if child is not None and not child.process.is_alive():
            self._stop(child, reason="crashed")
            child = None
        if child is None:
            child = self._start()
        result: Any = None
        killed_memory: Optional[int] = None
        try:
            child.connection.send(params)
            child.connection.send_bytes(table_bytes)
            del table_bytes
            while True:
                if child.connection.poll(MEDIA_CHILD_MEMORY_CHECK_INTERVAL_SECONDS):
                    result = child.connection.recv()
                    break
                private_memory = get_private_memory_bytes(child.pid)
                if (
                    self.kill_memory_bytes is not None
                    and private_memory is not None
                    and private_memory >= self.kill_memory_bytes
                ):
                    child.process.kill()
                    killed_memory = private_memory
                    break
        except (EOFError, OSError):
            # the child died: the kernel killed it, or it crashed
            pass
        except BaseException:
            self._stop(child, reason="crashed")
            raise
        if result is None:
            if killed_memory is not None:
                logging.warning(
                    f"rows worker {os.getpid()} killed its media child process {child.pid}: it held"
                    f" {killed_memory // 2**20} MiB of private memory while building the response of {description}"
                )
            child.process.join(MEDIA_CHILD_EXIT_WAIT_SECONDS)
            exit_code = self._stop(child, reason="killed" if killed_memory is not None else "crashed")
            self._child = self._start()
            if exit_code == -signal.SIGKILL:
                raise TooBigContentError(
                    "The media of the rows need more memory than the server allows. Please request fewer rows."
                )
            raise UnexpectedApiError(f"The process that builds the media of the rows crashed (exit code {exit_code}).")
        private_memory = get_private_memory_bytes(child.pid)
        if private_memory is None or private_memory >= self.max_idle_memory_bytes:
            self._stop(child, reason="replaced")
            child = self._start()
        self._child = child
        if isinstance(result, ApiError):
            raise result
        return result  # type: ignore[no-any-return]

    def _start(self) -> _Child:
        connection, child_connection = _SPAWN.Pipe()
        # daemon: the child must not outlive the worker, and multiprocessing would wait for it at the worker's exit
        process = _SPAWN.Process(
            target=_serve, args=(child_connection, self.app_config), name="rows-media", daemon=True
        )
        process.start()
        child_connection.close()
        logging.info(f"rows worker {os.getpid()} started its media child process {process.pid}")
        return _Child(process=process, connection=connection)

    def _stop(self, child: _Child, reason: str) -> Optional[int]:
        child.connection.close()
        child.process.kill()
        child.process.join()
        remove_process_metric_files(child.pid)
        ROWS_MEDIA_CHILD_STOPS_TOTAL.labels(reason=reason).inc()
        return child.process.exitcode

    def stop(self) -> None:
        """Stop the child, when the worker stops."""
        if self._child is not None:
            self._stop(self._child, reason="shutdown")
            self._child = None
