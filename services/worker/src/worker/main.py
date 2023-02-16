import json
import logging
import os
import sys
import tempfile
import time
from typing import Optional

from filelock import FileLock
from libcommon.log import init_logging
from libcommon.queue import Job, Status, get_datetime
from libcommon.resources import QueueMongoResource
from mirakuru import OutputExecutor

from worker import start_worker_loop
from worker.config import AppConfig
from worker.loop import WorkerState

WORKER_STATE_FILE_NAME = "worker_state.json"
START_WORKER_LOOP_PATH = start_worker_loop.__file__


class WorkerExecutor:
    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config

    def _create_worker_loop_executor(self) -> OutputExecutor:
        banner = self.app_config.worker.state_path
        if not banner:
            raise ValueError("Failed to create the executor because WORKER_STATE_PATH is missing.")
        start_worker_loop_command = [
            sys.executable,
            START_WORKER_LOOP_PATH,
            "--print-worker-state-path",
        ]
        return OutputExecutor(start_worker_loop_command, banner, timeout=10)

    def start(self):
        worker_loop_executor = self._create_worker_loop_executor()
        worker_loop_executor.start()  # blocking until the banner is printed
        logging.info("Starting heartbeat.")
        while worker_loop_executor.running():
            self.heartbeat()
            time.sleep(self.app_config.worker.heartbeat_time_interval_seconds)

    def get_state(self) -> WorkerState:
        worker_state_path = self.app_config.worker.state_path
        if os.path.exists(worker_state_path):
            with FileLock(worker_state_path + ".lock"):
                try:
                    with open(worker_state_path, "r") as worker_state_f:
                        return json.load(worker_state_f)
                except json.JSONDecodeError:
                    return WorkerState(current_job_info=None)
        else:
            return WorkerState(current_job_info=None)

    def get_current_job(self) -> Optional[Job]:
        worker_state = self.get_state()
        if worker_state["current_job_info"]:
            job = Job.objects.with_id(worker_state["current_job_info"]["job_id"])
            if job and job.status == Status.STARTED:
                return job

    def heartbeat(self):
        current_job = self.get_current_job()
        if current_job:
            current_job.update(last_heartbeat=get_datetime())


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "WORKER_STATE_PATH" not in os.environ:
            os.environ["WORKER_STATE_PATH"] = os.path.join(tmp_dir, WORKER_STATE_FILE_NAME)

        app_config = AppConfig.from_env()
        init_logging(log_level=app_config.common.log_level)

        with QueueMongoResource(
            database=app_config.queue.mongo_database, host=app_config.queue.mongo_url
        ) as queue_resource:
            if not queue_resource.is_available():
                raise RuntimeError("The connection to the queue database could not be established. Exiting.")
            worker_executor = WorkerExecutor(app_config)
            worker_executor.start()
