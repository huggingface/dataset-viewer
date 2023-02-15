from mirakuru import OutputExecutor
import time
import logging
import sys
import tempfile
import os

from filelock import FileLock
from libcommon.log import init_logging
from worker import start_worker_loop
from worker.config import AppConfig

BANNER = "worker_banner"

if __name__ == "__main__":
    app_config = AppConfig.from_env()
    init_logging(log_level=app_config.common.log_level)
    
    
    with tempfile.TemporaryDirectory() as tmp_dir:

        worker_state_path = os.path.join(tmp_dir, 'worker_state.json')

        process = OutputExecutor(
            [
                sys.executable,
                start_worker_loop.__file__,
                "--banner",
                BANNER,
                "--worker_state_path",
                worker_state_path
            ],
            banner=BANNER,
            timeout=10
        )
        process.start()
        logging.warning("Worker process started.")
        
        for _ in range(5):
            assert process.running()
            with FileLock(worker_state_path + ".lock"):
                with open(worker_state_path, 'r') as worker_state_f:
                    logging.warning(f" OMG {worker_state_f.read()}")
            time.sleep(1)

        process.stop()
