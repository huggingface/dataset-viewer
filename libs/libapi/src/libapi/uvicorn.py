# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

"""Starting uvicorn so that respawned workers still have their secrets.

uvicorn starts each worker with the "spawn" method, so a worker is a fresh interpreter that re-imports
the app and rebuilds its config from scratch. Its supervisor also restarts a worker that dies, without
touching the container. Put together, a worker that dies once the read window has closed comes back to
blank files and crash-loops, while the container stays up and the window never reopens.

The parent read the secrets while the window was open, so it carries them to every worker it starts,
including the ones it starts hours later. They travel in the Config object uvicorn already pickles to
each worker, never through the environment or the disk.
"""

from collections.abc import Mapping
from typing import Any

import uvicorn
from libcommon.secrets import get_secrets, inherit_secrets
from uvicorn.supervisors import Multiprocess


class _Config(uvicorn.Config):
    """A uvicorn Config that carries the parent's secrets to the worker it is pickled into."""

    def __init__(self, *args: Any, secrets: Mapping[str, str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.secrets = secrets

    def load(self) -> None:
        # Runs in the worker, before the app factory builds any config from the environment.
        inherit_secrets(self.secrets)
        super().load()


def run(app: str, *, host: str, port: int, workers: int, factory: bool = True) -> None:
    """Serve `app`, mirroring uvicorn.run for the cases we use."""
    config = _Config(app, host=host, port=port, workers=workers, factory=factory, secrets=dict(get_secrets()))
    server = uvicorn.Server(config)
    if workers > 1:
        Multiprocess(config, target=server.run, sockets=[config.bind_socket()]).run()
    else:
        server.run()
