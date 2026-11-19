# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

if __name__ == "__main__":
    # uvicorn spawns its workers by re-running this module: a slow import here delays their liveness pong past
    # the 5s ping timeout of the supervisor, which then kills them
    from webhook.app import start

    start()
