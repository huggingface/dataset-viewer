# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Any, Literal, Optional, TypedDict

from jsonschema import ValidationError, validate
from libapi.utils import Endpoint, get_response
from libcommon.exceptions import CustomError
from libcommon.operations import delete_dataset, update_dataset
from libcommon.prometheus import StepProfiler
from libcommon.storage_client import StorageClient
from libcommon.utils import Priority
from starlette.requests import Request
from starlette.responses import Response

schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "event": {"type": "string", "enum": ["add", "remove", "update", "move"]},
        "movedTo": {"type": "string"},
        "repo": {
            "type": "object",
            "properties": {
                "headSha": {"type": "string"},
                "name": {"type": "string"},
                "type": {"type": "string", "enum": ["dataset", "model", "space"]},
            },
            "required": ["type", "name"],
        },
    },
    "required": ["event", "repo"],
}


class _MoonWebhookV2PayloadRepo(TypedDict):
    type: Literal["model", "dataset", "space"]
    name: str


class MoonWebhookV2PayloadRepo(_MoonWebhookV2PayloadRepo, total=False):
    headSha: Optional[str]


class MoonWebhookV2Payload(TypedDict):
    """
    Payload from a moon-landing webhook call, v2.
    """

    event: Literal["add", "remove", "update", "move"]
    movedTo: Optional[str]
    repo: MoonWebhookV2PayloadRepo


def parse_payload(json: Any) -> MoonWebhookV2Payload:
    validate(instance=json, schema=schema)
    return json  # type: ignore
    # ^ validate() ensures the content is correct, but does not give the type


def process_payload(
    payload: MoonWebhookV2Payload,
    blocked_datasets: list[str],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> None:
    if payload["repo"]["type"] != "dataset":
        return
    dataset = payload["repo"]["name"]
    if dataset is None:
        return
    event = payload["event"]
    if event == "remove":
        delete_dataset(dataset=dataset, storage_clients=storage_clients)
    elif event in ["add", "update", "move"]:
        delete_dataset(dataset=dataset, storage_clients=storage_clients)
        # ^ delete the old contents (cache + jobs + assets) to avoid mixed content
        new_dataset = (event == "move" and payload["movedTo"]) or dataset
        update_dataset(
            dataset=new_dataset,
            priority=Priority.NORMAL,
            blocked_datasets=blocked_datasets,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            hf_timeout_seconds=hf_timeout_seconds,
            storage_clients=storage_clients,
        )


def create_webhook_endpoint(
    blocked_datasets: list[str],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    hf_webhook_secret: Optional[str] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> Endpoint:
    async def webhook_endpoint(request: Request) -> Response:
        with StepProfiler(method="webhook_endpoint", step="all"):
            with StepProfiler(method="webhook_endpoint", step="get JSON"):
                try:
                    json = await request.json()
                except Exception:
                    content = {"status": "error", "error": "the body could not be parsed as a JSON"}
                    logging.info("/webhook: the body could not be parsed as a JSON.")
                    return get_response(content, 400)
            logging.info(f"/webhook: {json}")
            with StepProfiler(method="webhook_endpoint", step="parse payload and headers"):
                try:
                    payload = parse_payload(json)
                except ValidationError as e:
                    content = {"status": "error", "error": "the JSON payload is invalid"}
                    logging.info(f"/webhook: the JSON body is invalid. JSON: {json}. Error: {e}")
                    return get_response(content, 400)
                except Exception as e:
                    logging.exception("Unexpected error", exc_info=e)
                    content = {"status": "error", "error": "unexpected error"}
                    logging.warning(f"/webhook: unexpected error while parsing the JSON body is invalid. Error: {e}")
                    return get_response(content, 500)

                HEADER = "x-webhook-secret"
                trust_sender = (
                    hf_webhook_secret is not None
                    and (secret := request.headers.get(HEADER)) is not None
                    and secret == hf_webhook_secret
                )
                if not trust_sender:
                    logging.info(f"/webhook: the sender is not trusted. JSON: {json}")
                    return get_response(
                        {"status": "error", "error": "The sender is not trusted. Retry with a valid secret."}, 400
                    )

            with StepProfiler(method="webhook_endpoint", step="process payload"):
                try:
                    process_payload(
                        payload=payload,
                        blocked_datasets=blocked_datasets,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                        hf_timeout_seconds=hf_timeout_seconds,
                        storage_clients=storage_clients,
                    )
                except CustomError as e:
                    content = {"status": "error", "error": "the dataset is not supported"}
                    dataset = payload["repo"]["name"]
                    logging.debug(f"/webhook: the dataset {dataset} is not supported. JSON: {json}. Error: {e}")
                    return get_response(content, 400)
                content = {"status": "ok"}
                return get_response(content, 200)

    return webhook_endpoint
