# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Any, Literal, Optional, TypedDict

from jsonschema import ValidationError, validate
from libapi.utils import Endpoint, get_response
from libcommon.exceptions import CustomError
from libcommon.operations import delete_dataset, update_dataset
from libcommon.prometheus import StepProfiler
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
    cache_max_days: int,
    blocked_datasets: list[str],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    trust_sender: bool = False,
) -> None:
    if payload["repo"]["type"] != "dataset":
        return
    dataset = payload["repo"]["name"]
    if dataset is None:
        return
    event = payload["event"]
    if event == "remove":
        # destructive actions (delete, move) require a trusted sender
        if trust_sender:
            delete_dataset(dataset=dataset)
        return
    if event in ["add", "update"]:
        update_dataset(
            dataset=dataset,
            priority=Priority.NORMAL,
            cache_max_days=cache_max_days,
            blocked_datasets=blocked_datasets,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            hf_timeout_seconds=hf_timeout_seconds,
        )
    elif event == "move" and (moved_to := payload["movedTo"]):
        # destructive actions (delete, move) require a trusted sender
        if trust_sender:
            delete_dataset(dataset=dataset)
            update_dataset(
                dataset=moved_to,
                priority=Priority.NORMAL,
                cache_max_days=cache_max_days,
                blocked_datasets=blocked_datasets,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                hf_timeout_seconds=hf_timeout_seconds,
            )


def create_webhook_endpoint(
    cache_max_days: int,
    blocked_datasets: list[str],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    hf_webhook_secret: Optional[str] = None,
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

            with StepProfiler(method="webhook_endpoint", step="process payload"):
                try:
                    process_payload(
                        payload=payload,
                        trust_sender=trust_sender,
                        cache_max_days=cache_max_days,
                        blocked_datasets=blocked_datasets,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                except CustomError as e:
                    content = {"status": "error", "error": "the dataset is not supported"}
                    dataset = payload["repo"]["name"]
                    logging.debug(f"/webhook: the dataset {dataset} is not supported. JSON: {json}. Error: {e}")
                    return get_response(content, 400)
                content = {"status": "ok"}
                return get_response(content, 200)

    return webhook_endpoint
