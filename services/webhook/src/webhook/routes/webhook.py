# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Any, Literal, Optional, TypedDict

from jsonschema import ValidationError, validate
from libapi.utils import Endpoint, get_response
from libcommon.dtos import Priority
from libcommon.exceptions import CustomError
from libcommon.operations import delete_dataset, get_current_revision, smart_update_dataset, update_dataset
from libcommon.prometheus import StepProfiler
from libcommon.storage_client import StorageClient
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
        "scope": {
            "type": "string",
        },
    },
    "required": ["event", "repo", "scope"],
}


class _MoonWebhookV2PayloadRepo(TypedDict):
    type: Literal["model", "dataset", "space"]
    name: str


class MoonWebhookV2PayloadRepo(_MoonWebhookV2PayloadRepo, total=False):
    headSha: Optional[str]


class UpdatedRefDict(TypedDict):
    ref: str
    oldSha: Optional[str]
    newSha: Optional[str]


class MoonWebhookV2Payload(TypedDict):
    """
    Payload from a moon-landing webhook call, v2.
    """

    event: Literal["add", "remove", "update", "move"]
    movedTo: Optional[str]
    repo: MoonWebhookV2PayloadRepo
    scope: str
    updatedRefs: Optional[list[UpdatedRefDict]]


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
    if payload["repo"]["type"] != "dataset" or payload["scope"] not in ("repo", "repo.content", "repo.config"):
        # ^ it filters out the webhook calls for non-dataset repos and discussions in dataset repos
        return None
    dataset = payload["repo"]["name"]
    if dataset is None:
        return None
    event = payload["event"]
    if event == "remove":
        delete_dataset(dataset=dataset, storage_clients=storage_clients)
    elif event in ["add", "update", "move"]:
        if (
            event == "update"
            and get_current_revision(dataset) == payload["repo"]["headSha"]
            and not payload["scope"] == "repo.config"
        ):
            # ^ it filters out the webhook calls when the refs/convert/parquet branch is updated
            logging.warning(
                f"Webhook revision for {dataset} is the same as the current revision in the db - skipping update."
            )
            return None
        revision = payload["repo"].get("headSha")
        old_revision: Optional[str] = None
        for updated_ref in payload.get("updatedRefs") or []:
            ref = updated_ref.get("ref")
            ref_new_sha = updated_ref.get("newSha")
            ref_old_sha = updated_ref.get("oldSha")
            if ref == "refs/heads/main" and isinstance(ref_new_sha, str) and isinstance(ref_old_sha, str):
                if revision != ref_new_sha:
                    logging.warning(
                        f"Unexpected headSha {revision} is different from newSha {ref_new_sha}. Processing webhook payload anyway."
                    )
                revision = ref_new_sha
                old_revision = ref_old_sha
        new_dataset = (event == "move" and payload["movedTo"]) or dataset
        if (
            event == "update" and revision and old_revision and dataset.startswith("datasets-maintainers/")
        ):  # TODO(QL): enable smart updates on more datasets
            try:
                smart_update_dataset(
                    dataset=new_dataset,
                    revision=revision,
                    old_revision=old_revision,
                    blocked_datasets=blocked_datasets,
                    hf_endpoint=hf_endpoint,
                    hf_token=hf_token,
                    hf_timeout_seconds=hf_timeout_seconds,
                    storage_clients=storage_clients,
                )
                return None
            except Exception as err:
                logging.error(f"smart_update_dataset failed with {type(err).__name__}: {err}")
        delete_dataset(dataset=dataset, storage_clients=storage_clients)
        # ^ delete the old contents (cache + jobs + assets) to avoid mixed content
        update_dataset(
            dataset=new_dataset,
            priority=Priority.NORMAL,
            blocked_datasets=blocked_datasets,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            hf_timeout_seconds=hf_timeout_seconds,
            storage_clients=storage_clients,
        )
    return None


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
