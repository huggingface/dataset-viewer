import json
import pickle
import re
from hashlib import sha1
from typing import (
    Mapping,
    Tuple,
    # List,
    Sequence,
)


import dagster._check as check
from dagster import (
    # OpExecutionContext,
    AssetMaterialization,
    DynamicOut,
    DynamicOutput,
    In,
    String,
    job,
    op,
    # asset,
    # DynamicPartitionsDefinition,
    RunRequest,
    sensor,
    Definitions,
    DefaultSensorStatus,
)


# @asset(
#     partitions_def=DynamicPartitionsDefinition(fetch_dataset_ids)
#     # {
#     # }config_schema={"dataset": Field(str, description="Dataset identifier")}
# )
# def characters(context: OpExecutionContext) -> List[str]:
#     # dataset = check.str_param(context.op_config["dataset"], "dataset")
#     dataset = context.asset_partition_key_for_output()

#     return list(dataset)


# @asset(config_schema={"dataset": Field(str, description="Dataset identifier")})
# def characters(context) -> List[str]:
#     dataset = check.str_param(context.op_config["dataset"], "dataset")
#     return list(dataset)


class SuspiciousFileOperation(Exception):
    """A Suspicious filesystem operation was attempted"""

    pass


# copied from django source code
# https://github.com/django/django/blob/0b78ac3fc7bd9f0c57518d0c1a153582318edd59/django/utils/text.py#LL237-L250C13
def get_valid_filename(name):
    """
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    if s in {"", ".", ".."}:
        raise SuspiciousFileOperation(f"Could not derive file name from '{name}'")
    return s


def inputs_to_key(inputs: Mapping[str, str]) -> str:
    hash_suffix = sha1(json.dumps(inputs, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:9]
    prefix = get_valid_filename("-".join(inputs.values()))[:128]
    return f"{prefix}_{hash_suffix}"
    # return get_valid_filename("--".join(inputs.values()))[:128]


def get_asset_key(op_name: str, inputs: Mapping[str, str]) -> str:
    return f"{op_name}.{inputs_to_key(inputs)}"


@op(ins={"dataset": In(String), "commit": In(String)}, out=DynamicOut())
def characters(context, dataset: str, commit: str):
    # config_schema={"dataset": Field(str, description="Dataset identifier")}, out=DynamicOut())
    # def characters(context):  # -> List[str]:
    # dataset = check.str_param(context.op_config["dataset"], "dataset")
    s = f"{dataset}{commit}"

    output = list(s)
    # for downstream ops
    for idx, character in enumerate(output):
        yield DynamicOutput((dataset, character), mapping_key=str(idx))
    # materialize an asset
    asset_key = get_asset_key("characters", {"dataset": dataset, "commit": commit})
    filename = f"/tmp/{asset_key}.pickle"
    with open(filename, "wb") as file:
        pickle.dump(output, file)
    context.log_event(
        AssetMaterialization(
            asset_key=asset_key,
            description=f"Persisted result of characters to storage: {filename}. Inputs: {dataset} {commit}",
        )
    )
    return output


@op()
def int_value(context, tuple: Tuple[str, str]) -> int:
    dataset = check.str_param(tuple[0], "dataset")
    character = check.str_param(tuple[1], "character")
    output = -1 if len(character) == 0 else ord(character[0])
    # if character == "e":
    #     raise ValueError("Something went wrong")
    asset_key = get_asset_key("int_value", {"dataset": dataset, "character": character})
    filename = f"/tmp/{asset_key}.pickle"
    with open(filename, "wb") as file:
        pickle.dump(output, file)
    context.log_event(
        AssetMaterialization(
            asset_key=asset_key,
            description=f"Persisted result of int_value to storage: {filename}. Inputs: {json.dumps(tuple)}",
            metadata={"dataset": dataset, "character": character, "path": filename},
        )
    )
    return output


@job()
def dataset_job():
    cs = characters()
    cs.map(int_value)


# dataset_job.execute_in_process(
#         run_config={"ops": {"characters": {"inputs": {"dataset": {"value": "glue"}}}}}
#     )

N = 100
M = 1


def fetch_dataset_tuples() -> Sequence[Tuple[str, str]]:
    return [(f"dataset{n}", f"commit{m}") for n in range(N) for m in range(M)]


# TODO: another sensor to remove datasets


# TODO: issue: the sensor is disabled while the last job is running. If a dataset blocks the job, everything is blocked
@sensor(job=dataset_job, default_status=DefaultSensorStatus.RUNNING)
def datasets_sensor():
    tuples = fetch_dataset_tuples()
    for dataset, commit in tuples:
        yield RunRequest(
            run_key=json.dumps([dataset, commit]),
            run_config={
                "ops": {"characters": {"inputs": {"dataset": {"value": dataset}, "commit": {"value": commit}}}}
            },
        )


defs = Definitions(
    # assets=[asset_one, asset_two],
    # schedules=[a_schedule],
    sensors=[datasets_sensor],
    jobs=[dataset_job],
    # resources={
    #     "a_resource": some_resource,
    # },
)
