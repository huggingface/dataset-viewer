from typing import List


from dagster import (
    OpExecutionContext,
    DynamicOut,
    DynamicOutput,
    op,
    asset,
    StaticPartitionsDefinition,
    Definitions,
    graph,
    AssetsDefinition,
)


# this is a static partition key
# we want to replace this with a "runtime" defined list of partitions
# see https://github.com/dagster-io/dagster/issues/7943
partition_keys = [f"dataset{n}" for n in range(10)]


@asset(partitions_def=StaticPartitionsDefinition(partition_keys=partition_keys))
def characters(context: OpExecutionContext) -> List[str]:
    """A silly asset that stores a set of characters.

    The asset is partitioned by dataset. The characters are the characters in the dataset name.

    Args:
        context (OpExecutionContext): the operation context, used to get the partition key

    Returns:
        List[str]: the set of characters in the dataset name (partition key)
    """
    dataset = context.asset_partition_key_for_output()
    return list(set(dataset))


@op(out=DynamicOut())
def fanout_characters(characters: List[str]):
    # for downstream ops
    for character in characters:
        yield DynamicOutput(character, mapping_key=character)


@op()
def character_to_int(character: str) -> int:
    return ord(character[0]) if character else -1


@op()
def fan_in(ints: List[int]) -> List[int]:
    return ints


@graph(name="int_values")
def fanoutandin(characters: List[str]):
    dynamic_characters = fanout_characters(characters)
    results = dynamic_characters.map(character_to_int)
    return fan_in(results.collect())


# the second asset is special: for each partition of the first asset, and for each character, we want to launch an op.
# that's why we use a graph based asset, and the graph is a dynamic graph that consist in:
# - a fanout op that takes the characters and yields them one at a time
# - a map op that takes the character and returns the corresponding int
# - a collect op that collects all the ints
# Note that we also pass the list of partition keys to the graph asset to make sure that one asset is created for each
# partition
int_values = AssetsDefinition.from_graph(
    graph_def=fanoutandin, partitions_def=StaticPartitionsDefinition(partition_keys=partition_keys)
)

defs = Definitions(
    assets=[characters, int_values],
)
