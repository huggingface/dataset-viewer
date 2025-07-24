import click
import pathlib

from libviewer import Dataset


@click.group()
@click.argument("dataset", type=str, required=True)
@click.option(
    "--metadata-dir",
    "-m",
    default="data",
    type=pathlib.Path,
    show_default=True,
    help="The output directory to save the dataset's metadata",
)
@click.option(
    "--use-cache",
    "-l",
    is_flag=True,
    default=False,
    help="Load the dataset from the local cache instead of the Hugging Face Hub",
)
@click.option(
    "--download",
    "-d",
    is_flag=True,
    default=False,
    help="Whether to download the dataset files if loading from cache",
)
@click.pass_context
def cli(ctx, dataset, metadata_dir, use_cache, download):
    """Dataset Viewer with parquet page pruning"""
    ctx.ensure_object(dict)
    ctx.obj["dataset_name"] = dataset
    ctx.obj["metadata_dir"] = metadata_dir

    metadata_store = f"file://{metadata_dir.absolute()}/{dataset}"
    if use_cache:
        print("Loading dataset from local cache...")
        ctx.obj["dataset"] = Dataset.from_cache(
            dataset, metadata_store, download=download
        )
    else:
        print("Loading dataset from Hugging Face Hub...")
        ctx.obj["dataset"] = Dataset.from_hub(dataset, metadata_store)


@cli.command()
@click.pass_context
def index(ctx):
    """Download and index parquet files from a dataset."""

    dataset = ctx.obj["dataset"]
    metadata_dir = ctx.obj["metadata_dir"]

    print(f"Indexing dataset '{dataset}' in '{metadata_dir}'")

    # Ensure the metadata directory exists
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("Indexing dataset with offset index enabled...")
    dataset.sync_index()


@cli.command()
@click.option(
    "--offset",
    "-o",
    type=int,
    default=0,
    show_default=True,
    help="The offset to start querying from",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=10,
    show_default=True,
    help="The number of records to query",
)
@click.pass_context
def query(ctx, offset, limit):
    """Query a dataset"""
    print(
        f"Querying dataset '{ctx.obj['dataset_name']}' with offset {offset} and limit {limit}"
    )

    dataset = ctx.obj["dataset"]    
    batches, files_to_index = dataset.sync_scan(limit, offset)

    # Print the result
    for batch in batches:
        print(f"Batch length: {len(batch)}")

    # Print the files that need to be indexed
    if files_to_index:
        print("Files that need to be indexed:")
        for file in files_to_index:
            print(f" - {file}")
    else:
        print("No files need to be indexed.")