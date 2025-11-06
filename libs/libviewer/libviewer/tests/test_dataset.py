import pytest
import pyarrow as pa
from pathlib import Path
import pyarrow.parquet as pq

from libviewer import Dataset


def generate_sample_table(num_rows: int) -> pa.Table:
    """Generate a sample PyArrow Table for testing.

    Args:
        num_rows: Number of rows to generate

    Returns:
        pa.Table with id, name, value, and category columns
    """
    return pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "name": pa.array([f"name_{i}" for i in range(num_rows)], type=pa.string()),
            "value": pa.array([i * 10.5 for i in range(num_rows)], type=pa.float64()),
            "category": pa.array(
                [f"cat_{i % 5}" for i in range(num_rows)], type=pa.string()
            ),
        }
    )


def write_partitioned_parquet_dataset(
    table: pa.Table, data_dir: Path, metadata_dir: Path, write_page_index: bool = True, num_partitions: int = 5,
) -> None:
    """
    Split table into partitions and write to parquet files with metadata.

    Args:
        table: The PyArrow Table to partition
        data_dir: Directory to write parquet data files
        metadata_dir: Directory to write parquet metadata files
        num_partitions: Number of partitions to create

    Returns:
        List of dict if the written files:
            path: str
            size: int
            num_rows: int
            metadata_path: str
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Split table into partitions
    num_rows = len(table)
    partition_size = num_rows // num_partitions

    files = []
    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size if i < num_partitions - 1 else num_rows
        partition_table = table.slice(start_idx, end_idx - start_idx)

        # Write partition to parquet
        data_path = f"data_partition_{i}.parquet"
        partition_file = data_dir / data_path
        pq.write_table(partition_table, partition_file)

        # Read the parquet metadata
        parquet_metadata = pq.read_metadata(partition_file)

        # Write parquet metadata to a separate file
        metadata_path = f"metadata_partition_{i}.parquet"
        metadata_file = metadata_dir / metadata_path
        with open(metadata_file, "wb") as f:
            parquet_metadata.write_metadata_file(f)

        files.append(
            {
                "path": data_path,
                "size": partition_file.stat().st_size,
                "num_rows": partition_table.num_rows,
                "metadata_path": metadata_path,
            }
        )

    return files


@pytest.mark.parametrize(
    ("limit", "offset"),
    [(0, 0), (1, 0), (10, 5), (20, 15), (150, 180), (100, 900), (250, 750)],
)
@pytest.mark.parametrize("num_partitions", [1, 5, 10])
@pytest.mark.parametrize("with_offset_index", [True, False])
def test_sync_scan(tmp_path, limit, offset, num_partitions, with_offset_index):
    data_dir = tmp_path / "data"
    metadata_dir = tmp_path / "metadata"

    table = generate_sample_table(num_rows=1000)
    files = write_partitioned_parquet_dataset(
        table=table,
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        num_partitions=num_partitions,
        write_page_index=with_offset_index,
    )

    # Calculate expected number of files to be read
    partition_size = 1000 // num_partitions
    if limit == 0:
        expected_files_to_read = 0
    else:
        start_partition = offset // partition_size
        end_partition = (offset + limit - 1) // partition_size
        expected_files_to_read = min(
            end_partition - start_partition + 1, num_partitions
        )

    dataset = Dataset(
        files=files,
        name="test_dataset",
        data_store=f"file://{data_dir}",
        metadata_store=f"file://{metadata_dir}",
    )

    # Perform synchronous scan, the returned batches should match
    # the number of scanned files
    batches = dataset.sync_scan(limit=limit, offset=offset)
    assert len(batches) == expected_files_to_read

    # Concatenate batches and compare with expected sliced table
    result = pa.Table.from_batches(batches, schema=table.schema)
    expected = table.slice(offset, limit)
    assert result.equals(expected)
