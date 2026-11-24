import asyncio
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import libviewer as lv

NUM_FILES = 13
ROWS_PER_FILE = 8


def make_dataset(tmp_path: Path) -> tuple[lv.Dataset, int]:
    """A split of small files with large rows: each file is under the limit, a page of rows is not."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    files = []
    for i in range(NUM_FILES):
        path = data_dir / f"{i:04d}.parquet"
        table = pa.table({"blob": [bytes([i]) * 100_000] * ROWS_PER_FILE})
        pq.write_table(table, path, compression="none")
        files.append(
            {
                "path": path.name,
                "size": path.stat().st_size,
                "num_rows": ROWS_PER_FILE,
                "metadata_path": path.name,
            }
        )
    largest_file = max(f["size"] for f in files)
    dataset = lv.Dataset(
        name="many-files",
        files=files,
        data_store=f"file://{data_dir}",
        metadata_store=f"file://{data_dir}",
    )
    return dataset, largest_file


def scan(dataset: lv.Dataset, **kwargs):
    async def run():
        return await dataset.scan(**kwargs)

    return asyncio.run(run())


def test_the_limit_applies_to_all_the_files_of_a_scan(tmp_path: Path) -> None:
    dataset, largest_file = make_dataset(tmp_path)
    batches, _ = scan(
        dataset, offset=0, limit=ROWS_PER_FILE, scan_size_limit=largest_file
    )
    assert sum(batch.num_rows for batch in batches) == ROWS_PER_FILE
    with pytest.raises(lv.DatasetError, match="Scan size limit exceeded"):
        scan(
            dataset,
            offset=0,
            limit=NUM_FILES * ROWS_PER_FILE,
            scan_size_limit=largest_file,
        )
