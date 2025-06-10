use futures::Stream;
use futures::StreamExt;
use std::io::Read;
use std::io::Write;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use object_store::path::Path;
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::ProjectionMask;
use parquet::arrow::{
    arrow_reader::ArrowReaderOptions,
    async_reader::{AsyncFileReader, ParquetObjectReader},
};
use parquet::errors::ParquetError;
use parquet::file::metadata::OffsetIndexBuilder;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::metadata::ParquetMetaDataWriter;
use parquet::file::page_index::offset_index::OffsetIndexMetaData;
use parquet::file::reader::ChunkReader;
use parquet::format::PageHeader;
use parquet::thrift::TSerializable;
use tempfile::NamedTempFile;
use thrift::protocol::TCompactInputProtocol;

type Result<T, E = ParquetError> = std::result::Result<T, E>;

struct TrackedRead<R>(R, usize);

impl<R: Read> Read for TrackedRead<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let v = self.0.read(buf)?;
        self.1 += v;
        Ok(v)
    }
}

/// Reads the page header at `offset` from `reader`, returning
/// both the `PageHeader` and its length in bytes
fn read_page_header(reader: &impl ChunkReader, offset: i64) -> Result<(usize, PageHeader)> {
    let input = reader.get_read(offset as u64)?;
    let mut tracked = TrackedRead(input, 0);
    let mut prot = TCompactInputProtocol::new(&mut tracked);
    let header = PageHeader::read_from_in_protocol(&mut prot)?;
    Ok((tracked.1, header))
}

pub fn calculate_offset_index(
    reader: &impl ChunkReader,
    metadata: &ParquetMetaData,
) -> Result<Vec<Vec<OffsetIndexMetaData>>> {
    let mut result: Vec<Vec<OffsetIndexMetaData>> = vec![vec![]; metadata.num_row_groups()];
    for row_group_idx in 0..metadata.num_row_groups() {
        let row_group = metadata.row_group(row_group_idx);
        for (_index, column) in row_group.columns().iter().enumerate() {
            let mut start = column
                .dictionary_page_offset()
                .unwrap_or_else(|| column.data_page_offset()) as i64;

            let end = start + column.compressed_size();
            let byte_range = column.byte_range();
            let mut builder = OffsetIndexBuilder::new();
            assert_eq!(
                byte_range.0 as i64, start,
                "Byte range start does not match the start of the column"
            );
            assert_eq!(
                start + byte_range.1 as i64,
                end,
                "Byte range end does not match the end of the column"
            );

            while start != end {
                let (header_len, header) = read_page_header(reader, start)?;
                let compressed_page_size = header.compressed_page_size + header_len as i32;
                if let Some(data_page) = header.data_page_header {
                    builder.append_row_count(data_page.num_values as i64);
                    builder.append_offset_and_size(start, compressed_page_size);
                } else if let Some(data_page) = header.data_page_header_v2 {
                    builder.append_row_count(data_page.num_values as i64);
                    builder.append_offset_and_size(start, compressed_page_size);
                }
                start += compressed_page_size as i64;
            }

            let offset_index = builder.build_to_thrift();
            result[row_group_idx].push(OffsetIndexMetaData {
                // ..offset_index
                page_locations: offset_index.page_locations,
                unencoded_byte_array_data_bytes: offset_index.unencoded_byte_array_data_bytes,
            });
        }
    }
    Ok(result)
}

pub async fn read_metadata_with_index(
    store: Arc<dyn ObjectStore>,
    path: impl Into<Path>,
) -> Result<Arc<ParquetMetaData>> {
    let path = path.into();
    let metadata = read_metadata(store.clone(), path.clone()).await?;

    // short circuit if offset index is already present
    if metadata.offset_index().is_some() {
        return Ok(metadata);
    }

    // download the file to local disk into a temporary file
    let mut local_file = NamedTempFile::new()?;
    let mut stream = store.get(&path).await?.into_stream();
    while let Some(bytes) = stream.next().await {
        let chunk = bytes?;
        local_file.write_all(&chunk)?;
    }
    local_file.flush()?;

    // This is blocking so we need to spawn a task
    let metadata_for_task = metadata.clone();
    let offset_index_metadata = tokio::task::spawn_blocking(move || {
        calculate_offset_index(local_file.as_file(), &metadata_for_task)
    })
    .await
    .map_err(|e| ParquetError::General(format!("Join error: {e}")))??;

    let metadata_with_index = Arc::unwrap_or_clone(metadata)
        .into_builder()
        .set_offset_index(Some(offset_index_metadata))
        .build();
    assert!(metadata_with_index.offset_index().is_some());

    Ok(metadata_with_index.into())
}

pub async fn read_metadata(
    store: Arc<dyn ObjectStore>,
    path: impl Into<Path>,
) -> Result<Arc<ParquetMetaData>> {
    let path = path.into();
    let mut reader = ParquetObjectReader::new(store, path.clone()).with_preload_offset_index(true);
    reader.get_metadata(None).await
}

// TODO(kszucs): consider to return with the PutResult's ETag
pub async fn write_metadata(
    metadata: Arc<ParquetMetaData>,
    store: Arc<dyn ObjectStore>,
    path: impl Into<Path>,
) -> Result<()> {
    // Use ParquetMetaDataWriter to serialize the metadata into in-memory buffer
    let mut buffer = Vec::new();
    let writer = ParquetMetaDataWriter::new(&mut buffer, &metadata);
    writer.finish()?;

    // Upload the metadata buffer to the metadata store
    store.put(&path.into(), buffer.into()).await?;
    Ok(())
}

pub fn read_batch_stream(
    store: Arc<dyn ObjectStore>,
    path: impl Into<Path>,
    metadata: Arc<ParquetMetaData>,
    offset: u64,
    limit: u64,
) -> Result<impl Stream<Item = Result<RecordBatch>>> {
    let path = path.into();
    let reader = ParquetObjectReader::new(store, path.clone())
        .with_preload_offset_index(false)
        .with_preload_column_index(false);
    // the page index configuration here shouldn't matter since the metadata is already
    // read and stored in the ParquetFile struct
    let reader_options = ArrowReaderOptions::default().with_page_index(true);
    let reader_metadata = ArrowReaderMetadata::try_new(metadata, reader_options)?;

    // let parquet_schema = reader_metadata.metadata().file_metadata().schema_descr();
    // let parquet_fields = ProjectionMask::leaves(parquet_schema, [0]);
    let parquet_fields = ProjectionMask::all();
    let batch_stream = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, reader_metadata)
        .with_projection(parquet_fields)
        .with_offset(offset as usize)
        .with_limit(limit as usize)
        .build()?;
    Ok(batch_stream)
}
