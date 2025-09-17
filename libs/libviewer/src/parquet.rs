use futures::Stream;
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
use parquet::file::metadata::ParquetMetaData;
use parquet::file::metadata::ParquetMetaDataWriter;

type Result<T, E = ParquetError> = std::result::Result<T, E>;

pub async fn read_metadata_with_index(
    store: Arc<dyn ObjectStore>,
    path: impl Into<Path>,
) -> Result<Arc<ParquetMetaData>> {
    let path = path.into();
    let metadata = read_metadata(store.clone(), path.clone()).await?;

    // short circuit if offset index is already present
    if metadata.offset_index().is_none() {
        println!("Offset index is not present");
    }

    Ok(metadata)
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
