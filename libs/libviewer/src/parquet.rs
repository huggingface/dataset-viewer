use bytes::Bytes;
use futures::future::BoxFuture;
use futures::Stream;
use std::ops::Range;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use object_store::path::Path;
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::async_reader::{AsyncFileReader, ParquetObjectReader};
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::ProjectionMask;
use parquet::errors::ParquetError;
use parquet::file::metadata::{
    PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader, ParquetMetaDataWriter,
};

type Result<T, E = ParquetError> = std::result::Result<T, E>;

struct LimitedAsyncReader<T: AsyncFileReader> {
    inner: T,
    scan_size_limit: u64,
}

impl<T: AsyncFileReader> LimitedAsyncReader<T> {
    pub fn new(inner: T, scan_size_limit: u64) -> Self {
        Self {
            inner,
            scan_size_limit,
        }
    }
}

impl<T: AsyncFileReader> AsyncFileReader for LimitedAsyncReader<T> {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
        let num_bytes = range.end - range.start;
        if num_bytes > self.scan_size_limit {
            Box::pin(async move {
                Err(ParquetError::General(format!(
                    "Scan size limit exceeded: attempted to read {} bytes, limit is {} bytes",
                    num_bytes, self.scan_size_limit
                )))
            })
        } else {
            self.inner.get_bytes(range)
        }
    }

    fn get_byte_ranges(&mut self, ranges: Vec<Range<u64>>) -> BoxFuture<'_, Result<Vec<Bytes>>>
    where
        Self: Send,
    {
        let num_bytes: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        if num_bytes > self.scan_size_limit {
            Box::pin(async move {
                Err(ParquetError::General(format!(
                    "Scan size limit exceeded: attempted to read {} bytes, limit is {} bytes",
                    num_bytes, self.scan_size_limit
                )))
            })
        } else {
            self.inner.get_byte_ranges(ranges)
        }
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, Result<Arc<ParquetMetaData>>> {
        self.inner.get_metadata(options)
    }
}

pub async fn read_metadata(
    store: Arc<dyn ObjectStore>,
    path: impl Into<Path>,
) -> Result<Arc<ParquetMetaData>> {
    let path = path.into();
    println!("Reading parquet metadata for {:?} on store {:?}", path, store);

    let mut object_reader = ParquetObjectReader::new(store, path.clone());
    let metadata_reader = ParquetMetaDataReader::new()
        .with_column_index_policy(PageIndexPolicy::Skip)
        .with_offset_index_policy(PageIndexPolicy::Optional);
    // .with_prefetch_hint(16 * 1024);

    // TODO(kszucs): if file_size is known then use load_and_finish
    // let metadata = if let Some(file_size) = self.file_size {
    //     metadata.load_and_finish(self, file_size).await?
    // } else {
    //     metadata.load_via_suffix_and_finish(self).await?
    // };

    let metadata = metadata_reader
        .load_via_suffix_and_finish(&mut object_reader)
        .await?;

    Ok(Arc::new(metadata))
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
    scan_size_limit: u64,
) -> Result<impl Stream<Item = Result<RecordBatch>>> {
    let path = path.into();
    let reader = ParquetObjectReader::new(store, path.clone())
        .with_preload_offset_index(false)
        .with_preload_column_index(false);
    let limited_reader = LimitedAsyncReader::new(reader, scan_size_limit);
    // the page index configuration here shouldn't matter since the metadata is already
    // read and stored in the ParquetFile struct
    let reader_options = ArrowReaderOptions::default().with_page_index(true);
    let reader_metadata = ArrowReaderMetadata::try_new(metadata, reader_options)?;

    // TODO(kszucs): projection pushdown can be handled here if needed
    // let parquet_schema = reader_metadata.metadata().file_metadata().schema_descr();
    // let parquet_fields = ProjectionMask::leaves(parquet_schema, [0]);
    let parquet_fields = ProjectionMask::all();
    let batch_stream =
        ParquetRecordBatchStreamBuilder::new_with_metadata(limited_reader, reader_metadata)
            .with_projection(parquet_fields)
            .with_offset(offset as usize)
            .with_limit(limit as usize)
            .build()?;
    Ok(batch_stream)
}
