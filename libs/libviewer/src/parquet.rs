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
    prefetch_hint: Option<usize>,
) -> Result<Arc<ParquetMetaData>> {
    let path = path.into();

    // configure the metadata reader with optionally reading the offset index if present
    let mut object_reader = ParquetObjectReader::new(store.clone(), path.clone());
    let mut metadata_reader = ParquetMetaDataReader::new()
        .with_column_index_policy(PageIndexPolicy::Skip)
        .with_offset_index_policy(PageIndexPolicy::Optional)
        .with_prefetch_hint(prefetch_hint);

    // first try to read the metadata with offset index but if it fails, retry without
    // reading the offset index; this is necessary because pyarrow writes metadata only
    // parquet files without adjusting offset_index_offset to correspond to the actual
    // location in the newly written metadata only file
    if metadata_reader
        .try_load_via_suffix(&mut object_reader)
        .await
        .is_err()
    {
        // expecting a External(Generic { source: InvalidRange { source: StartTooLarge { .. } } })
        // error here if the offset index is corrupted; retry without reading the offset index
        metadata_reader = metadata_reader.with_offset_index_policy(PageIndexPolicy::Skip);
        metadata_reader
            .try_load_via_suffix(&mut object_reader)
            .await?;
    }
    let metadata = metadata_reader.finish()?;

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
    file_size: u64,
) -> Result<impl Stream<Item = Result<RecordBatch>>> {
    let path = path.into();
    let mut reader = ParquetObjectReader::new(store, path.clone())
        .with_preload_offset_index(false)
        .with_preload_column_index(false);

    // use file_size to ensure that only bounded range requests are used
    reader = reader.with_file_size(file_size);
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

/// Checks if a parquet file is written with content-defined chunking (CDC).
///
/// CDC parquet files have variable-sized pages (not fixed 1MB blocks),
/// which enables more efficient data transfers and storage.
///
/// Detection criteria:
/// 1. Page index must be present (offset_index available)
/// 2. At least one column must have multiple pages with variable sizes
/// 3. Page sizes must vary significantly (not all the same within ~5% tolerance)
///
/// This is more robust than checking for the `content_defined_chunking` metadata key
/// which is only added by the `datasets` library.
pub fn is_content_defined_chunked(metadata: &ParquetMetaData) -> bool {
    // Check if offset index is present (indicates page index was written)
    let offset_index = match metadata.offset_index() {
        Some(idx) => idx,
        None => return false,
    };

    let mut variable_size_columns = 0;

    // Iterate through row groups and columns
    for row_group_offset_index in offset_index.iter() {
        for column_offset_index in row_group_offset_index.iter() {
            let page_locations = column_offset_index.page_locations();

            if page_locations.is_empty() {
                continue;
            }

            // Check if page sizes vary within this column
            if is_variable_page_size(page_locations) {
                variable_size_columns += 1;
            }
        }
    }

    // A file is considered content-defined chunked if:
    // - It has at least one column with variable page sizes
    // - AND at least 50% of columns have variable sizes (to handle edge cases)
    if metadata.num_row_groups() == 0 {
        return false;
    }
    let num_columns = metadata.row_group(0).num_columns();

    if num_columns == 0 {
        return false;
    }

    let ratio = variable_size_columns as f64 / num_columns as f64;
    ratio >= 0.5
}

/// Checks if page sizes vary within a column chunk.
/// Returns true if there's significant variation (> 5% coefficient of variation).
fn is_variable_page_size(
    page_locations: &[parquet::file::page_index::offset_index::PageLocation],
) -> bool {
    if page_locations.len() <= 1 {
        // Single page - can't determine variation
        return false;
    }

    let sizes: Vec<i32> = page_locations
        .iter()
        .map(|pl| pl.compressed_page_size)
        .collect();

    // Calculate average size
    let avg_size = sizes.iter().sum::<i32>() as f64 / sizes.len() as f64;

    if avg_size == 0.0 {
        return false;
    }

    // Calculate coefficient of variation
    let variance = sizes
        .iter()
        .map(|s| ((*s as f64) - avg_size).powi(2))
        .sum::<f64>()
        / sizes.len() as f64;
    let std_dev = variance.sqrt();
    let cv = std_dev / avg_size;

    // If coefficient of variation > 5%, consider it variable
    cv > 0.05
}

/// Checks if a parquet file has content-defined chunking from raw bytes.
///
/// This is a wrapper around `is_content_defined_chunked` that parses the metadata from bytes.
pub fn is_content_defined_chunked_from_bytes(
    metadata_bytes: &[u8],
) -> std::result::Result<bool, ParquetError> {
    // Parse the Parquet metadata using the parquet crate's ParquetMetaDataReader
    let metadata = ParquetMetaDataReader::new()
        .with_offset_index_policy(PageIndexPolicy::Optional)
        .parse_and_finish(&Bytes::from(metadata_bytes.to_vec()))?;

    // Use our existing detection logic
    Ok(is_content_defined_chunked(&metadata))
}

/// Reads parquet metadata from an hf:// URL using opendal's Huggingface backend (XET).
/// Only reads the parquet footer (~8KB) instead of downloading the full file.
pub async fn read_metadata_from_hub(
    path: &str,
    hf_token: Option<&str>,
    hf_endpoint: Option<&str>,
) -> std::result::Result<Arc<ParquetMetaData>, ParquetError> {
    if !path.starts_with("hf://datasets/") {
        return Err(ParquetError::General("Invalid hf:// URL".to_string()));
    }

    let rest = &path["hf://datasets/".len()..];

    // Parse hf://datasets/<repo_id>[@<revision>]/<file_path>
    // where repo_id = <username>/<dataset_name>
    let (repo_id, revision, file_path) = if let Some(at_idx) = rest.find('@') {
        // Has revision: <repo_id>@<revision>/<file_path>
        let repo_id_candidate = &rest[..at_idx];
        if !repo_id_candidate.contains('/') {
            return Err(ParquetError::General(
                "repo_id must be in 'username/dataset_name' format".to_string(),
            ));
        }

        let after_at = &rest[at_idx + 1..];
        let slash_idx = after_at.find('/').ok_or_else(|| {
            ParquetError::General(
                "missing '/' after revision (expected format: <repo_id>@<revision>/<file_path>)"
                    .to_string(),
            )
        })?;

        let rev = &after_at[..slash_idx];
        let fpath = &after_at[slash_idx + 1..];

        if rev.is_empty() || fpath.is_empty() {
            return Err(ParquetError::General(
                "revision and file_path must not be empty".to_string(),
            ));
        }

        (repo_id_candidate, rev, fpath)
    } else {
        // No revision: <repo_id>/<file_path>
        // repo_id contains one '/' so we need the SECOND slash
        let first_slash = rest
            .find('/')
            .ok_or_else(|| ParquetError::General("repo_id is missing".to_string()))?;

        let second_slash = rest[first_slash + 1..]
            .find('/')
            .map(|i| i + first_slash + 1)
            .ok_or_else(|| ParquetError::General("file_path is missing".to_string()))?;

        let repo_id = &rest[..second_slash];
        let file_path = &rest[second_slash + 1..];

        if file_path.is_empty() {
            return Err(ParquetError::General(
                "file_path must not be empty".to_string(),
            ));
        }

        (repo_id, "main", file_path)
    };

    let endpoint = hf_endpoint.unwrap_or("https://huggingface.co");
    let mut builder = opendal::services::Huggingface::default()
        .repo_type("dataset")
        .repo_id(repo_id)
        .revision(revision)
        .endpoint(endpoint);

    if let Some(token) = hf_token {
        builder = builder.token(token);
    }

    let operator = opendal::Operator::new(builder)
        .map_err(|e| ParquetError::General(format!("opendal: {}", e)))?
        .finish();
    let store = Arc::new(object_store_opendal::OpendalStore::new(operator));

    let obj: Path = file_path.into();

    // Use ParquetObjectReader from the parquet crate to handle footer reading via range requests
    let file_size = store
        .head(&obj)
        .await
        .map_err(|e| ParquetError::General(format!("head: {}", e)))?
        .size as u64;
    let mut object_reader = ParquetObjectReader::new(store, obj)
        .with_file_size(file_size)
        .with_preload_offset_index(true);

    // Try loading with offset index, fall back to suffix requests if it fails
    let mut metadata_reader = ParquetMetaDataReader::new()
        .with_column_index_policy(PageIndexPolicy::Skip)
        .with_offset_index_policy(PageIndexPolicy::Optional);

    if metadata_reader
        .try_load_via_suffix(&mut object_reader)
        .await
        .is_err()
    {
        metadata_reader = metadata_reader.with_offset_index_policy(PageIndexPolicy::Skip);
        metadata_reader
            .try_load_via_suffix(&mut object_reader)
            .await?;
    }

    Ok(Arc::new(metadata_reader.finish()?))
}
