use std::sync::Arc;

use arrow::array::RecordBatch;
use futures::future;
use futures::TryStreamExt;
use object_store::prefix::PrefixStore;
use object_store::ObjectStore;
use object_store_opendal::OpendalStore;
use opendal::services::Huggingface;
use opendal::Operator;
use parquet::file::metadata::ParquetMetaData;
use thiserror::Error;
use tokio::task;
use url::Url;

use crate::parquet::{read_batch_stream, read_metadata, write_metadata};
use crate::IndexedFile;

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("Failed to parse URL: {0}")]
    Url(#[from] url::ParseError),

    #[error("Failed to parse object store URL: {0}")]
    OpendalError(#[from] opendal::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Parquet error: {0}")]
    Parquet(#[from] ::parquet::errors::ParquetError),

    #[error("Object store error: {0}")]
    ObjectStore(#[from] object_store::Error),

    #[error("Object store path error: {0}")]
    ObjectStorePath(#[from] object_store::path::Error),

    #[error("Join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),
}

type Result<T, E = DatasetError> = std::result::Result<T, E>;

#[derive(Debug)]
struct ParquetScan {
    file: IndexedFile,
    limit: u64,
    offset: u64,
    metadata: Arc<ParquetMetaData>,
}

impl ParquetScan {
    fn has_offset_index(&self) -> bool {
        self.metadata.offset_index().is_some()
    }

    fn estimate_scan_size(&self) -> i64 {
        let mut scan_size = 0;
        let mut rows_to_skip = self.offset;
        let mut rows_needed = self.limit;

        for row_group in self.metadata.row_groups() {
            let num_rows = row_group.num_rows() as u64;

            // Skip row groups until we reach the offset
            if rows_to_skip >= num_rows {
                rows_to_skip -= num_rows;
                continue;
            }

            // Calculate how many rows to read from this row group
            let rows_in_group = num_rows - rows_to_skip;
            let rows_to_read = rows_needed.min(rows_in_group);

            // Accumulate the size if we need to scan any rows from this row group
            if rows_to_read > 0 {
                scan_size += row_group.compressed_size();
                rows_needed -= rows_to_read;
            }

            // After the first row group, we don't need to skip any more rows
            rows_to_skip = 0;

            // Stop if we've accumulated enough rows
            if rows_needed == 0 {
                break;
            }
        }

        scan_size
    }

    fn shall_index(&self, scan_size_limit: i64) -> bool {
        // TODO(kszucs): reconsider the case when we want to index:
        // 1. file reads with row groups larger than the scan size limit
        // 2. file reads with overall size larger than the scan size limit (multiple row groups)

        // If the file has an offset index, we don't need to index it again
        if self.has_offset_index() {
            return false;
        }
        // If the estimated scan size is larger than the limit, we should index it
        self.estimate_scan_size() > scan_size_limit
    }

    async fn execute(&self, data_store: Arc<dyn ObjectStore>) -> Result<Vec<RecordBatch>> {
        let stream = read_batch_stream(
            data_store,
            self.file.path.clone(),
            self.metadata.clone(),
            self.offset,
            self.limit,
        )?;
        Ok(stream.try_collect::<Vec<_>>().await?)
    }
}

fn store_from_uri(
    uri: &str,
    repo: &str,
    revision: Option<&str>,
) -> Result<Arc<dyn ObjectStore>, DatasetError> {
    let url = Url::parse(uri)?;

    if uri.starts_with("hf://") {
        // You can parse additional parameters from the URI if needed
        let builder = Huggingface::default().repo_type("dataset").repo_id(repo);
        let builder = if let Some(rev) = revision {
            builder.revision(rev)
        } else {
            builder
        };
        let operator = Operator::new(builder)?.finish();
        let store = OpendalStore::new(operator);
        Ok(Arc::new(store))
    } else {
        let (store, prefix) = object_store::parse_url(&url)?;
        let store = PrefixStore::new(store, prefix);
        Ok(Arc::new(store))
    }
}

#[derive(Clone, Debug)]
pub struct Dataset {
    /// The name of the dataset
    pub name: String,
    /// The parquet files in the dataset
    pub files: Vec<IndexedFile>,
    /// Optional revision (branch, tag, commit) for the dataset
    pub revision: Option<String>,
    /// The underlying object store for the dataset
    data_store: Arc<dyn ObjectStore>,
    pub data_store_uri: String,
    /// The local metadata store for the dataset
    metadata_store: Arc<dyn ObjectStore>,
    pub metadata_store_uri: String,
    /// Scan size limit for triggering indexing
    indexing_size_threshold: i64,
}

impl Dataset {
    pub fn try_new(
        name: &str,
        files: Vec<IndexedFile>,
        revision: Option<&str>,
        data_uri: &str,
        metadata_uri: &str,
        indexing_size_threshold: i64,
    ) -> Result<Self> {
        Ok(Self {
            name: name.to_string(),
            files,
            revision: revision.map(|s| s.to_string()),
            data_store: store_from_uri(data_uri, name, revision)?,
            data_store_uri: data_uri.to_string(),
            metadata_store: store_from_uri(metadata_uri, name, revision)?,
            metadata_store_uri: metadata_uri.to_string(),
            indexing_size_threshold,
        })
    }

    async fn plan(&self, limit: Option<u64>, offset: Option<u64>) -> Result<Vec<ParquetScan>> {
        // collect the list of files which must be scanned depending on the number of rows in each file
        // and have a list of (file, offset, limit) tuples which produce the final result after concatenation
        let mut scans = Vec::<ParquetScan>::new();
        let mut current_offset: u64 = 0;

        let offset = offset.unwrap_or(0);
        let mut remaining_limit = limit.unwrap_or(u64::MAX);

        for file in &self.files {
            let (metadata, num_rows) = match file.num_rows {
                Some(rows) => (None, rows),
                None => {
                    let metadata =
                        read_metadata(self.metadata_store.clone(), file.metadata_path.as_ref())
                            .await?;
                    let num_rows = metadata.file_metadata().num_rows() as u64;
                    (Some(metadata), num_rows)
                }
            };

            // here we handle file pruning based on the offset and limit
            if current_offset + num_rows > offset {
                // this file must be scanned, calculate the offset within the file
                let file_offset = if offset > current_offset {
                    offset - current_offset
                } else {
                    0
                };

                // calculate the limit for this file
                let file_limit = remaining_limit.min(num_rows - file_offset);
                remaining_limit -= file_limit;

                let metadata = match metadata {
                    Some(meta) => meta,
                    None => {
                        read_metadata(self.metadata_store.clone(), file.metadata_path.as_ref())
                            .await?
                    }
                };
                scans.push(ParquetScan {
                    file: file.clone(),
                    limit: file_limit,
                    offset: file_offset,
                    metadata: metadata,
                });

                if remaining_limit == 0 {
                    break;
                }
            }

            current_offset += num_rows;
        }
        Ok(scans)
    }

    pub async fn index(&self, files: Option<&[IndexedFile]>) -> Result<()> {
        let tasks = files
            .unwrap_or(&self.files)
            .iter()
            .map(async move |file| {
                let metadata = read_metadata(self.data_store.clone(), file.path.as_ref()).await?;
                write_metadata(
                    metadata,
                    self.metadata_store.clone(),
                    file.metadata_path.as_ref(),
                )
                .await
            })
            .collect::<Vec<_>>();

        future::try_join_all(tasks).await?;

        Ok(())
    }

    pub async fn scan(
        &self,
        limit: Option<u64>,
        offset: Option<u64>,
    ) -> Result<(Vec<RecordBatch>, Vec<IndexedFile>)> {
        // 1. create an object reader for each file in the access plan
        // 2. generate a stream of record batches from each reader
        // 3. flatten the streams into a single stream
        // 4. collect the record batches into a single vector

        let plan = self.plan(limit, offset).await?;
        let files_to_index: Vec<IndexedFile> = plan
            .iter()
            .filter(|scan| scan.shall_index(self.indexing_size_threshold))
            .map(|scan| scan.file.clone())
            .collect();

        let tasks = plan.into_iter().map(|scan| {
            let data_store = self.data_store.clone();
            task::spawn(async move { scan.execute(data_store).await })
        });
        let results = future::try_join_all(tasks).await?;
        let batches = results
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        Ok((batches, files_to_index))
    }
}
