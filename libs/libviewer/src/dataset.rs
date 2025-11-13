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
    Opendal(#[from] opendal::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("{0}")]
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
    async fn execute(
        &self,
        data_store: Arc<dyn ObjectStore>,
        scan_size_limit: u64,
    ) -> Result<Vec<RecordBatch>> {
        let stream = read_batch_stream(
            data_store,
            self.file.path.clone(),
            self.metadata.clone(),
            self.offset,
            self.limit,
            scan_size_limit,
        )?;
        Ok(stream.try_collect::<Vec<_>>().await?)
    }
}

fn store_from_uri(uri: &str) -> Result<Arc<dyn ObjectStore>, DatasetError> {
    let url = Url::parse(uri)?;
    let (store, prefix) = object_store::parse_url(&url)?;
    let store = PrefixStore::new(store, prefix);
    Ok(Arc::new(store))
}

#[derive(Clone, Debug)]
pub struct Dataset {
    /// The name of the dataset
    pub name: String,
    /// The parquet files in the dataset
    pub files: Vec<IndexedFile>,
    /// The underlying object store for the dataset
    data_store: Arc<dyn ObjectStore>,
    /// The local metadata store for the dataset
    metadata_store: Arc<dyn ObjectStore>,
}

impl Dataset {
    pub fn from_hub(
        name: &str,
        files: Vec<IndexedFile>,
        metadata_uri: &str,
        revision: Option<&str>,
        hf_token: Option<&str>,
    ) -> Result<Self> {
        // Initialize the data store (Huggingface in this case)
        let mut builder = Huggingface::default().repo_type("dataset").repo_id(name);
        if let Some(token) = hf_token {
            builder = builder.token(token);
        }
        if let Some(rev) = revision {
            builder = builder.revision(rev);
        }
        let operator = Operator::new(builder)?.finish();
        let data_store = Arc::new(OpendalStore::new(operator));

        // Initialize the metadata store from the given URI, usually a local directory
        let metadata_store = store_from_uri(metadata_uri)?;

        Ok(Self {
            name: name.to_string(),
            files,
            data_store,
            metadata_store,
        })
    }

    pub fn from_uri(
        name: &str,
        files: Vec<IndexedFile>,
        data_uri: &str,
        metadata_uri: &str,
    ) -> Result<Self> {
        let data_store = store_from_uri(data_uri)?;
        let metadata_store = store_from_uri(metadata_uri)?;

        Ok(Self {
            name: name.to_string(),
            files,
            data_store,
            metadata_store,
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
                    let metadata = read_metadata(
                        self.metadata_store.clone(),
                        file.metadata_path.as_ref(),
                        None,
                    )
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
                        read_metadata(
                            self.metadata_store.clone(),
                            file.metadata_path.as_ref(),
                            None,
                        )
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
                let metadata =
                    read_metadata(self.data_store.clone(), file.path.as_ref(), None).await?;
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
        scan_size_limit: u64,
    ) -> Result<(Vec<RecordBatch>, Vec<IndexedFile>)> {
        // 1. create an object reader for each file in the access plan
        // 2. generate a stream of record batches from each reader
        // 3. flatten the streams into a single stream
        // 4. collect the record batches into a single vector

        let plan = self.plan(limit, offset).await?;
        let files_to_index = plan
            .iter()
            .filter_map(|scan| {
                if scan.metadata.offset_index().is_none() {
                    Some(scan.file.clone())
                } else {
                    None
                }
            })
            .collect();

        let tasks = plan.into_iter().map(|scan| {
            let data_store = self.data_store.clone();
            task::spawn(async move { scan.execute(data_store, scan_size_limit).await })
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
