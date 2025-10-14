mod dataset;
mod parquet;

use arrow::pyarrow::IntoPyArrow;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_async_runtimes;
use tokio;

use crate::dataset::{Dataset, DatasetError};

const INDEXING_SIZE_THRESHOLD: i64 = 100 * 1024 * 1024; // 100 MiB

impl From<DatasetError> for PyErr {
    fn from(err: DatasetError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

#[derive(Debug, Clone, IntoPyObject, FromPyObject)]
pub struct IndexedFile {
    #[pyo3(item)]
    pub path: String,
    #[pyo3(item)]
    pub size: Option<u64>,
    #[pyo3(item)]
    pub num_rows: Option<u64>,
    #[pyo3(item)]
    pub metadata_path: String,
}

#[pyclass(subclass)]
#[derive(Debug, Clone)]
struct PyDataset {
    /// The wrapped dataset object
    dataset: Dataset,
}

#[pymethods]
impl PyDataset {
    #[new]
    #[pyo3(signature = (name, files, metadata_store, data_store = "hf://", revision = None, indexing_size_threshold = INDEXING_SIZE_THRESHOLD))]
    fn new(
        name: &str,
        files: Vec<IndexedFile>,
        metadata_store: &str,
        data_store: &str,
        revision: Option<&str>,
        indexing_size_threshold: i64,
    ) -> PyResult<Self> {
        let dataset = Dataset::try_new(
            name,
            files,
            revision,
            data_store,
            metadata_store,
            indexing_size_threshold,
        )?;
        Ok(PyDataset { dataset })
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.dataset))
    }

    #[getter]
    fn name(&self) -> PyResult<&str> {
        Ok(&self.dataset.name)
    }

    #[getter]
    fn revision(&self) -> PyResult<Option<&str>> {
        Ok(self.dataset.revision.as_deref())
    }

    #[getter]
    fn files(&self) -> PyResult<Vec<IndexedFile>> {
        Ok(self.dataset.files.clone())
    }

    #[getter]
    fn data_store_uri(&self) -> PyResult<&str> {
        Ok(&self.dataset.data_store_uri)
    }

    #[getter]
    fn metadata_store_uri(&self) -> PyResult<&str> {
        Ok(&self.dataset.metadata_store_uri)
    }

    #[pyo3(signature = (limit=None, offset=None))]
    fn sync_scan(
        &self,
        py: Python<'_>,
        limit: Option<u64>,
        offset: Option<u64>,
    ) -> PyResult<(Vec<PyObject>, Vec<IndexedFile>)> {
        let rt = tokio::runtime::Runtime::new()?;
        let (record_batches, files_to_index) = rt.block_on(self.dataset.scan(limit, offset))?;
        let pyarrow_batches = record_batches
            .into_iter()
            .map(|batch| batch.into_pyarrow(py))
            .collect::<PyResult<Vec<_>>>()?;
        Ok((pyarrow_batches, files_to_index))
    }

    #[pyo3(signature = (limit=None, offset=None))]
    fn scan<'py>(
        &self,
        py: Python<'py>,
        limit: Option<u64>,
        offset: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let this = self.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (record_batches, files_to_index) = this.dataset.scan(limit, offset).await?;
            let pyarrow_batches = Python::with_gil(|py| {
                record_batches
                    .into_iter()
                    .map(|batch| batch.into_pyarrow(py))
                    .collect::<PyResult<Vec<_>>>()
            })?;
            Ok((pyarrow_batches, files_to_index))
        })
    }

    #[pyo3(signature = (files=None))]
    fn sync_index(&self, files: Option<Vec<IndexedFile>>) -> PyResult<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(self.dataset.index(files.as_deref()))?;
        Ok(())
    }

    #[pyo3(signature = (files=None))]
    fn index<'py>(
        &self,
        py: Python<'py>,
        files: Option<Vec<IndexedFile>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let this = self.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(this.dataset.index(files.as_deref()).await?)
        })
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
#[pyo3(name = "_internal")]
fn dv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataset>()?;
    Ok(())
}
