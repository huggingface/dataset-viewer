mod dataset;
mod parquet;

use arrow::pyarrow::IntoPyArrow;
use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3_async_runtimes;
use tokio;

use crate::dataset::{Dataset, DatasetError};

const DEFAULT_SCAN_SIZE_LIMIT: u64 = 1 * 1024 * 1024 * 1024; // 1 GiB

create_exception!(libviewer, PyDatasetError, pyo3::exceptions::PyException);

impl From<DatasetError> for PyErr {
    fn from(err: DatasetError) -> Self {
        PyDatasetError::new_err(err.to_string())
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
    #[pyo3(signature = (name, files, metadata_store, data_store = None, hf_token=None, revision = None))]
    fn new(
        name: &str,
        files: Vec<IndexedFile>,
        metadata_store: &str,
        data_store: Option<&str>,
        hf_token: Option<&str>,
        revision: Option<&str>,
    ) -> PyResult<Self> {
        let dataset = if let Some(data_store) = data_store {
            Dataset::from_uri(name, files, data_store, metadata_store)?
        } else {
            Dataset::from_hub(name, files, metadata_store, revision, hf_token)?
        };
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
    fn files(&self) -> PyResult<Vec<IndexedFile>> {
        Ok(self.dataset.files.clone())
    }

    #[pyo3(signature = (limit=None, offset=None, scan_size_limit=DEFAULT_SCAN_SIZE_LIMIT))]
    fn sync_scan(
        &self,
        py: Python<'_>,
        limit: Option<u64>,
        offset: Option<u64>,
        scan_size_limit: u64,
    ) -> PyResult<(Vec<Py<PyAny>>, Vec<IndexedFile>)> {
        let rt = tokio::runtime::Runtime::new()?;
        let (record_batches, files_to_index) = rt.block_on(self.dataset.scan(limit, offset, scan_size_limit))?;
        let pyarrow_batches = record_batches
            .into_iter()
            .map(|batch| Ok(batch.into_pyarrow(py)?.unbind()))
            .collect::<PyResult<Vec<_>>>()?;

        Ok((pyarrow_batches, files_to_index))
    }

    #[pyo3(signature = (limit=None, offset=None, scan_size_limit=DEFAULT_SCAN_SIZE_LIMIT))]
    fn scan<'py>(
        &self,
        py: Python<'py>,
        limit: Option<u64>,
        offset: Option<u64>,
        scan_size_limit: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let this = self.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (record_batches, files_to_index) = this.dataset.scan(limit, offset, scan_size_limit).await?;

            let pyarrow_batches = Python::attach(|py| {
                record_batches
                    .into_iter()
                    .map(|batch| Ok(batch.into_pyarrow(py)?.unbind()))
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

#[pymodule]
#[pyo3(name = "_internal")]
fn dv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataset>()?;
    m.add("PyDatasetError", m.py().get_type::<PyDatasetError>())?;
    Ok(())
}
