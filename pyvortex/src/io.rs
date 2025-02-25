use std::ops::Deref;

use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3::types::PyString;
use tokio::fs::File;
use vortex::file::VortexWriteOptions;
use vortex::stream::ArrayStreamArrayExt;

use crate::arrays::PyArray;
use crate::dataset::{ObjectStoreUrlDataset, TokioFileDataset};
use crate::expr::PyExpr;
use crate::{TOKIO_RUNTIME, install_module};

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "io")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.io", &m)?;

    m.add_function(wrap_pyfunction!(read_url, &m)?)?;
    m.add_function(wrap_pyfunction!(read_path, &m)?)?;
    m.add_function(wrap_pyfunction!(write_path, &m)?)?;

    Ok(())
}

/// Read a vortex struct array from the local filesystem.
///
/// Parameters
/// ----------
/// path : :class:`str`
///     The file path to read from.
/// projection : :class:`list` [ :class:`str` ``|`` :class:`int` ]
///     The columns to read identified either by their index or name.
/// row_filter : :class:`.Expr`
///     Keep only the rows for which this expression evaluates to true.
/// indices : :class:`vortex.Array`
///     The indices of the rows to read.
///
/// Examples
/// --------
///
/// Read an array with a structured column and nulls at multiple levels and in multiple columns.
///
///     >>> import vortex as vx
///     >>> a = vx.array([
///     ...     {'name': 'Joseph', 'age': 25},
///     ...     {'name': None, 'age': 31},
///     ...     {'name': 'Angela', 'age': None},
///     ...     {'name': 'Mikhail', 'age': 57},
///     ...     {'name': None, 'age': None},
///     ... ])
///     >>> vx.io.write_path(a, "a.vortex")
///     >>> b = vx.io.read_path("a.vortex")
///     >>> b.to_arrow_array()
///     <pyarrow.lib.StructArray object at ...>
///     -- is_valid: all not null
///     -- child 0 type: int64
///       [
///         25,
///         31,
///         null,
///         57,
///         null
///       ]
///     -- child 1 type: string_view
///       [
///         "Joseph",
///         null,
///         "Angela",
///         "Mikhail",
///         null
///       ]
///
/// Read just the age column:
///
///     >>> c = vx.io.read_path("a.vortex", projection = ["age"])
///     >>> c.to_arrow_array()
///     <pyarrow.lib.StructArray object at ...>
///     -- is_valid: all not null
///     -- child 0 type: int64
///       [
///         25,
///         31,
///         null,
///         57,
///         null
///       ]
///
///
/// Keep rows with an age above 35. This will read O(N_KEPT) rows, when the file format allows.
///
///     >>> e = vx.io.read_path("a.vortex", row_filter = vx.expr.column("age") > 35)
///     >>> e.to_arrow_array()
///     <pyarrow.lib.StructArray object at ...>
///     -- is_valid: all not null
///     -- child 0 type: int64
///       [
///         57
///       ]
///     -- child 1 type: string_view
///       [
///         "Mikhail"
///       ]
///
/// Read the age column by name, twice, and the name column by index, once:
///
///     >>> # e = vx.io.read_path("a.vortex", projection = ["age", 1, "age"])
///     >>> # e.to_arrow_array()
///     >>> a = vx.array([
///     ...     {'name': 'Joseph', 'age': 25},
///     ...     {'name': None, 'age': 31},
///     ...     {'name': 'Angela', 'age': None},
///     ...     None,
///     ...     {'name': 'Mikhail', 'age': 57},
///     ...     {'name': None, 'age': None},
///     ... ])
///     >>> vx.io.write_path(a, "a.vortex") # doctest: +SKIP
///     >>> # b = vx.io.read_path("a.vortex")
///     >>> # b.to_arrow_array()
#[pyfunction]
#[pyo3(signature = (path, *, projection = None, row_filter = None, indices = None))]
pub fn read_path<'py>(
    path: Bound<'py, PyString>,
    projection: Option<Vec<Bound<'py, PyAny>>>,
    row_filter: Option<&Bound<'py, PyExpr>>,
    indices: Option<&PyArray>,
) -> PyResult<Bound<'py, PyArray>> {
    let dataset = TOKIO_RUNTIME.block_on(TokioFileDataset::try_new(path.extract()?))?;
    dataset.to_array(path.py(), projection, row_filter, indices)
}

/// Read a vortex struct array from a URL.
///
/// .. seealso::
///     :func:`.read_path`
///
/// Parameters
/// ----------
/// url : :class:`str`
///     The URL to read from.
/// projection : :class:`list` [ :class:`str` ``|`` :class:`int` ]
///     The columns to read identified either by their index or name.
/// row_filter : :class:`.Expr`
///     Keep only the rows for which this expression evaluates to true.
///
/// Examples
/// --------
///
/// Read an array from an HTTPS URL:
///
///     >>> import vortex as vx
///     >>> a = vx.io.read_url("https://example.com/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from an S3 URL:
///
///     >>> a = vx.io.read_url("s3://bucket/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from an Azure Blob File System URL:
///
///     >>> a = vx.io.read_url("abfss://my_file_system@my_account.dfs.core.windows.net/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from an Azure Blob Stroage URL:
///
///     >>> a = vx.io.read_url("https://my_account.blob.core.windows.net/my_container/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from a Google Stroage URL:
///
///     >>> a = vx.io.read_url("gs://bucket/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from a local file URL:
///
///     >>> a = vx.io.read_url("file:/path/to/dataset.vortex")  # doctest: +SKIP
///
#[pyfunction]
#[pyo3(signature = (url, *, projection = None, row_filter = None, indices = None))]
pub fn read_url<'py>(
    url: Bound<'py, PyString>,
    projection: Option<Vec<Bound<'py, PyAny>>>,
    row_filter: Option<&Bound<'py, PyExpr>>,
    indices: Option<&PyArray>,
) -> PyResult<Bound<'py, PyArray>> {
    let dataset = TOKIO_RUNTIME.block_on(ObjectStoreUrlDataset::try_new(url.extract()?))?;
    dataset.to_array(url.py(), projection, row_filter, indices)
}

/// Write a vortex struct array to the local filesystem.
///
/// Parameters
/// ----------
/// array : :class:`~vortex.Array`
///     The array. Must be an array of structures.
///
/// f : :class:`str`
///     The file path.
///
/// Examples
/// --------
///
/// Write the array `a` to the local file `a.vortex`.
///
///     >>> import vortex as vx
///     >>> a = vx.array([
///     ...     {'x': 1},
///     ...     {'x': 2},
///     ...     {'x': 10},
///     ...     {'x': 11},
///     ...     {'x': None},
///     ... ])
///     >>> vx.io.write_path(a, "a.vortex")
///
#[pyfunction]
#[pyo3(signature = (array, path))]
pub fn write_path(array: PyArray, path: &str) -> PyResult<()> {
    let array = array.deref().clone();

    TOKIO_RUNTIME.block_on(async move {
        VortexWriteOptions::default()
            .write(File::create(path).await?, array.to_array_stream())
            .await
    })?;

    Ok(())
}
