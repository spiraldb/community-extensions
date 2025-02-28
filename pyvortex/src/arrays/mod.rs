pub(crate) mod builtins;
pub(crate) mod compressed;
pub(crate) mod fastlanes;
pub(crate) mod from_arrow;
pub(crate) mod py;
pub(crate) mod typed;

use std::ops::Deref;

use arrow::array::{Array as ArrowArray, ArrayRef as ArrowArrayRef};
use arrow::pyarrow::ToPyArrow;
use pyo3::PyClass;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use vortex::arrays::ChunkedArray;
use vortex::arrow::{IntoArrowArray, infer_data_type};
use vortex::compute::{Operator, compare, fill_forward, scalar_at, slice, take};
use vortex::dtype::{DType, PType};
use vortex::error::VortexExpect;
use vortex::mask::Mask;
use vortex::nbytes::NBytes;
use vortex::vtable::EncodingVTable;
use vortex::{Array, ArrayExt, ArrayRef, Encoding};

use crate::arrays::typed::{
    PyBinaryTypeArray, PyBoolTypeArray, PyExtensionTypeArray, PyFloat16TypeArray,
    PyFloat32TypeArray, PyFloat64TypeArray, PyFloatTypeArray, PyInt8TypeArray, PyInt16TypeArray,
    PyInt32TypeArray, PyInt64TypeArray, PyIntTypeArray, PyIntegerTypeArray, PyListTypeArray,
    PyNullTypeArray, PyPrimitiveTypeArray, PyStructTypeArray, PyUInt8TypeArray, PyUInt16TypeArray,
    PyUInt32TypeArray, PyUInt64TypeArray, PyUIntTypeArray, PyUtf8TypeArray,
};
use crate::dtype::PyDType;
use crate::install_module;
use crate::python_repr::PythonRepr;
use crate::scalar::PyScalar;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "arrays")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.arrays", &m)?;

    m.add_class::<PyArray>()?;

    // Canonical encodings
    m.add_class::<builtins::PyConstantArray>()?;
    m.add_class::<builtins::PyChunkedArray>()?;
    m.add_class::<builtins::PyNullArray>()?;
    m.add_class::<builtins::PyBoolArray>()?;
    m.add_class::<builtins::PyPrimitiveArray>()?;
    m.add_class::<builtins::PyVarBinArray>()?;
    m.add_class::<builtins::PyVarBinViewArray>()?;
    m.add_class::<builtins::PyStructArray>()?;
    m.add_class::<builtins::PyListArray>()?;
    m.add_class::<builtins::PyExtensionArray>()?;

    // Compressed encodings
    m.add_class::<compressed::PyAlpArray>()?;
    m.add_class::<compressed::PyAlpRdArray>()?;
    m.add_class::<compressed::PyDateTimePartsArray>()?;
    m.add_class::<compressed::PyDictArray>()?;
    m.add_class::<compressed::PyFsstArray>()?;
    m.add_class::<compressed::PyRunEndArray>()?;
    m.add_class::<compressed::PySparseArray>()?;
    m.add_class::<compressed::PyZigZagArray>()?;

    // Fastlanes encodings
    m.add_class::<fastlanes::PyFastLanesBitPackedArray>()?;
    m.add_class::<fastlanes::PyFastLanesDeltaArray>()?;
    m.add_class::<fastlanes::PyFastLanesFoRArray>()?;

    // Pluggable encodings
    m.add_class::<py::PyEncoding>()?;

    // Typed arrays
    m.add_class::<PyNullTypeArray>()?;
    m.add_class::<PyBoolTypeArray>()?;
    m.add_class::<PyPrimitiveTypeArray>()?;
    m.add_class::<PyIntegerTypeArray>()?;
    m.add_class::<PyUIntTypeArray>()?;
    m.add_class::<PyIntTypeArray>()?;
    m.add_class::<PyFloatTypeArray>()?;
    m.add_class::<PyUInt8TypeArray>()?;
    m.add_class::<PyUInt16TypeArray>()?;
    m.add_class::<PyUInt32TypeArray>()?;
    m.add_class::<PyUInt64TypeArray>()?;
    m.add_class::<PyInt8TypeArray>()?;
    m.add_class::<PyInt16TypeArray>()?;
    m.add_class::<PyInt32TypeArray>()?;
    m.add_class::<PyInt64TypeArray>()?;
    m.add_class::<PyFloat16TypeArray>()?;
    m.add_class::<PyFloat32TypeArray>()?;
    m.add_class::<PyFloat64TypeArray>()?;
    m.add_class::<PyUtf8TypeArray>()?;
    m.add_class::<PyBinaryTypeArray>()?;
    m.add_class::<PyStructTypeArray>()?;
    m.add_class::<PyListTypeArray>()?;
    m.add_class::<PyExtensionTypeArray>()?;

    Ok(())
}

/// An array of zero or more *rows* each with the same set of *columns*.
///
/// Examples
/// --------
///
/// Arrays support all the standard comparison operations:
///
///     >>> import vortex as vx
///     >>> a = vx.array(['dog', None, 'cat', 'mouse', 'fish'])
///     >>> b = vx.array(['doug', 'jennifer', 'casper', 'mouse', 'faust'])
///     >>> (a < b).to_arrow_array()
///     <pyarrow.lib.BooleanArray object at ...>
///     [
///        true,
///        null,
///        false,
///        false,
///        false
///     ]
///     >>> (a <= b).to_arrow_array()
///     <pyarrow.lib.BooleanArray object at ...>
///     [
///        true,
///        null,
///        false,
///        true,
///        false
///     ]
///     >>> (a == b).to_arrow_array()
///     <pyarrow.lib.BooleanArray object at ...>
///     [
///        false,
///        null,
///        false,
///        true,
///        false
///     ]
///     >>> (a != b).to_arrow_array()
///     <pyarrow.lib.BooleanArray object at ...>
///     [
///        true,
///        null,
///        true,
///        false,
///        true
///     ]
///     >>> (a >= b).to_arrow_array()
///     <pyarrow.lib.BooleanArray object at ...>
///     [
///        false,
///        null,
///        true,
///        true,
///        true
///     ]
///     >>> (a > b).to_arrow_array()
///     <pyarrow.lib.BooleanArray object at ...>
///     [
///        false,
///        null,
///        true,
///        false,
///        true
///     ]
#[pyclass(name = "Array", module = "vortex", sequence, subclass, frozen)]
#[derive(Clone)]
pub struct PyArray(ArrayRef);

impl Deref for PyArray {
    type Target = ArrayRef;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PyArray {
    /// Initialize a [`PyArray`] from a Vortex [`ArrayRef`], ensuring we return the correct typed
    /// subclass array.
    pub fn init(py: Python, array: ArrayRef) -> PyResult<Bound<PyArray>> {
        fn unsigned(array: ArrayRef) -> PyClassInitializer<PyUIntTypeArray> {
            PyClassInitializer::from(PyArray(array))
                .add_subclass(PyPrimitiveTypeArray)
                .add_subclass(PyIntegerTypeArray)
                .add_subclass(PyUIntTypeArray)
        }

        fn signed(array: ArrayRef) -> PyClassInitializer<PyIntTypeArray> {
            PyClassInitializer::from(PyArray(array))
                .add_subclass(PyPrimitiveTypeArray)
                .add_subclass(PyIntegerTypeArray)
                .add_subclass(PyIntTypeArray)
        }

        fn float(array: ArrayRef) -> PyClassInitializer<PyFloatTypeArray> {
            PyClassInitializer::from(PyArray(array))
                .add_subclass(PyPrimitiveTypeArray)
                .add_subclass(PyFloatTypeArray)
        }

        match array.dtype() {
            DType::Null => Self::with_subclass(py, array, PyNullTypeArray),
            DType::Bool(_) => Self::with_subclass(py, array, PyBoolTypeArray),
            DType::Primitive(ptype, _) => match ptype {
                PType::U8 => Self::with_subclass_initializer(
                    py,
                    unsigned(array).add_subclass(PyUInt8TypeArray),
                ),
                PType::U16 => Self::with_subclass_initializer(
                    py,
                    unsigned(array).add_subclass(PyUInt16TypeArray),
                ),
                PType::U32 => Self::with_subclass_initializer(
                    py,
                    unsigned(array).add_subclass(PyUInt32TypeArray),
                ),
                PType::U64 => Self::with_subclass_initializer(
                    py,
                    unsigned(array).add_subclass(PyUInt64TypeArray),
                ),
                PType::I8 => {
                    Self::with_subclass_initializer(py, signed(array).add_subclass(PyInt8TypeArray))
                }
                PType::I16 => Self::with_subclass_initializer(
                    py,
                    signed(array).add_subclass(PyInt16TypeArray),
                ),
                PType::I32 => Self::with_subclass_initializer(
                    py,
                    signed(array).add_subclass(PyInt32TypeArray),
                ),
                PType::I64 => Self::with_subclass_initializer(
                    py,
                    signed(array).add_subclass(PyInt64TypeArray),
                ),
                PType::F16 => Self::with_subclass_initializer(
                    py,
                    float(array).add_subclass(PyFloat16TypeArray),
                ),
                PType::F32 => Self::with_subclass_initializer(
                    py,
                    float(array).add_subclass(PyFloat32TypeArray),
                ),
                PType::F64 => Self::with_subclass_initializer(
                    py,
                    float(array).add_subclass(PyFloat64TypeArray),
                ),
            },
            DType::Utf8(_) => Self::with_subclass(py, array, PyUtf8TypeArray),
            DType::Binary(_) => Self::with_subclass(py, array, PyBinaryTypeArray),
            DType::Struct(..) => Self::with_subclass(py, array, PyStructTypeArray),
            DType::List(..) => Self::with_subclass(py, array, PyListTypeArray),
            DType::Extension(_) => Self::with_subclass(py, array, PyExtensionTypeArray),
        }
    }

    /// Initialize a [`PyArray`] with an [`EncodingSubclass`].
    pub fn init_encoding<'py, S: EncodingSubclass>(
        array: Bound<'py, PyArray>,
        encoding: &'static <S as EncodingSubclass>::Encoding,
        subclass: S,
    ) -> PyResult<Bound<'py, S>> {
        if array.get().deref().encoding() != encoding.id() {
            return Err(PyValueError::new_err(format!(
                "Array has encoding {}, expected {}",
                array.get().deref().encoding(),
                encoding.id()
            )));
        }
        Bound::new(
            array.py(),
            PyClassInitializer::from(array.get().clone()).add_subclass(subclass),
        )
    }

    fn with_subclass<S: PyClass<BaseType = PyArray>>(
        py: Python,
        array: ArrayRef,
        subclass: S,
    ) -> PyResult<Bound<PyArray>> {
        Ok(Bound::new(
            py,
            PyClassInitializer::from(PyArray(array)).add_subclass(subclass),
        )?
        .into_any()
        .downcast_into::<PyArray>()?)
    }

    fn with_subclass_initializer<S: PyClass>(
        py: Python,
        intializer: PyClassInitializer<S>,
    ) -> PyResult<Bound<PyArray>> {
        Ok(Bound::new(py, intializer)?
            .into_any()
            .downcast_into::<PyArray>()?)
    }

    pub fn inner(&self) -> &ArrayRef {
        &self.0
    }

    pub fn into_inner(self) -> ArrayRef {
        self.0
    }
}

#[pymethods]
impl PyArray {
    /// Convert a PyArrow object into a Vortex array.
    ///
    /// One of :class:`pyarrow.Array`, :class:`pyarrow.ChunkedArray`, or :class:`pyarrow.Table`.
    ///
    /// Returns
    /// -------
    /// :class:`~vortex.Array`
    #[staticmethod]
    fn from_arrow(obj: Bound<'_, PyAny>) -> PyResult<Bound<'_, PyArray>> {
        from_arrow::from_arrow(&obj)
    }

    /// Convert this array to a PyArrow array.
    ///
    /// Convert this array to an Arrow array.
    ///
    /// .. seealso::
    ///     :meth:`.to_arrow_table`
    ///
    /// Returns
    /// -------
    /// :class:`pyarrow.Array`
    ///
    /// Examples
    /// --------
    ///
    /// Round-trip an Arrow array through a Vortex array:
    ///
    ///     >>> import vortex as vx
    ///     >>> vx.array([1, 2, 3]).to_arrow_array()
    ///     <pyarrow.lib.Int64Array object at ...>
    ///     [
    ///       1,
    ///       2,
    ///       3
    ///     ]
    fn to_arrow_array(self_: PyRef<'_, Self>) -> PyResult<Bound<PyAny>> {
        // NOTE(ngates): for struct arrays, we could also return a RecordBatchStreamReader.
        let py = self_.py();
        let vortex = &self_.0;

        if let Some(chunked_array) = vortex.as_opt::<ChunkedArray>() {
            // We figure out a single Arrow Data Type to convert all chunks into, otherwise
            // the preferred type of each chunk may be different.
            let arrow_dtype = infer_data_type(chunked_array.dtype())?;

            let chunks = chunked_array
                .chunks()
                .iter()
                .map(|chunk| PyResult::Ok(chunk.clone().into_arrow(&arrow_dtype)?))
                .collect::<PyResult<Vec<ArrowArrayRef>>>()?;
            if chunks.is_empty() {
                return Err(PyValueError::new_err("No chunks in array"));
            }

            let pa_data_type = chunks[0].data_type().clone().to_pyarrow(py)?;
            let chunks = chunks
                .iter()
                .map(|arrow_array| arrow_array.into_data().to_pyarrow(py))
                .collect::<Result<Vec<_>, _>>()?;

            let kwargs =
                PyDict::from_sequence(&PyList::new(py, vec![("type", pa_data_type)])?.into_any())?;

            // Combine into a chunked array
            PyModule::import(py, "pyarrow")?.call_method(
                "chunked_array",
                (PyList::new(py, chunks)?,),
                Some(&kwargs),
            )
        } else {
            Ok(vortex
                .clone()
                .into_arrow_preferred()?
                .into_data()
                .to_pyarrow(py)?
                .into_bound(py))
        }
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    /// Returns the encoding ID of this array.
    #[getter]
    fn encoding(&self) -> String {
        self.0.encoding().to_string()
    }

    /// Returns the number of bytes used by this array.
    #[getter]
    fn nbytes(&self) -> usize {
        self.0.nbytes()
    }

    /// Returns the data type of this array.
    ///
    /// Returns
    /// -------
    /// :class:`vortex.DType`
    ///
    /// Examples
    /// --------
    ///
    /// By default, :func:`vortex.array` uses the largest available bit-width:
    ///
    ///     >>> import vortex as vx
    ///     >>> vx.array([1, 2, 3]).dtype
    ///     int(64, nullable=False)
    ///
    /// Including a :obj:`None` forces a nullable type:
    ///
    ///     >>> vx.array([1, None, 2, 3]).dtype
    ///     int(64, nullable=True)
    ///
    /// A UTF-8 string array:
    ///
    ///     >>> vx.array(['hello, ', 'is', 'it', 'me?']).dtype
    ///     utf8(nullable=False)
    #[getter]
    fn dtype(self_: PyRef<Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(self_.py(), self_.0.dtype().clone())
    }

    ///Rust docs are *not* copied into Python for __lt__: https://github.com/PyO3/pyo3/issues/4326
    fn __lt__(&self, other: &Bound<PyArray>) -> PyResult<PyArray> {
        let other = other.borrow();
        let inner = compare(&self.0, &other.0, Operator::Lt)?;
        Ok(PyArray(inner))
    }

    ///Rust docs are *not* copied into Python for __le__: https://github.com/PyO3/pyo3/issues/4326
    fn __le__(&self, other: &Bound<PyArray>) -> PyResult<PyArray> {
        let other = other.borrow();
        let inner = compare(&self.0, &other.0, Operator::Lte)?;
        Ok(PyArray(inner))
    }

    ///Rust docs are *not* copied into Python for __eq__: https://github.com/PyO3/pyo3/issues/4326
    fn __eq__(&self, other: &Bound<PyArray>) -> PyResult<PyArray> {
        let other = other.borrow();
        let inner = compare(&self.0, &other.0, Operator::Eq)?;
        Ok(PyArray(inner))
    }

    ///Rust docs are *not* copied into Python for __ne__: https://github.com/PyO3/pyo3/issues/4326
    fn __ne__(&self, other: &Bound<PyArray>) -> PyResult<PyArray> {
        let other = other.borrow();
        let inner = compare(&self.0, &other.0, Operator::NotEq)?;
        Ok(PyArray(inner))
    }

    ///Rust docs are *not* copied into Python for __ge__: https://github.com/PyO3/pyo3/issues/4326
    fn __ge__(&self, other: &Bound<PyArray>) -> PyResult<PyArray> {
        let other = other.borrow();
        let inner = compare(&self.0, &other.0, Operator::Gte)?;
        Ok(PyArray(inner))
    }

    ///Rust docs are *not* copied into Python for __gt__: https://github.com/PyO3/pyo3/issues/4326
    fn __gt__(&self, other: &Bound<PyArray>) -> PyResult<PyArray> {
        let other = other.borrow();
        let inner = compare(&self.0, &other.0, Operator::Gt)?;
        Ok(PyArray(inner))
    }

    /// Filter an Array by another Boolean array.
    ///
    /// Parameters
    /// ----------
    /// filter : :class:`~vortex.Array`
    ///     Keep all the rows in ``self`` for which the correspondingly indexed row in `filter` is True.
    ///
    /// Returns
    /// -------
    /// :class:`~vortex.Array`
    ///
    /// Examples
    /// --------
    ///
    /// Keep only the single digit positive integers.
    ///
    ///     >>> import vortex as vx
    ///     >>> a = vx.array([0, 42, 1_000, -23, 10, 9, 5])
    ///     >>> filter = vx.array([True, False, False, False, False, True, True])
    ///     >>> a.filter(filter).to_arrow_array()
    ///     <pyarrow.lib.Int64Array object at ...>
    ///     [
    ///       0,
    ///       9,
    ///       5
    ///     ]
    fn filter(&self, mask: &Bound<PyArray>) -> PyResult<PyArray> {
        let mask = mask.borrow();
        let inner = vortex::compute::filter(&self.0, &Mask::try_from(mask.0.as_ref())?)?;
        Ok(PyArray(inner))
    }

    /// Fill forward non-null values over runs of nulls.
    ///
    /// Leading nulls are replaced with the "zero" for that type. For integral and floating-point
    /// types, this is zero. For the Boolean type, this is `:obj:`False`.
    ///
    /// Fill forward sensor values over intermediate missing values. Note that leading nulls are
    /// replaced with 0.0:
    ///
    ///     >>> import vortex as vx
    ///     >>> a = vx.array([
    ///     ...      None,  None, 30.29, 30.30, 30.30,  None,  None, 30.27, 30.25,
    ///     ...     30.22,  None,  None,  None,  None, 30.12, 30.11, 30.11, 30.11,
    ///     ...     30.10, 30.08,  None, 30.21, 30.03, 30.03, 30.05, 30.07, 30.07,
    ///     ... ])
    ///     >>> a.fill_forward().to_arrow_array()
    ///     <pyarrow.lib.DoubleArray object at ...>
    ///     [
    ///       0,
    ///       0,
    ///       30.29,
    ///       30.3,
    ///       30.3,
    ///       30.3,
    ///       30.3,
    ///       30.27,
    ///       30.25,
    ///       30.22,
    ///       ...
    ///       30.11,
    ///       30.1,
    ///       30.08,
    ///       30.08,
    ///       30.21,
    ///       30.03,
    ///       30.03,
    ///       30.05,
    ///       30.07,
    ///       30.07
    ///     ]
    fn fill_forward(&self) -> PyResult<PyArray> {
        let inner = fill_forward(&self.0)?;
        Ok(PyArray(inner))
    }

    /// Retrieve a row by its index.
    ///
    /// Parameters
    /// ----------
    /// index : :class:`int`
    ///     The index of interest. Must be greater than or equal to zero and less than the length of
    ///     this array.
    ///
    /// Returns
    /// -------
    /// :class:`vortex.Scalar`
    ///
    /// Examples
    /// --------
    ///
    /// Retrieve the last element from an array of integers:
    ///
    ///     >>> import vortex as vx
    ///     >>> vx.array([10, 42, 999, 1992]).scalar_at(3).as_py()
    ///     1992
    ///
    /// Retrieve the third element from an array of strings:
    ///
    ///     >>> array = vx.array(["hello", "goodbye", "it", "is"])
    ///     >>> array.scalar_at(2).as_py()
    ///     'it'
    ///
    /// Retrieve an element from an array of structures:
    ///
    ///     >>> array = vx.array([
    ///     ...     {'name': 'Joseph', 'age': 25},
    ///     ...     {'name': 'Narendra', 'age': 31},
    ///     ...     {'name': 'Angela', 'age': 33},
    ///     ...     None,
    ///     ...     {'name': 'Mikhail', 'age': 57},
    ///     ... ])
    ///     >>> array.scalar_at(2).as_py()
    ///     {'age': 33, 'name': 'Angela'}
    ///
    /// Retrieve a missing element from an array of structures:
    ///
    ///     >>> array.scalar_at(3).as_py() is None
    ///     True
    ///
    /// Out of bounds accesses are prohibited:
    ///
    ///     >>> vx.array([10, 42, 999, 1992]).scalar_at(10)
    ///     Traceback (most recent call last):
    ///     ...
    ///     ValueError: index 10 out of bounds from 0 to 4
    ///     ...
    ///
    /// Unlike Python, negative indices are not supported:
    ///
    ///     >>> vx.array([10, 42, 999, 1992]).scalar_at(-2)
    ///     Traceback (most recent call last):
    ///     ...
    ///     OverflowError: can't convert negative int to unsigned
    // TODO(ngates): return a vortex.Scalar
    fn scalar_at(self_: PyRef<'_, Self>, index: usize) -> PyResult<Bound<PyScalar>> {
        PyScalar::init(self_.py(), scalar_at(&self_.0, index)?)
    }

    /// Filter, permute, and/or repeat elements by their index.
    ///
    /// Parameters
    /// ----------
    /// indices : :class:`~vortex.Array`
    ///     An array of indices to keep.
    ///
    /// Returns
    /// -------
    /// :class:`~vortex.Array`
    ///
    /// Examples
    /// --------
    ///
    /// Keep only the first and third elements:
    ///
    ///     >>> a = vx.array(['a', 'b', 'c', 'd'])
    ///     >>> indices = vx.array([0, 2])
    ///     >>> a.take(indices).to_arrow_array()
    ///     <pyarrow.lib.StringArray object at ...>
    ///     [
    ///       "a",
    ///       "c"
    ///     ]
    ///
    /// Permute and repeat the first and second elements:
    ///
    ///     >>> a = vx.array(['a', 'b', 'c', 'd'])
    ///     >>> indices = vx.array([0, 1, 1, 0])
    ///     >>> a.take(indices).to_arrow_array()
    ///     <pyarrow.lib.StringArray object at ...>
    ///     [
    ///       "a",
    ///       "b",
    ///       "b",
    ///       "a"
    ///     ]
    fn take(&self, indices: &Bound<PyArray>) -> PyResult<PyArray> {
        let indices = &indices.borrow().0;

        if !indices.dtype().is_int() {
            return Err(PyValueError::new_err(format!(
                "indices: expected int or uint array, but found: {}",
                indices.dtype().python_repr()
            )));
        }

        let inner = take(&self.0, indices)?;
        Ok(PyArray(inner))
    }

    /// Slice this array.
    ///
    /// Parameters
    /// ----------
    /// start : :class:`int`
    ///     The start index of the range to keep, inclusive.
    ///
    /// end : :class:`int`
    ///     The end index, exclusive.
    ///
    /// Returns
    /// -------
    /// :class:`~vortex.Array`
    ///
    /// Examples
    /// --------
    ///
    /// Keep only the second through third elements:
    ///
    ///     >>> import vortex as vx
    ///     >>> a = vx.array(['a', 'b', 'c', 'd'])
    ///     >>> a.slice(1, 3).to_arrow_array()
    ///     <pyarrow.lib.StringArray object at ...>
    ///     [
    ///       "b",
    ///       "c"
    ///     ]
    ///
    /// Keep none of the elements:
    ///
    ///     >>> a = vx.array(['a', 'b', 'c', 'd'])
    ///     >>> a.slice(3, 3).to_arrow_array()
    ///     <pyarrow.lib.StringViewArray object at ...>
    ///     []
    ///
    /// Unlike Python, it is an error to slice outside the bounds of the array:
    ///
    ///     >>> a = vx.array(['a', 'b', 'c', 'd'])
    ///     >>> a.slice(2, 10).to_arrow_array()
    ///     Traceback (most recent call last):
    ///     ...
    ///     ValueError: index 10 out of bounds from 0 to 4
    ///
    /// Or to slice with a negative value:
    ///
    ///     >>> a = vx.array(['a', 'b', 'c', 'd'])
    ///     >>> a.slice(-2, -1).to_arrow_array()
    ///     Traceback (most recent call last):
    ///     ...
    ///     OverflowError: can't convert negative int to unsigned
    #[pyo3(signature = (start, end))]
    fn slice(&self, start: usize, end: usize) -> PyResult<PyArray> {
        let inner = slice(&self.0, start, end)?;
        Ok(PyArray(inner))
    }

    /// Internal technical details about the encoding of this Array.
    ///
    /// Warnings
    /// --------
    /// The format of the returned string may change without notice.
    ///
    /// Returns
    /// -------
    /// :class:`.str`
    ///
    /// Examples
    /// --------
    ///
    /// Uncompressed arrays have straightforward encodings:
    ///
    ///     >>> import vortex as vx
    ///     >>> arr = vx.array([1, 2, None, 3])
    ///     >>> print(arr.tree_display())
    ///     root: vortex.primitive(i64?, len=4) nbytes=33 B (100.00%)
    ///       metadata: EmptyMetadata
    ///       buffer (align=8): 32 B
    ///       validity: vortex.bool(bool, len=4) nbytes=1 B (3.03%)
    ///         metadata: BoolMetadata { offset: 0 }
    ///         buffer (align=1): 1 B
    ///     <BLANKLINE>
    ///
    /// Compressed arrays often have more complex, deeply nested encoding trees.
    fn tree_display(&self) -> String {
        self.0.tree_display().to_string()
    }
}

/// A marker trait indicating a PyO3 class is a subclass of Vortex `Array`.
pub trait EncodingSubclass: PyClass<BaseType = PyArray> {
    type Encoding: Encoding;
}

/// Unwrap a downcasted Vortex array from a `PyRef<ArraySubclass>`.
pub trait AsArrayRef<T> {
    fn as_array_ref(&self) -> &T;
}

impl<A: EncodingSubclass> AsArrayRef<<A::Encoding as Encoding>::Array> for PyRef<'_, A> {
    fn as_array_ref(&self) -> &<A::Encoding as Encoding>::Array {
        self.as_super()
            .inner()
            .as_any()
            .downcast_ref::<<A::Encoding as Encoding>::Array>()
            .vortex_expect("Failed to downcast array")
    }
}
