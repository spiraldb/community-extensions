use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::Array;

/// Limit array to start...stop range
pub trait SliceFn<A> {
    /// Return a zero-copy slice of an array, between `start` (inclusive) and `end` (exclusive).
    /// If start >= stop, returns an empty array of the same type as `self`.
    /// Assumes that start or stop are out of bounds, may panic otherwise.
    fn slice(&self, array: &A, start: usize, stop: usize) -> VortexResult<Array>;
}

impl<E: Encoding> SliceFn<Array> for E
where
    E: SliceFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn slice(&self, array: &Array, start: usize, stop: usize) -> VortexResult<Array> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        SliceFn::slice(encoding, array_ref, start, stop)
    }
}

/// Return a zero-copy slice of an array, between `start` (inclusive) and `end` (exclusive).
///
/// # Errors
///
/// Slicing returns an error if you attempt to slice a range that exceeds the bounds of the
/// underlying array.
///
/// Slicing returns an error if the underlying codec's [slice](SliceFn::slice()) implementation
/// returns an error.
pub fn slice(array: impl AsRef<Array>, start: usize, stop: usize) -> VortexResult<Array> {
    let array = array.as_ref();

    if start == 0 && stop == array.len() {
        return Ok(array.clone());
    }
    check_slice_bounds(array, start, stop)?;

    let sliced = array
        .vtable()
        .slice_fn()
        .map(|f| f.slice(array, start, stop))
        .unwrap_or_else(|| {
            Err(vortex_err!(
                NotImplemented: "slice",
                array.encoding()
            ))
        })?;

    debug_assert_eq!(
        sliced.len(),
        stop - start,
        "Slice length mismatch {}",
        array.encoding()
    );
    debug_assert_eq!(
        sliced.dtype(),
        array.dtype(),
        "Slice dtype mismatch {}",
        array.encoding()
    );

    Ok(sliced)
}

fn check_slice_bounds(array: &Array, start: usize, stop: usize) -> VortexResult<()> {
    if start > array.len() {
        vortex_bail!(OutOfBounds: start, 0, array.len());
    }
    if stop > array.len() {
        vortex_bail!(OutOfBounds: stop, 0, array.len());
    }
    if start > stop {
        vortex_bail!("start ({start}) must be <= stop ({stop})");
    }
    Ok(())
}
